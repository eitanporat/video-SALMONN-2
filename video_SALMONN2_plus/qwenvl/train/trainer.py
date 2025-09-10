# Copyright (2025) Tsinghua University, Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adopted from https://github.com/QwenLM/Qwen2.5-VL. The original license is located at 'third-party-license/qwenvl.txt'.
# Adopted from https://github.com/huggingface/transformers. The original license is located at 'third-party-license/transformers.txt'.

import copy
import os
import sys
import time

import tqdm
import wandb
import logging
import tempfile
import shutil
import traceback
from pathlib import Path
from transformers import TrainerCallback
from transformers.utils import logging as transformers_logging

from txt2img.common import dist_util

# Add parent directory to path for llava imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../"))
from llava import conversation as conversation_lib
from llava.mm_utils import KeywordsStoppingCriteria
from qwenvl.data.dataset import LazySupervisedDataset

logger = transformers_logging.get_logger(__name__)
# from contextlib import contextmanager, nullcontext

import torch
import torch.nn as nn
from transformers import Trainer
from transformers.trainer import (
    ALL_LAYERNORM_LAYERS,
    get_parameter_names,
)
import torch.distributed as dist

from liger_kernel.chunked_loss.dpo_loss import LigerFusedLinearDPOLoss
from qwenvl.data.image_processing_qwen2_vl_fast import Qwen2VLImageProcessorFast


def _is_peft_model(model):
    # if is_peft_available():
    #     classes_to_check = (PeftModel,) if is_peft_available() else ()
    #     # Here we also check if the model is an instance of `PeftMixedModel` introduced in peft>=0.7.0: https://github.com/huggingface/transformers/pull/28321
    #     if version.parse(importlib.metadata.version("peft")) >= version.parse("0.7.0"):
    #         from peft import PeftMixedModel

    #         classes_to_check = (*classes_to_check, PeftMixedModel)
    #     return isinstance(model, classes_to_check)
    return False


class QwenVLTrainer(Trainer):

    def __init__(self, tokenizer=None, eval_args=None, *args, **kwargs):
        # Extract eval_args from kwargs before calling super
        super().__init__(*args, **kwargs)
        self.dpo_loss_fct = LigerFusedLinearDPOLoss()
        self.tokenizer = tokenizer
        self.eval_args = eval_args
        self.eval_iterator = iter(self.eval_dataset)

    @torch.no_grad()
    def evaluate(self, eval_dataset=None, **kwargs):
        self.model.eval()

        batch = next(self.eval_iterator)

        collated_batch = self.data_collator(batch)

        loss, preds, labels = self.prediction_step(
            self.model, collated_batch, prediction_loss_only=False
        )

        metrics = {}
        if loss is not None:
            metrics["eval_loss"] = float(loss.item())

        self.log(metrics)
        self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)

        self._debug_generate_and_log(sample := batch[0])

        return metrics

    @torch.no_grad()
    def _debug_generate_and_log(self, batch):
        if not self.is_world_process_zero() or batch is None:
            return

        tok = self.tokenizer
        video = batch.pop("video_file", None)

        if video is None:
            return

        should_use = batch.pop("should_use", True)
        if not should_use or video is None:
            return

        dev = next(self.model.parameters()).device
        inputs = {
            k: v.clone().to(dev) for k, v in batch.items() if torch.is_tensor(v)
        }  # we don't need labels for generation

        inputs["input_ids"] = (
            inputs["input_ids"].unsqueeze(0)
            if inputs["input_ids"].dim() == 1
            else inputs["input_ids"]
        )

        IGNORE_INDEX = -100
        keep_mask = batch["labels"].ravel() == IGNORE_INDEX
        keep_mask[-2:] = False

        assert batch["input_ids"].shape[0] == batch["position_ids"].shape[2]
        inputs["input_ids"] = batch["input_ids"].unsqueeze(0)[:, keep_mask].to(dev)
        inputs["position_ids"] = batch["position_ids"][:, :, keep_mask].to(dev)

        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.eval_args.max_new_tokens,
                do_sample=self.eval_args.do_sample,
                top_p=self.eval_args.top_p,
                temperature=self.eval_args.temperature,
            )

        self.model.train()

        gen_tokens = (
            out[0, inputs["input_ids"].shape[1] :] if "input_ids" in inputs else out[0]
        )
        pred = tok.decode(
            gen_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        ref = batch["input_ids"][~keep_mask]
        ref = tok.decode(
            ref, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        table = wandb.Table(columns=["video", "reference", "prediction"])
        table.add_data(
            wandb.Video(video, format="mp4"),
            str(ref).strip(),
            str(pred).strip(),
        )

        print("Logged table at step", self.state.global_step)
        print(f"Ref: {str(ref)}\nPred: {str(pred)}")

        self.log({f"eval/table-{self.state.global_step}": table})

    def create_optimizer(self):
        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None and self.args.mm_projector_lr != 0:
                projector_parameters = [
                    name for name, _ in opt_model.named_parameters() if "merger" in name
                ]
                if (
                    self.args.vision_tower_lr is not None
                    and self.args.vision_tower_lr != 0
                ):
                    vision_tower_parameters = [
                        name
                        for name, _ in opt_model.named_parameters()
                        if "visual" in name
                    ]
                    optimizer_grouped_parameters = [
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (
                                    n in decay_parameters
                                    and n not in projector_parameters
                                    and n not in vision_tower_parameters
                                    and p.requires_grad
                                )
                            ],
                            "weight_decay": self.args.weight_decay,
                        },
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (
                                    n in decay_parameters
                                    and n not in projector_parameters
                                    and n in vision_tower_parameters
                                    and p.requires_grad
                                )
                            ],
                            "weight_decay": self.args.weight_decay,
                            "lr": self.args.vision_tower_lr,
                        },
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (
                                    n not in decay_parameters
                                    and n not in projector_parameters
                                    and n not in vision_tower_parameters
                                    and p.requires_grad
                                )
                            ],
                            "weight_decay": 0.0,
                        },
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (
                                    n not in decay_parameters
                                    and n not in projector_parameters
                                    and n in vision_tower_parameters
                                    and p.requires_grad
                                )
                            ],
                            "weight_decay": 0.0,
                            "lr": self.args.vision_tower_lr,
                        },
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (
                                    n in decay_parameters
                                    and n in projector_parameters
                                    and p.requires_grad
                                )
                            ],
                            "weight_decay": self.args.weight_decay,
                            "lr": self.args.mm_projector_lr,
                        },
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (
                                    n not in decay_parameters
                                    and n in projector_parameters
                                    and p.requires_grad
                                )
                            ],
                            "weight_decay": 0.0,
                            "lr": self.args.mm_projector_lr,
                        },
                    ]
                else:
                    optimizer_grouped_parameters = [
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (
                                    n in decay_parameters
                                    and n not in projector_parameters
                                    and p.requires_grad
                                )
                            ],
                            "weight_decay": self.args.weight_decay,
                        },
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (
                                    n not in decay_parameters
                                    and n not in projector_parameters
                                    and p.requires_grad
                                )
                            ],
                            "weight_decay": 0.0,
                        },
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (
                                    n in decay_parameters
                                    and n in projector_parameters
                                    and p.requires_grad
                                )
                            ],
                            "weight_decay": self.args.weight_decay,
                            "lr": self.args.mm_projector_lr,
                        },
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (
                                    n not in decay_parameters
                                    and n in projector_parameters
                                    and p.requires_grad
                                )
                            ],
                            "weight_decay": 0.0,
                            "lr": self.args.mm_projector_lr,
                        },
                    ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                self.args
            )
            self.optimizer = optimizer_cls(
                optimizer_grouped_parameters, **optimizer_kwargs
            )

        return self.optimizer

    def calc_dpo_loss(
        self, policy_input, policy_target, ref_input, ce_loss=None, beta=0.1
    ):
        lm_head = self.model.lm_head.weight
        dpo_loss, (
            chosen_logp,
            reject_logp,
            chosen_logit,
            reject_logit,
            chosen_nll_loss,
            chosen_rewards,
            reject_rewards,
        ) = self.dpo_loss_fct(
            lm_head,
            policy_input,
            policy_target,
            ref_input=ref_input,
            ref_weight=lm_head,
        )
        if ce_loss is not None:
            loss = dpo_loss + beta * ce_loss
        else:
            loss = dpo_loss
        print(
            f"RANK {dist.get_rank()} chosen: {chosen_rewards.item()}, reject: {reject_rewards.item()}"
        )
        return (loss, dpo_loss, chosen_rewards, reject_rewards)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}

        train_type = inputs.get("train_type", "")
        if train_type == "sft":
            start = time.time()
            outputs = model(**inputs)
            end = time.time()
            print(f"Time taken for model forward pass: {end - start} seconds")
        elif train_type == "dpo":
            policy_input, policy_target = model(**inputs)
            with self.accelerator.unwrap_model(self.model).disable_adapter():
                with torch.no_grad():
                    reference_input, reference_target = model(**inputs)
            outputs = self.calc_dpo_loss(policy_input, policy_target, reference_input)

        elif train_type == "gdpo":
            policy_input, policy_target, ce_loss = model(**inputs)
            inputs["train_type"] = "dpo"
            with self.accelerator.unwrap_model(self.model).disable_adapter():
                with torch.no_grad():
                    reference_input, reference_target = model(**inputs)
            outputs = self.calc_dpo_loss(
                policy_input, policy_target, reference_input, ce_loss=ce_loss
            )
        else:
            raise NotImplementedError

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if (
            self.args.average_tokens_across_devices
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
            and num_items_in_batch is not None
        ):
            loss *= self.accelerator.num_processes

        return (loss, outputs) if return_outputs else loss
