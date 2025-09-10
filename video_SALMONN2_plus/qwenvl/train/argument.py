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

import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="/home/eporat/txt2img/txt2img/captioner/video_SALMONN_2/video_SALMONN2_plus/output/models/Qwen2.5-VL-7B-Instruct-Audio"
    )
    model_base: str = field(default="Qwen/Qwen2.5-VL-7B-Instruct")
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)
    tune_mm_audio: bool = field(default=False)
    tune_mm_qformer: bool = field(default=False)
    use_lora: bool = field(default=True)
    lora_r: int = field(default=128)
    lora_alpha: int = field(default=256)
    lora_dropout: float = field(default=0.05)
    lora_bias: str = field(default="none")
    lora_ckpt: str = field(default="tsinghua-ee/video-SALMONN-2_plus_7B")
    model_type: str = field(default="qwen2.5vl")


@dataclass
class DataArguments:
    dataset_use: str = field(default="")
    use_iterator: bool = field(default=True)
    num_train_samples: int = field(default=10000)
    temp_model_dir: str = field(
        default="/home/eporat/txt2img/txt2img/captioner/video-SALMONN-2/video_SALMONN2_plus/output/models/"
    )
    video_max_frames: Optional[int] = field(default=128)
    video_min_frames: Optional[int] = field(default=64)
    base_interval: float = field(default=0.2)
    max_pixels: int = field(default=176400)
    min_pixels: int = field(default=784)
    video_max_frame_pixels: int = field(default=176400)
    video_min_frame_pixels: int = field(default=784)
    run_test: bool = field(default=False)
    train_type: str = field(default="sft")
    feature_size: int = field(default=128)
    chunk_length: int = field(default=30)
    hop_length: int = field(default=160)
    sampling_rate: int = field(default=16000)
    dataset_metadata: List[str] = field(default_factory=lambda: ["mevaseret-v2"])
    eval_dataset_metadata: List[str] = field(default_factory=lambda: ["mevaseret-v2"])
    dataset_region: str = field(default="europe-west4")
    debug_mode: bool = field(default=False)
    loop: bool = field(default=True)
    train_batch_size: int = field(default=1)
    eval_batch_size: int = field(default=1)
    num_eval_samples: int = field(default=1000)
    shuffle_seed: Optional[int] = field(default=42)

    max_temporal_trim: float = field(default=0.0)
    max_spatial_crop: float = field(default=0.0)

    num_decode_workers: int = field(default=4)
    num_download_workers: int = field(default=4)
    num_video_processing_workers: int = field(default=4)

    buffer_size: int = field(default=8)
    thread_count: int = field(default=4)


@dataclass
class EvalArguments:
    """Arguments for evaluation and generation during training."""

    max_new_tokens: int = field(
        default=2048,
        metadata={"help": "Maximum number of tokens to generate during evaluation"},
    )
    top_p: float = field(
        default=0.9, metadata={"help": "Top-p sampling parameter for generation"}
    )
    temperature: float = field(
        default=0.2, metadata={"help": "Temperature for generation sampling"}
    )
    do_sample: bool = field(
        default=False, metadata={"help": "Whether to use sampling during generation"}
    )
    num_sample: int = field(
        default=1, metadata={"help": "Number of samples to generate per input"}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None
    pred_rank: int = field(default=0)
    no_audio: bool = field(default=False)
