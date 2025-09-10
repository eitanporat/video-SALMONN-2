import math

import numpy as np
import torch
from tqdm.auto import tqdm
from dataset_metadata.data_field_name import DataFieldName
from typing import Any, Dict, List
from txt2img.data_loaders.native_dataloader.data_fields_mapper import BatchFieldsMapper
from pqdm.processes import pqdm
from ..data.rope2d import get_rope_index_25
from transformers import WhisperFeatureExtractor

from qwenvl.data.image_processing_qwen2_vl_fast import Qwen2VLImageProcessorFast
import torchaudio
from ltx_video.utils.mel_spectrogram import to_mono

IGNORE_INDEX = -100


def split_into_groups(count, group):
    base = count // group
    remainder = count % group
    return [base + 1] * remainder + [base] * (group - remainder)


def resample_and_downmix(audios, audio_sample_rates, target_sr):
    monos = []

    # print(audios)
    for audio, sr in zip(audios, audio_sample_rates):
        if audio is None:
            monos.append(None)
        else:
            monos.append(
                torchaudio.functional.resample(
                    to_mono(torch.from_numpy(audio)).flatten(),
                    orig_freq=sr,
                    new_freq=target_sr,
                )
            )

    return monos


class QwenVLMapper(BatchFieldsMapper):

    def __init__(self, model_args, data_args, tokenizer):
        self.tokenizer = tokenizer

        self.image_processor = Qwen2VLImageProcessorFast.from_pretrained(
            model_args.model_base
        )

        self.audio_processor = WhisperFeatureExtractor(
            feature_size=data_args.feature_size,  # to do in model args
            sampling_rate=data_args.sampling_rate,
            hop_length=data_args.hop_length,
            chunk_length=data_args.chunk_length,
        )

        self.video_max_frame_pixels = data_args.video_max_frame_pixels
        self.video_min_frame_pixels = data_args.video_min_frame_pixels
        self.video_max_frames = data_args.video_max_frames
        self.sampling_rate = data_args.sampling_rate

        self.num_video_processing_workers = data_args.num_video_processing_workers
        self.base_interval = data_args.base_interval
        self.video_min_frames = data_args.video_min_frames
        self.video_max_frames = data_args.video_max_frames

    @staticmethod
    def from_config(config: Dict[str, Any]) -> "QwenVLMapper":
        return QwenVLMapper(**config)

    def subsample_frames(
        self, batch: Dict[DataFieldName, Any]
    ) -> Dict[DataFieldName, Any]:
        subsampled_videos = []
        batch_dict = batch[DataFieldName.VIDEO_ORIGINAL]

        for frames, average_frame_rate in zip(
            batch_dict["frames"], batch_dict["average_frame_rate"]
        ):
            video_length = len(frames) / average_frame_rate
            num_frames_to_sample = round(video_length / self.base_interval)

            target_frames = np.clip(
                num_frames_to_sample, self.video_min_frames, self.video_max_frames
            )

            frame_idx = np.linspace(0, len(frames) - 1, target_frames, dtype=int)
            frame_idx = np.unique(frame_idx)

            subsampled_video = frames[frame_idx.tolist()]
            subsampled_videos.append(subsampled_video)

        batch[DataFieldName.VIDEO_ORIGINAL]["subsampled_video"] = subsampled_videos

        return batch

    def generate_id_target(self, conversation, thw, audio_lengths):
        T, H, W = thw

        input_id, target = [], []

        for message in conversation:
            role = message["role"]
            content = message["content"]

            if role == "user":
                replacement = "<|vision_start|>"
                if audio_lengths is None:
                    replacement += f"<|video_pad|>" * (
                        T * H * W // self.image_processor.merge_size**2
                    )
                else:
                    per_timestep_audio_len = split_into_groups(audio_lengths, T)
                    for timestep in range(T):
                        replacement += (
                            f"<|video_pad|>"
                            * (H * W // self.image_processor.merge_size**2)
                            + f"<|audio_pad|>" * per_timestep_audio_len[timestep]
                        )
                replacement += "<|vision_end|>"

                content = content.replace("<video>", replacement)

            encode_id = self.tokenizer.apply_chat_template(
                [{"role": role, "content": content}]
            )
            input_id += encode_id

            if role == "user":
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                PREFIX_TOKENS = 2  # <|im_start|>assistant
                SUFFIX_TOKENS = 2  # <|im_end|>\n

                target_mask = encode_id.copy()
                target_mask[:PREFIX_TOKENS] = [IGNORE_INDEX] * PREFIX_TOKENS
                target_mask[-SUFFIX_TOKENS:] = [IGNORE_INDEX] * SUFFIX_TOKENS

                target += target_mask

        return input_id, target

    def preprocess_qwen_2_visual(self, batch) -> Dict:
        chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        self.tokenizer.chat_template = chat_template

        batch["input_ids"], batch["targets"] = [], []

        for conversation, thw, audio_lengths in tqdm(
            zip(
                batch["conversations"], batch["video_grid_thw"], batch["audio_lengths"]
            ),
            total=len(batch["conversations"]),
            desc="Preprocessing conversations",
        ):
            input_id, target = self.generate_id_target(
                conversation, thw.squeeze(0), audio_lengths
            )
            import IPython

            IPython.embed()

            assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
            batch["input_ids"].append(torch.tensor(input_id, dtype=torch.long))
            batch["targets"].append(torch.tensor(target, dtype=torch.long))

        return batch

    def process_video_frames(self, batch):
        video_batch = batch[DataFieldName.VIDEO_ORIGINAL]

        batch["video_tensor"], batch["video_grid_thw"], batch["second_per_grid_ts"] = (
            [],
            [],
            [],
        )

        for subsampled_video, frames, average_frame_rate in tqdm(
            zip(
                video_batch["subsampled_video"],
                video_batch["frames"],
                video_batch["average_frame_rate"],
            ),
            desc="Processing video frames",
            total=len(video_batch["subsampled_video"]),
        ):
            video_length = len(frames) / average_frame_rate
            fps = len(subsampled_video) / video_length
            self.image_processor.max_pixels = self.video_max_frame_pixels * max(
                self.video_max_frames / len(subsampled_video), 1
            )
            self.image_processor.min_pixels = self.video_min_frame_pixels
            self.image_processor.size["longest_edge"] = self.image_processor.max_pixels
            self.image_processor.size["shortest_edge"] = self.image_processor.min_pixels

            video_processed = self.image_processor.preprocess(
                images=None, videos=subsampled_video, return_tensors="pt"
            )

            batch["video_tensor"].append(video_processed["pixel_values_videos"])
            batch["video_grid_thw"].append(video_processed["video_grid_thw"].squeeze(0))
            batch["second_per_grid_ts"].append(
                self.image_processor.temporal_patch_size / fps
            )

        return batch

    def process_audio(self, batch):
        THIRTY_SECONDS = 30 * self.sampling_rate

        batch["audio_inputs"], batch["audio_lengths"] = [], []

        video_batch = batch[DataFieldName.VIDEO_ORIGINAL]

        for audio_data in tqdm(
            resample_and_downmix(
                audios=video_batch["audio"],
                audio_sample_rates=video_batch["audio_sample_rate"],
                target_sr=self.sampling_rate,
            ),
            desc="Processing audios...",
            total=len(video_batch["audio"]),
        ):
            if audio_data is None:
                batch["audio_inputs"].append(None)
                batch["audio_lengths"].append(None)
                continue

            steps = audio_data.shape[0]
            if audio_data.shape[0] < self.sampling_rate:
                padding = self.sampling_rate - steps
                audio_data = torch.nn.functional.pad(
                    audio_data, (0, padding), mode="constant", value=0
                )

            chunks = torch.split(audio_data, THIRTY_SECONDS)

            spectrogram_lst = [
                self.audio_processor(
                    chunk, sampling_rate=self.sampling_rate, return_tensors="pt"
                )["input_features"].squeeze()
                for chunk in chunks
            ]

            batch["audio_inputs"].append(torch.stack(spectrogram_lst, dim=0))
            batch["audio_lengths"].append(
                math.ceil(len(audio_data) / THIRTY_SECONDS) * 60
            )

        return batch

    def get_rope_index_25(self, batch):
        batch["position_ids"] = []
        for input_ids, video_grid_thw, second_per_grid_ts, audio_lengths in zip(
            batch["input_ids"],
            batch["video_grid_thw"],
            batch["second_per_grid_ts"],
            batch["audio_lengths"],
        ):
            position_ids, _ = get_rope_index_25(
                spatial_merge_size=self.image_processor.merge_size,
                input_ids=input_ids.unsqueeze(0),
                image_grid_thw=None,
                video_grid_thw=(
                    video_grid_thw.unsqueeze(0)
                    if video_grid_thw.dim() == 1
                    else video_grid_thw
                ),
                second_per_grid_ts=[second_per_grid_ts] * len(video_grid_thw),
                audio_lengths=[audio_lengths] if audio_lengths is not None else None,
            )

            batch["position_ids"].append(position_ids)

        return batch

    def permute_frames(self, batch):
        # Change from (F, H, W, C) to (F, C, H, W)
        batch[DataFieldName.VIDEO_ORIGINAL]["frames"] = [
            frames.transpose(0, 3, 1, 2)
            for frames in batch[DataFieldName.VIDEO_ORIGINAL]["frames"]
        ]

        return batch

    def __call__(self, batch: Dict[DataFieldName, Any]):
        try:
            # user_prompt = "Please provide a thorough description of all the content in the video, including every detail. As you describe, ensure that you also cover as much information from the audio as possible, and be mindful of the synchronization between the audio and video as you do so."
            batch = self.permute_frames(batch)
            batch = self.subsample_frames(batch)
            batch = self.process_video_frames(batch)
            batch = self.process_audio(batch)

            batch["conversations"] = [
                [
                    {"role": "user", "content": f"<video>"},
                    {"role": "assistant", "content": f"{str(caption)}"},
                ]
                for caption in batch[DataFieldName.GEMINI_STRUCTURED_CAPTIONS]
            ]

            batch = self.preprocess_qwen_2_visual(batch)
            batch = self.get_rope_index_25(batch)
            batch["train_type"] = "sft"
        except Exception as e:
            print(f"Processing failed: {e}")
            raise e

        return batch
