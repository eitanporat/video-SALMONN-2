import io
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoTokenizer
from dataset_metadata.data_field_name import DataFieldName
from txt2img.common import dist_util
from txt2img.common.machine import AccelerationType
from txt2img.config.data import DatasetConfig, DatasetFieldConfig
from txt2img.data_loaders.native_dataloader.native_dataloader_factory import (
    DataloaderFactory,
)
from torch.utils.data import Dataset
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
)
from txt2img.config.data import (
    DatasetConfig,
    DatasetFieldConfig,
    DatasetFieldMapperConfig,
    NativeDataloaderMediaDecoderConfig,
    FieldsMapperName,
)


def map_batch_to_sample(batch):
    videos = batch[DataFieldName.VIDEO_ORIGINAL]
    captions = batch[DataFieldName.AUDIO_VISUAL_CAPTION]

    sample = {
        "video": [io.BytesIO(v) for v in videos],
        "conversations": [
            {
                "from": "human",
                "value": f"<video>",
            },
            {
                "from": "gpt",
                "value": "\n".join([str(c) or "" for c in captions]),
            },
        ],
        "caption": captions,
        "should_use": True,
        "use_audio": True,
    }

    return sample


def wrap_dataloader_as_qwen_sources(dataloader):
    while True:
        batch = next(dataloader)

        sample = map_batch_to_sample(batch)

        for key in batch:
            if key not in sample:
                sample[key] = batch[key]

        yield sample


import threading
import queue


class PrefetchIterator:
    def __init__(self, it, buffer_size: int, thread_count: int):
        self._it = iter(it)
        self._q = queue.Queue(maxsize=buffer_size)
        self._sentinel = object()
        self._thread_count = thread_count

        self._threads = []

        for _ in range(self._thread_count):
            self._threads.append(threading.Thread(target=self._worker, daemon=True))

        for thread in self._threads:
            thread.start()

    def _worker(self):
        try:
            for item in self._it:
                self._q.put(item)
        finally:
            self._q.put(self._sentinel)

    def __iter__(self):
        return self

    def __next__(self):
        item = self._q.get()
        if item is self._sentinel:
            raise StopIteration
        return item

    def __getattribute__(self, name: str):
        if name in (
            "_it",
            "_q",
            "_sentinel",
            "_t",
            "_worker",
            "__iter__",
            "__next__",
            "_threads",
            "_thread_count",
        ):
            return object.__getattribute__(self, name)
        try:
            return getattr(self._it, name)
        except AttributeError:
            raise


def make_qwen_lazy_iterator(
    batch_size: int, *, debug_mode: bool = False, loop: bool = False
):
    try:
        dist_util.init(acceleration_type=AccelerationType.GPU, devices_per_vm=1)
    except:
        pass

    dataset_config = DatasetConfig(
        dataset_metadata=["mevaseret-v2"],
        region="europe-west4",
        fields=[
            DatasetFieldConfig(name=DataFieldName.VIDEO_ORIGINAL),
            DatasetFieldConfig(name=DataFieldName.AUDIO_VISUAL_CAPTION),
        ],
        mappers=[],
        debug_mode=debug_mode,
    )

    dataloader = DataloaderFactory.create_data_loader(
        dataset_config=dataset_config,
        loop=loop,
        batch_size=batch_size,
    )

    dataloader = dist_util.BackgroundDeviceLoader(
        dataloader=dataloader,
        pl_kwargs={
            "batches_per_execution": 1,
            "loader_prefetch_size": 1,
            "device_prefetch_size": 1,
            "host_to_device_transfer_threads": 1,
        },
    )

    iterator = iter(dataloader)
    return wrap_dataloader_as_qwen_sources(iterator)


class QwenDataset(Dataset):
    def __init__(
        self,
        model_args: ModelArguments,
        data_args: DataArguments,
        run_prefetch_iterator: bool = False,
        mode: str = "train",
        convert_to_collator_instances: bool = True,
        dtype: torch.dtype = torch.float32,
        download_video: bool = False,
        **kwargs,
    ):
        super(QwenDataset, self).__init__()

        self.dtype = dtype

        print(f"DEBUG MODE: {data_args.debug_mode}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.lora_ckpt
            if model_args.lora_ckpt
            else model_args.model_name_or_path
        )

        fields = [
            DatasetFieldConfig(name=DataFieldName.AUDIO_VISUAL_CAPTION),
            DatasetFieldConfig(name=DataFieldName.GEMINI_STRUCTURED_CAPTIONS),
        ]

        self.run_decoder = kwargs.get("run_decoder", True)
        self.run_mapper = kwargs.get("run_mapper", True)
        self.convert_to_collator_instances = convert_to_collator_instances
        self.download_video = download_video
        self.video_cache_dir = Path("downloaded_videos") if download_video else None
        if self.download_video and self.video_cache_dir:
            self.video_cache_dir.mkdir(exist_ok=True)
        if self.run_decoder:
            fields.append(
                DatasetFieldConfig(
                    name=DataFieldName.VIDEO_ORIGINAL,
                    decoder_config=NativeDataloaderMediaDecoderConfig(
                        resize_method=None,
                        random=False,
                    ),
                )
            )
        else:
            fields.append(DatasetFieldConfig(name=DataFieldName.VIDEO_ORIGINAL))

        mappers = []
        if self.run_mapper:
            mappers.append(
                DatasetFieldMapperConfig(
                    name=FieldsMapperName.QWEN_VL_MAPPER,
                    params={
                        "model_args": model_args,
                        "data_args": data_args,
                        "tokenizer": tokenizer,
                    },
                )
            )

        dataset_config = DatasetConfig(
            dataset_metadata=(
                data_args.dataset_metadata
                if mode == "train"
                else data_args.eval_dataset_metadata
            ),
            region=data_args.dataset_region,
            fields=fields,
            mappers=mappers,
            debug_mode=data_args.debug_mode,
        )

        dataloader = DataloaderFactory.create_data_loader(
            dataset_config,
            loop=data_args.loop,
            batch_size=(
                data_args.train_batch_size
                if mode == "train"
                else data_args.eval_batch_size
            ),
            num_decode_workers=(
                1 if data_args.debug_mode else data_args.num_decode_workers
            ),
            num_download_workers=(
                1 if data_args.debug_mode else data_args.num_download_workers
            ),
            decode_queue_size=8,
            shuffle_seed=(None if data_args.debug_mode else data_args.shuffle_seed),
            mode=mode,
        )

        if run_prefetch_iterator:
            dataloader = PrefetchIterator(
                it=dataloader,
                buffer_size=data_args.buffer_size,
                thread_count=data_args.thread_count,
            )

        self.dataloader = dataloader
        self.data_args = data_args
        self.model_args = model_args
        self.mode = mode
        self.iterator = iter(self.dataloader)

    def _download_video(
        self, sample_id: str, source_file: Optional[str] = None
    ) -> Optional[str]:
        """Download video file from GCS and return local path."""
        if not self.download_video or not self.video_cache_dir:
            return None

        video_filename = f"{sample_id}.mp4"
        local_path = self.video_cache_dir / video_filename

        if local_path.exists():
            return str(local_path)

        if source_file:
            video_tar = source_file.replace(
                "audio_visual_caption", "videos_original"
            ).replace(".feather", ".tar")

        try:
            result = subprocess.run(
                f"gsutil cat {video_tar} | tar -xf - {video_filename}",
                shell=True,
                cwd=str(self.video_cache_dir),
                capture_output=True,
            )

            if result.returncode == 0 and local_path.exists():
                return str(local_path)
        except Exception as e:
            print(f"Failed to download video {sample_id}: {e}")

        return None

    def __getitem__(self, index):
        start = time.time()

        for _ in range(3):
            try:
                batch = next(self.iterator)
                break
            except Exception as e:
                print(f"Failed to get batch {index}: {e}")
                time.sleep(1)

        if self.download_video:
            sample_ids = []
            source_files = []
            if DataFieldName.SAMPLE_ID in batch:
                sample_ids = batch[DataFieldName.SAMPLE_ID]
            if DataFieldName.SOURCE_FILE in batch:
                source_files = batch[DataFieldName.SOURCE_FILE]

            video_files = []
            for source_file, sample_id in zip(source_files, sample_ids):
                video_path = self._download_video(str(sample_id), source_file)
                video_files.append(video_path)
            batch["video_file"] = video_files

        if self.convert_to_collator_instances:
            batch = mapper_batch_to_collator_instances(batch)

        end = time.time()
        print(f"Time taken for preparing batch: {end - start} seconds")
        return batch

    def __len__(self):
        return (
            self.data_args.num_train_samples
            if self.mode == "train"
            else self.data_args.num_eval_samples
        )


def mapper_batch_to_collator_instances(
    mapper_batch: Dict[str, Any],
) -> List[Dict[str, Any]]:
    n = len(mapper_batch["input_ids"])

    input_ids_list = mapper_batch["input_ids"]
    targets_list = mapper_batch["targets"]
    pos_ids_list = mapper_batch["position_ids"]
    video_tensors_list = mapper_batch["video_tensor"]
    video_thw_list = mapper_batch["video_grid_thw"]
    audio_feats_list = mapper_batch["audio_inputs"]
    audio_lens_list = mapper_batch["audio_lengths"]
    train_type = mapper_batch["train_type"]

    instances: List[Dict[str, Any]] = []
    for i in range(n):
        inp: torch.Tensor = input_ids_list[i]

        inst: Dict[str, Any] = {
            "input_ids": inp,
            "labels": targets_list[i],
            "position_ids": pos_ids_list[i],
            "chosen_ids": None,
            "chosen_labels": None,
            "chosen_position_ids": None,
            "reject_ids": None,
            "reject_labels": None,
            "reject_position_ids": None,
            "audio_feature": audio_feats_list[i],
            "train_type": train_type,
            "attention_mask": [inp.numel()],
            "pixel_values_videos": video_tensors_list[i],
            "video_grid_thw": (
                video_thw_list[i].unsqueeze(0)
                if video_thw_list[i].dim() == 1
                else video_thw_list[i]
            ),
            "audio_lengths": [audio_lens_list[i]],
        }

        if "video_file" in mapper_batch:
            inst["video_file"] = mapper_batch["video_file"][i]

        instances.append(inst)

    return instances
