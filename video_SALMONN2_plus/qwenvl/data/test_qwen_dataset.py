import pytest
from qwen_dataset import QwenDataset, map_batch_to_sample
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
)
from qwenvl.data.dataset import DataCollatorForSupervisedDataset, LazySupervisedDataset
from transformers import AutoTokenizer
from txt2img.common import dist_util
from txt2img.common.machine import AccelerationType
from qwenvl.data.image_processing_qwen2_vl_fast import Qwen2VLImageProcessorFast
from transformers import WhisperFeatureExtractor
from dataset_metadata.data_field_name import DataFieldName
from torchcodec.decoders import VideoDecoder
import io
import torch


def setup_args():
    model_args = ModelArguments()
    data_args = DataArguments()

    data_args.image_processor = Qwen2VLImageProcessorFast.from_pretrained(
        model_args.model_base,
    )
    data_args.audio_processor = WhisperFeatureExtractor(
        feature_size=data_args.feature_size,
        sampling_rate=data_args.sampling_rate,
        hop_length=data_args.hop_length,
        chunk_length=data_args.chunk_length,
    )
    data_args.model_type = "qwen2.5vl"
    data_args.use_iterator = True
    data_args.debug_mode = False

    return model_args, data_args


def test_deterministic():
    model_args, data_args = setup_args()

    test_qwen_dataset = QwenDataset(
        model_args=model_args,
        data_args=data_args,
        debug_mode=True,
        run_decoder=False,
        run_mapper=False,
    )

    test_qwen_dataset_2 = QwenDataset(
        model_args=model_args,
        data_args=data_args,
        debug_mode=True,
        run_decoder=False,
        run_mapper=False,
    )

    batch_qwen = next(iter(test_qwen_dataset))

    batch_qwen_2 = next(iter(test_qwen_dataset_2))

    assert (
        batch_qwen[DataFieldName.SOURCE_FILE] == batch_qwen_2[DataFieldName.SOURCE_FILE]
    ), "Source file is not deterministic"
    assert (
        batch_qwen[DataFieldName.AUDIO_VISUAL_CAPTION]
        == batch_qwen_2[DataFieldName.AUDIO_VISUAL_CAPTION]
    ), "Audio visual caption is not deterministic"


def setup_datasets(model_args, data_args, convert_to_collator_instances=False):
    pl_kwargs = None
    mode = "train"

    qwen_dataset = QwenDataset(
        model_args=model_args,
        data_args=data_args,
        pl_kwargs=pl_kwargs,
        mode=mode,
        convert_to_collator_instances=convert_to_collator_instances,
        dtype=torch.float16,
    )

    # I only use this dataset to read the data from dataloader
    test_qwen_dataset = QwenDataset(
        model_args=model_args,
        data_args=data_args,
        pl_kwargs=pl_kwargs,
        mode=mode,
        debug_mode=True,
        run_decoder=False,
        run_mapper=False,
        convert_to_collator_instances=False,
        dtype=torch.float16,
    )

    return qwen_dataset, test_qwen_dataset


def test_decoder():
    model_args, data_args = setup_args()

    qwen_dataset, test_qwen_dataset = setup_datasets(
        model_args, data_args, convert_to_collator_instances=False
    )

    batch_qwen = next(iter(qwen_dataset))
    batch_test_qwen = next(iter(test_qwen_dataset))

    assert (
        batch_test_qwen[DataFieldName.SOURCE_FILE]
        == batch_qwen[DataFieldName.SOURCE_FILE]
    )

    video = batch_test_qwen[DataFieldName.VIDEO_ORIGINAL]
    video_test_torchcodec = VideoDecoder(io.BytesIO(video[0]))

    frames_qwen = batch_qwen[DataFieldName.VIDEO_ORIGINAL]["frames"][0]
    indices = list(range(len(frames_qwen)))

    assert abs(video_test_torchcodec.metadata.num_frames - len(frames_qwen)) <= 5

    frames_test_torchcodec = video_test_torchcodec.get_frames_at(
        indices=indices,
    ).data

    frames_qwen = batch_qwen[DataFieldName.VIDEO_ORIGINAL]["frames"][0]
    assert frames_qwen.shape == frames_test_torchcodec.shape

    assert (frames_qwen == frames_test_torchcodec).all()


def load_tokenizer():
    lora_ckpt = "tsinghua-ee/video-SALMONN-2_plus_7B"
    tokenizer = AutoTokenizer.from_pretrained(lora_ckpt)

    return tokenizer


def test_subsample_video():
    model_args, data_args = setup_args()
    qwen_dataset, test_qwen_dataset = setup_datasets(model_args, data_args)

    tokenizer = load_tokenizer()

    batch_qwen = next(iter(qwen_dataset))

    batch_test_qwen = next(iter(test_qwen_dataset))

    lazy_supervised_dataset = LazySupervisedDataset(
        tokenizer=tokenizer, data_args=data_args
    )

    sample = map_batch_to_sample(batch_test_qwen)
    video = sample["video"]

    device = "cpu"  # or e.g. "cuda"
    decoder = VideoDecoder(video[0], device=device)

    subsampled_video_lazy, _, _ = lazy_supervised_dataset.subsample_video(decoder)

    subsampled_video_qwen = batch_qwen[DataFieldName.VIDEO_ORIGINAL][
        "subsampled_video"
    ][0]

    assert (subsampled_video_lazy == subsampled_video_qwen).all()


def test_conversations():
    model_args, data_args = setup_args()
    qwen_dataset, test_qwen_dataset = setup_datasets(model_args, data_args)

    tokenizer = load_tokenizer()

    lazy_supervised_dataset = LazySupervisedDataset(
        tokenizer=tokenizer, data_args=data_args
    )

    batch_test_qwen = next(iter(test_qwen_dataset))

    sample = map_batch_to_sample(batch_test_qwen)
    batch_lazy = lazy_supervised_dataset.process_source(sample)

    batch_qwen = next(iter(qwen_dataset))

    assert len(batch_qwen["conversations"][0]) == len(batch_lazy["conversations"])

    assert (
        batch_qwen["conversations"][0][0]["content"]
        == batch_lazy["conversations"][0]["value"]
    )

    assert (
        batch_qwen["conversations"][0][1]["content"]
        == batch_lazy["conversations"][1]["value"]
    )


def test_input_ids():
    model_args, data_args = setup_args()
    qwen_dataset, test_qwen_dataset = setup_datasets(model_args, data_args)
    batch_test_qwen = next(iter(test_qwen_dataset))
    tokenizer = load_tokenizer()

    lazy_supervised_dataset = LazySupervisedDataset(
        tokenizer=tokenizer, data_args=data_args
    )

    sample = map_batch_to_sample(batch_test_qwen)
    batch_lazy = lazy_supervised_dataset.process_source(sample)
    batch_qwen = next(iter(qwen_dataset))

    assert (batch_lazy["video_grid_thw"][0] == batch_qwen["video_grid_thw"][0]).all()
    assert (batch_lazy["input_ids"][0] == batch_qwen["input_ids"][0]).all()


def test_position_ids():
    model_args, data_args = setup_args()
    qwen_dataset, test_qwen_dataset = setup_datasets(model_args, data_args)
    batch_test_qwen = next(iter(test_qwen_dataset))
    tokenizer = load_tokenizer()

    lazy_supervised_dataset = LazySupervisedDataset(
        tokenizer=tokenizer, data_args=data_args
    )

    sample = map_batch_to_sample(batch_test_qwen)
    batch_lazy = lazy_supervised_dataset.process_source(sample)
    batch_qwen = next(iter(qwen_dataset))

    assert (batch_lazy["position_ids"] == batch_qwen["position_ids"][0]).all()


def test_features():
    model_args, data_args = setup_args()
    qwen_dataset, test_qwen_dataset = setup_datasets(model_args, data_args)
    batch_test_qwen = next(iter(test_qwen_dataset))
    batch_qwen = next(iter(qwen_dataset))
    tokenizer = load_tokenizer()

    lazy_supervised_dataset = LazySupervisedDataset(
        tokenizer=tokenizer, data_args=data_args
    )

    sample = map_batch_to_sample(batch_test_qwen)
    batch_lazy = lazy_supervised_dataset.process_source(sample)

    audio_lazy = batch_lazy["audio_feature"]
    audio_qwen = batch_qwen["audio_inputs"]

    def angle_distance(a, b):
        import numpy as np

        return np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    # i don't know why its not perfect
    assert audio_lazy.shape == audio_qwen[0].shape
    assert angle_distance(audio_lazy[0][0], audio_qwen[0][0][0]) < 0.2

    assert (batch_lazy["pixel_values_videos"] == batch_qwen["video_tensor"][0]).all()


def test_collator_integration():
    tokenizer = load_tokenizer()
    collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    model_args, data_args = setup_args()
    data_args.batch_size = 8
    qwen_dataset, _ = setup_datasets(
        model_args, data_args, convert_to_collator_instances=True
    )

    qwen_dataset.dtype = torch.bfloat16

    iter_qwen = iter(qwen_dataset)

    import time

    n_times = 1
    for i in range(n_times):
        start_time = time.time()
        batch_qwen = next(iter_qwen)
        batch_collator = collator([batch_qwen])
        end_time = time.time()
        print(f"Time taken for {i}th batch: {end_time - start_time} seconds")

    assert (
        max(sample["position_ids"].shape[0] for sample in batch_qwen)
        == batch_collator["position_ids"].shape[0]
    )

    import IPython

    IPython.embed()

    assert batch_collator["video_grid_thw"].dim() == 2


if __name__ == "__main__":
    dist_util.init(acceleration_type=AccelerationType.GPU, devices_per_vm=1)
    # print("Testing deterministic...")
    # test_deterministic()
    # print("Testing decoder...")
    # test_decoder()
    # print("Testing qwen dataset...")
    # test_subsample_video()
    # print("Testing conversations...")
    # test_conversations()
    # print("Testing input_ids...")
    # test_input_ids()
    # print("Testing position_ids...")
    # test_position_ids()
    print("Testing collator integration...")
    test_collator_integration()
    # print("Testing features...")
    # test_features()
