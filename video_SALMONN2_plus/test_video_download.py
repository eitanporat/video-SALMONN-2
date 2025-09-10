#!/usr/bin/env python3
import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from qwenvl.data.qwen_dataset import QwenDataset
from qwenvl.train.argument import ModelArguments, DataArguments
from txt2img.common import dist_util
from txt2img.common.machine import AccelerationType

# Initialize distributed utilities
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")

try:
    dist_util.init(AccelerationType.GPU, devices_per_vm=1)
except:
    dist_util.init(AccelerationType.CPU, devices_per_vm=1)

# Setup arguments
model_args = ModelArguments()
data_args = DataArguments()
data_args.debug_mode = True

print("Testing QwenDataset with download_video=True...")

# Create dataset with video downloading enabled
dataset = QwenDataset(
    model_args=model_args,
    data_args=data_args,
    mode="train",
    download_video=True,  # Enable video downloading!
    convert_to_collator_instances=True,  # Enable collator conversion
    run_mapper=True,  # Enable mapping for collator
)

print(f"Dataset created with {len(dataset)} samples")

# Test one sample
print("Getting first sample...")
batch = dataset[0]

# Handle the case where convert_to_collator_instances=True returns a list
if isinstance(batch, list):
    print(f"Got {len(batch)} collator instances")
    sample = batch[0]  # Take first instance
    print(f"Sample keys: {list(sample.keys())}")

    if "video_file" in sample:
        video_path = sample["video_file"]
        print(f"Video file: {video_path}")

        if video_path and os.path.exists(video_path):
            size = os.path.getsize(video_path)
            print(f"✅ Video: {video_path} ({size} bytes)")
        else:
            print(f"❌ Video: {video_path} (not found)")
    else:
        print("❌ No 'video_file' field found in sample")
else:
    # Handle the batch format case
    print(f"Sample keys: {list(batch.keys())}")

    if "video_file" in batch:
        video_files = batch["video_file"]
        print(f"Video files: {video_files}")

        for i, video_path in enumerate(video_files):
            if video_path and os.path.exists(video_path):
                size = os.path.getsize(video_path)
                print(f"✅ Video {i}: {video_path} ({size} bytes)")
            else:
                print(f"❌ Video {i}: {video_path} (not found)")
    else:
        print("❌ No 'video_file' field found in batch")
