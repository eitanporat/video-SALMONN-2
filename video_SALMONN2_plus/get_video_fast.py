#!/usr/bin/env python3
import pandas as pd
import subprocess
import sys
import os

# Get sample ID and paths
sample_id = (
    "7304fc381e9f8bf82ec398e981efe68c_521059712" if len(sys.argv) == 1 else sys.argv[1]
)
feather_path = "gs://lt-research-mm-datasets-europe-west4/mevaseret-v2/audio_visual_caption/audio_visual_caption_000_2046.feather"
video_tar = "gs://lt-research-mm-datasets-europe-west4/mevaseret-v2/videos_original/videos_original_000_2046.tar"

print(f"Looking for sample: {sample_id}")

# Download feather, find sample, clean up
subprocess.run(["gsutil", "cp", feather_path, "temp.feather"], check=True)
df = pd.read_feather("temp.feather")
if df[df["keys"] == sample_id].empty:
    print(f"Sample {sample_id} not found!")
    exit(1)

print("Found sample! Streaming video...")

# Stream tar and extract video
result = subprocess.run(
    f"gsutil cat {video_tar} | tar -xf - {sample_id}.mp4", shell=True
)

if result.returncode == 0:
    size = os.path.getsize(f"{sample_id}.mp4")
    print(f"✅ Extracted {sample_id}.mp4 ({size} bytes)")

    # Read the video file
    with open(f"{sample_id}.mp4", "rb") as f:
        video_data = f.read()
        print(f"📖 Read video file: {len(video_data)} bytes in memory")

else:
    print("❌ Failed to extract video")

# Clean up
subprocess.run(["rm", "temp.feather"], check=False)
