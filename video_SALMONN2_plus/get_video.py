#!/usr/bin/env python3
import pandas as pd
import subprocess
import sys

# The sample ID we want
sample_id = (
    "7304fc381e9f8bf82ec398e981efe68c_521059712" if len(sys.argv) == 1 else sys.argv[1]
)
feather_path = "gs://lt-research-mm-datasets-europe-west4/mevaseret-v2/audio_visual_caption/audio_visual_caption_000_2046.feather"

print(f"Looking for sample: {sample_id}")

# Download and read feather file
subprocess.run(["gsutil", "cp", feather_path, "temp.feather"], check=True)
df = pd.read_feather("temp.feather")

# Find the sample
matching_row = df[df["keys"] == sample_id]
if matching_row.empty:
    print(f"Sample {sample_id} not found!")
    exit(1)

print(f"Found sample! Caption: {matching_row['caption'].iloc[0][:100]}...")

# Now we need to figure out where the actual video file is stored
# Based on the logs, videos are in tar files like videos_original_000_0000.tar
# The sample key should map to a file in there
print(f"Sample key: {matching_row['keys'].iloc[0]}")
print("Video should be in the videos_original tar files...")

# Try to download with parallel/faster settings
video_tar = "gs://lt-research-mm-datasets-europe-west4/mevaseret-v2/videos_original/videos_original_000_2046.tar"
print(f"Downloading tar file (parallel)...")
subprocess.run(
    [
        "gsutil",
        "-m",
        "-o",
        "GSUtil:parallel_composite_upload_threshold=150M",
        "cp",
        video_tar,
        "videos.tar",
    ],
    check=True,
)

# Extract the video file (it should be named with the sample key)
import tarfile

with tarfile.open("videos.tar", "r") as tar:
    for member in tar.getmembers():
        if sample_id in member.name:
            print(f"Found video file: {member.name}")
            tar.extract(member, ".")
            print(f"Extracted to: {member.name}")
            break
    else:
        print(f"Video file not found in tar for sample {sample_id}")
