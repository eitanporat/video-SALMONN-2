#!/usr/bin/env python3
import pandas as pd
import subprocess
import sys
import tarfile
import io

# The sample ID we want
sample_id = (
    "7304fc381e9f8bf82ec398e981efe68c_521059712" if len(sys.argv) == 1 else sys.argv[1]
)
feather_path = "gs://lt-research-mm-datasets-europe-west4/mevaseret-v2/audio_visual_caption/audio_visual_caption_000_2046.feather"

print(f"Looking for sample: {sample_id}")

# Download and read feather file (small, fast)
subprocess.run(["gsutil", "cp", feather_path, "temp.feather"], check=True)
df = pd.read_feather("temp.feather")

# Find the sample
matching_row = df[df["keys"] == sample_id]
if matching_row.empty:
    print(f"Sample {sample_id} not found!")
    exit(1)

print(f"Found sample! Caption: {matching_row['caption'].iloc[0][:100]}...")

# Stream the tar file and extract only our video
video_tar = "gs://lt-research-mm-datasets-europe-west4/mevaseret-v2/videos_original/videos_original_000_2046.tar"
video_filename = f"{sample_id}.mp4"

print(f"Streaming tar to extract only: {video_filename}")

# Stream tar file directly from GCS and extract only the target file
cmd = ["gsutil", "cat", video_tar, "|", "tar", "-xf", "-", video_filename]

# Use shell=True to handle the pipe
result = subprocess.run(
    f"gsutil cat {video_tar} | tar -xf - {video_filename}",
    shell=True,
    capture_output=True,
    text=True,
)

if result.returncode == 0:
    print(f"✅ Extracted {video_filename} via streaming!")

    # Check file size
    import os

    size = os.path.getsize(video_filename)
    print(f"📁 File size: {size} bytes")

else:
    print(f"❌ Failed to extract: {result.stderr}")
    print(f"Trying alternative method...")

    # Fallback: Use Python streaming
    proc = subprocess.Popen(["gsutil", "cat", video_tar], stdout=subprocess.PIPE)

    with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
        for member in tar:
            if member.name == video_filename:
                print(f"🎯 Found {video_filename} in stream!")
                tar.extract(member)
                print(f"✅ Extracted via Python streaming!")
                break
        else:
            print(f"❌ {video_filename} not found in tar")

    proc.wait()

# Clean up
subprocess.run(["rm", "temp.feather"], check=False)
