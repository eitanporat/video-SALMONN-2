#!/usr/bin/env python3
import os
import json
import glob
import pandas as pd
from pathlib import Path
from typing import Any

FEATHER_DIR = Path("/home/user/structured_captions")
VIDEO_DIR = Path("/home/user/videos/videos")
OUT_PATH   = Path("/home/user/structured_captions_dataset.json")

# Build a map: <id> -> absolute path to video
video_map = {}
for mp4 in VIDEO_DIR.glob("*.mp4"):
    vid_id = mp4.stem  # filename without extension
    video_map[vid_id] = str(mp4.resolve())

print(f"Indexed {len(video_map)} videos from {VIDEO_DIR}")

def normalize_response(val: Any) -> str:
    """
    Ensure the 'gpt' response is a JSON string (text), not a Python dict.
    - If val is a dict, dump to pretty/compact JSON.
    - If val is a string that looks like JSON, keep it as-is.
    - If it's a string with single quotes (Python repr), try to parse safely.
    """
    if isinstance(val, dict):
        return json.dumps(val, ensure_ascii=False)
    if isinstance(val, str):
        s = val.strip()
        # try parsing as JSON first
        try:
            parsed = json.loads(s)
            # re-dump to normalized compact JSON string
            return json.dumps(parsed, ensure_ascii=False)
        except Exception:
            pass
        # try Python-literal style fallback (single quotes, etc.)
        try:
            import ast
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (dict, list, str, int, float, bool)) or parsed is None:
                return json.dumps(parsed, ensure_ascii=False)
        except Exception:
            pass
        # last resort: return the raw string
        return s
    # anything else, make it JSON text
    try:
        return json.dumps(val, ensure_ascii=False)
    except Exception:
        return str(val)

results = []
seen = set()

feathers = sorted(glob.glob(str(FEATHER_DIR / "*.feather")))
print(f"Found {len(feathers)} feather files in {FEATHER_DIR}")

for fpath in feathers:
    try:
        df = pd.read_feather(fpath)
    except Exception as e:
        print(f"[WARN] Failed to read {fpath}: {e}")
        continue

    # Basic column sanity
    if not {"keys", "structured_caption"}.issubset(df.columns):
        print(f"[WARN] Skipping {fpath}: missing required columns")
        continue

    for _, row in df.iterrows():
        vid_id = str(row["keys"])
        if vid_id in seen:
            continue
        video_path = video_map.get(vid_id)
        if not video_path:
            # no matching video; skip
            continue

        response_text = normalize_response(row["structured_caption"])

        # Build one entry
        item = {
            "video": video_path,
            "use_audio": True,
            "conversations": [
                {
                    "from": "human",
                    "value": "<video>\nPlease describe this video."
                },
                {
                    "from": "gpt",
                    "value": response_text
                }
            ]
        }
        results.append(item)
        seen.add(vid_id)

print(f"Matched {len(results)} videos with captions. Writing to {OUT_PATH} ...")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("Done.")