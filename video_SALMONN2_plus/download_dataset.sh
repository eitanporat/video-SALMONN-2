#!/bin/sh
mkdir -p videos
mkdir -p eval_videos

mkdir -p visual_caption 
mkdir -p eval_visual_caption 

mkdir -p audio_caption
mkdir -p eval_audio_caption

mkdir -p visual_audio_concatenated_caption
mkdir -p eval_visual_audio_concatenated_caption

mkdir -p structured_captions
mkdir -p eval_structured_captions

gcloud storage cp "gs://lt-research-mm-datasets-europe-west4/mevaseret-v2/videos_original/videos_original_001_0*" videos/
gcloud storage cp "gs://lt-research-mm-datasets-europe-west4/mevaseret-v2/videos_original/videos_original_002_0*" eval_videos/

gcloud storage cp "gs://lt-research-mm-datasets-europe-west4/mevaseret-v2/visual_caption/visual_caption_001_0*" visual_caption/
gcloud storage cp "gs://lt-research-mm-datasets-europe-west4/mevaseret-v2/visual_caption/visual_caption_002_0*" eval_visual_caption/

gcloud storage cp "gs://lt-research-mm-datasets-europe-west4/mevaseret-v2/audio_caption/audio_caption_001_0*" audio_caption
gcloud storage cp "gs://lt-research-mm-datasets-europe-west4/mevaseret-v2/audio_caption/audio_caption_002_0*" eval_audio_caption

gcloud storage cp "gs://lt-research-mm-datasets-europe-west4/mevaseret-v2/visual_audio_concatenated_caption/visual_audio_concatenated_caption_001_0*" visual_audio_concatenated_caption 
gcloud storage cp "gs://lt-research-mm-datasets-europe-west4/mevaseret-v2/visual_audio_concatenated_caption/visual_audio_concatenated_caption_002_0*" eval_visual_audio_concatenated_caption 

gcloud storage cp "gs://lt-research-mm-datasets-europe-west4/mevaseret-v2/structured_captions/structured_captions_001_0*" structured_captions 
gcloud storage cp "gs://lt-research-mm-datasets-europe-west4/mevaseret-v2/structured_captions/structured_captions_002_0*" eval_structured_captions 
