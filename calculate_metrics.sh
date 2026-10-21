#!/usr/bin/env bash
set -euo pipefail

DATASET=${1:?dataset: ffhq, cat, or church}
PROMPT_KEY=${2:?prompt key from configs/prompts.yaml}
SOURCE_DIR=${3:?256x256 real source directory}
EDITED_DIR=${4:?256x256 edited image directory}
MASK_DIR=${5:?normalized mask directory}
OUTPUT_DIR=${6:?metric JSON output directory}
IDS_WEIGHTS=${7:?IR-SE50 for FFHQ or MoCo-v2 for Cat/Church}
DEVICE=${8:-cuda:0}

mkdir -p "$OUTPUT_DIR"

python -m metrics.ids --source-dir "$SOURCE_DIR" --edited-dir "$EDITED_DIR" --domain "$DATASET" --weights "$IDS_WEIGHTS" --device "$DEVICE" --output "$OUTPUT_DIR/ids.json"
python -m metrics.psnr --source-dir "$SOURCE_DIR" --edited-dir "$EDITED_DIR" --output "$OUTPUT_DIR/psnr.json"
python -m metrics.lpips --source-dir "$SOURCE_DIR" --edited-dir "$EDITED_DIR" --device "$DEVICE" --output "$OUTPUT_DIR/lpips.json"
python -m metrics.topiq --source-dir "$SOURCE_DIR" --edited-dir "$EDITED_DIR" --device "$DEVICE" --output "$OUTPUT_DIR/topiq.json"
python -m metrics.hpsv3 --source-dir "$SOURCE_DIR" --edited-dir "$EDITED_DIR" --dataset "$DATASET" --prompt-key "$PROMPT_KEY" --device "$DEVICE" --output "$OUTPUT_DIR/hpsv3.json"
python -m metrics.clip_directional --source-dir "$SOURCE_DIR" --edited-dir "$EDITED_DIR" --dataset "$DATASET" --prompt-key "$PROMPT_KEY" --device "$DEVICE" --output "$OUTPUT_DIR/clip_directional.json"
python -m metrics.clip_image_text --source-dir "$SOURCE_DIR" --edited-dir "$EDITED_DIR" --dataset "$DATASET" --prompt-key "$PROMPT_KEY" --device "$DEVICE" --output "$OUTPUT_DIR/clip_image_text.json"
python -m metrics.ntpd --source-dir "$SOURCE_DIR" --edited-dir "$EDITED_DIR" --mask-dir "$MASK_DIR" --device "$DEVICE" --output "$OUTPUT_DIR/ntpd.json"
python -m metrics.change_precision --source-dir "$SOURCE_DIR" --edited-dir "$EDITED_DIR" --mask-dir "$MASK_DIR" --device "$DEVICE" --output "$OUTPUT_DIR/change_precision.json"
