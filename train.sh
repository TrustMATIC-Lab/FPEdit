#!/usr/bin/env bash
set -euo pipefail

DOMAIN=${1:?domain: ffhq, cat, or church}
PROMPT=${2:?model condition prompt}
OUTPUT_DIR=${3:?output directory}
TRAIN_LATENTS=${4:?training latent file}
VAL_LATENTS=${5:?validation latent file}
STYLEGAN_CKPT=${6:?StyleGAN checkpoint}
IDENTITY_CKPT=${7:?IR-SE50 for FFHQ or MoCo-v2 for Cat/Church}
STYLEGAN_SIZE=${8:?1024 for FFHQ; 256 for Cat/Church}

python train.py \
  --domain "$DOMAIN" \
  --description "$PROMPT" \
  --output-dir "$OUTPUT_DIR" \
  --train-latents "$TRAIN_LATENTS" \
  --val-latents "$VAL_LATENTS" \
  --stylegan-ckpt "$STYLEGAN_CKPT" \
  --identity-ckpt "$IDENTITY_CKPT" \
  --stylegan-size "$STYLEGAN_SIZE" \
  --max-steps 2000 \
  --save-interval 100 \
  --val-interval 100
