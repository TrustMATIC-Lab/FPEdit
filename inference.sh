#!/usr/bin/env bash
set -euo pipefail

DOMAIN=${1:?domain: ffhq, cat, or church}
PROMPT_KEY=${2:?prompt key from configs/prompts.yaml}
INPUT_DIR=${3:?real image directory}
OUTPUT_DIR=${4:?edited image directory}
MAPPER_CKPT=${5:?mapper-only checkpoint}
STYLEGAN_CKPT=${6:?StyleGAN checkpoint}
E4E_CKPT=${7:?e4e checkpoint}
MOCO_CKPT=${8:-/path/to/moco_v2_800ep_pretrain.pt}

case "$DOMAIN" in
  ffhq) MODULE=inference.infer_ffhq ;;
  cat) MODULE=inference.infer_lsun_cat ;;
  church) MODULE=inference.infer_lsun_church ;;
  *) echo "Unknown domain: $DOMAIN" >&2; exit 2 ;;
esac

python -m "$MODULE" \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --mapper-ckpt "$MAPPER_CKPT" \
  --stylegan-ckpt "$STYLEGAN_CKPT" \
  --e4e-ckpt "$E4E_CKPT" \
  --moco-ckpt "$MOCO_CKPT" \
  --prompt-key "$PROMPT_KEY"
