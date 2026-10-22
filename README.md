# FPEdit: Exploiting joint latent representation modulation for fast and precise text-guided image manipulation

This repository contains the official PyTorch implementation of the paper **"FPEdit: Exploiting joint latent representation modulation for fast and precise text-guided image manipulation"**.

## Environment

The reference environment is Python 3.10, PyTorch 2.0.0, torchvision 0.15.0,
and CUDA 11.8. Install the remaining packages with:

```bash
pip install -r requirements.txt
```

## Released files

- The architectural definition of FPEdit mapper.
- Training entry point for FFHQ, LSUN Cat, and LSUN Church.
- The e4e real-image inversion and domain-specific inference entry points.
- Fixed model and evaluation prompts in `configs/prompts.yaml`.
- Nine standalone metrics: IDS, PSNR, LPIPS, TOPIQ, HPSv3, CLIP Directional
  Similarity, CLIP Image-Text Similarity, NTPD, and Change Precision.

## Public data and weights

The fixed latent splits and real-image evaluation sets can be [downloaded here](https://drive.google.com/drive/folders/1B8WLMjQtGJrFN2kESrDiHkWOdqWZlUJO?usp=sharing).

The folder contains the following data for FFHQ, LSUN Cat, and LSUN Church:

- `train_ffhq.pt`, `train_cat.pt`, and `train_church.pt`: 24,300 training
  latent codes per domain.
- `val_ffhq.pt`, `val_cat.pt`, and `val_church.pt`: 2,700 validation latent
  codes per domain.
- A fixed set of 200 real evaluation images per domain.
- JSON files recording the training and validation latent splits.

The FFHQ example FPEdit checkpoint can be [downloaded here](https://drive.google.com/file/d/1BZCX60V0rvGFyH45hVBegTNT8ApOIaLS/view?usp=sharing).

The FFHQ StyleGAN2 checkpoint can be [downloaded here](https://drive.google.com/file/d/1zm09PbOfvGyinw9BGPva5d9zm94df3pZ/view?usp=sharing).

The e4e encoder implementation is available from the [official e4e repository](https://github.com/omertov/encoder4editing).

## Data protocol

For each domain, 27,000 latent codes are sampled from the corresponding frozen
StyleGAN latent distribution. The fixed split contains 24,300 training codes
and 2,700 validation codes. All prompt-specific mappers in a domain use the
same split. StyleGAN remains frozen and only the FPEdit mapper is optimized.

Final evaluation uses 200 fixed real images per domain. These images are not
used for FPEdit mapper training, validation, or checkpoint selection.

| Domain | Train latents | Validation latents | Real test images | Prompts | Image-prompt pairs |
|---|---:|---:|---:|---:|---:|
| FFHQ | 24,300 | 2,700 | 200 | 14 | 2,800 |
| LSUN Cat | 24,300 | 2,700 | 200 | 5 | 1,000 |
| LSUN Church | 24,300 | 2,700 | 200 | 5 | 1,000 |

All real inputs are resized to 256x256 before e4e
inversion. FFHQ, LSUN Cat, and LSUN Church use their corresponding e4e encoders.

## Prompts

FPEdit is trained with the short model conditions listed in
`configs/prompts.yaml`. CLIP and HPSv3 use the complete target
sentences stored separately in the same file. The neutral texts are:

- FFHQ model prompts: `angry`, `bald`, `beard`, `black hair`, `blonde hair`,
  `bob cut`, `curly hairstyle`, `happy`, `makeup`, `mohawk hairstyle`,
  `purple hairstyle`, `sad`, `straight hair`, and `surprised`.
- LSUN Cat model prompts: `cute cat`, `fat cat`, `lying cat`, `orange cat`, and
  `standing cat`.
- LSUN Church model prompts: `gothic church`, `more trees`, `round roof`,
  `spires`, and `white church`.
  
## Training

Example FFHQ training command:

```bash
bash train.sh ffhq "TARGET_PROMPT" /path/to/training_output_directory \
  /path/to/train_ffhq.pt /path/to/val_ffhq.pt \
  /path/to/stylegan2-ffhq-config-f.pt /path/to/model_ir_se50.pth 1024
```

Replace `TARGET_PROMPT` with the desired model prompt defined in
`configs/prompts.yaml` and set the output directory as needed.

For LSUN Cat and LSUN Church, use `cat` or `church` as the first argument for the corresponding domain, pass the corresponding 256x256 StyleGAN weights, latent files, and MoCo-v2 feature checkpoint to the same script.

## Real-image inference

Example FFHQ inferencing command:

```bash
bash inference.sh ffhq "TARGET_PROMPT" /path/to/source_images \
  /path/to/inference_output_directory /path/to/ckpt_FFHQ.pt \
  /path/to/stylegan2-ffhq-config-f.pt /path/to/e4e_ffhq_encode.pt
```

For LSUN Cat and LSUN Church, use `cat` or `church` as the first argument for the corresponding domain, pass the corresponding 256x256 StyleGAN weights, and e4e checkpoint to the same script.

## Metrics

Images are evaluated at 256x256. Each script
writes one JSON file containing `mean`, `std`, and `n`.

Metric inputs are fixed as follows:

- IDS, PSNR, LPIPS, and TOPIQ compare the real source image with the edited
  image.
- HPSv3 and CLIP Image-Text Similarity use the edited image and the registered
  target sentence.
- CLIP Directional Similarity uses the source image, edited image, target text,
  and domain-neutral text.
- NTPD and Change Precision use source/edited pairs and a normalized
  target mask.

Run all nine metrics for one dataset and prompt with:

```bash
bash calculate_metrics.sh ffhq "TARGET_PROMPT" \
  /path/to/source_images /path/to/edited_images /path/to/normalized_masks \
  /path/to/metrics_output_directory /path/to/model_ir_se50.pth cuda:0
```

For Cat or Church, pass the MoCo-v2 checkpoint as the IDS weight argument. The
script writes nine independent JSON files in the requested output directory.

## Intended use
FPEdit is intended to support non-commercial academic research and method
reproducibility. Users are responsible for complying with applicable laws, dataset terms, and
the licenses and usage conditions of all third-party models and software.
