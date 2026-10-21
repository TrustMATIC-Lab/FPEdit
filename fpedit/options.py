
import argparse
import math
from pathlib import Path


def add_mapper_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--num-self-attn-layers", type=int, default=4)
    parser.add_argument("--num-cross-attn-layers", type=int, default=4)
    parser.add_argument("--num-attention-blocks", type=int, default=1)
    parser.add_argument("--num-self-attn-heads", type=int, default=4)
    parser.add_argument("--num-cross-attn-heads", type=int, default=4)
    parser.add_argument("--qk-out-dim", type=int, default=None)
    parser.add_argument("--v-out-dim", type=int, default=None)
    parser.add_argument("--mapper-dropout", type=float, default=0.0)
    parser.add_argument("--self-attention-dropout", type=float, default=0.0)
    parser.add_argument("--cross-attention-dropout", type=float, default=0.0)
    parser.add_argument("--self-residual-coefficient", type=float, default=1.0)
    parser.add_argument("--cross-residual-coefficient", type=float, default=1.0)
    parser.add_argument("--no-coarse-mapper", action="store_true")
    parser.add_argument("--no-medium-mapper", action="store_true")
    parser.add_argument("--no-fine-mapper", action="store_true")


def finish_options(options):
    options.stylegan_size = int(options.stylegan_size)
    options.num_ws = int(math.log2(options.stylegan_size)) * 2 - 2
    if options.stylegan_size not in (256, 1024):
        raise ValueError("This release supports StyleGAN sizes 256 and 1024")
    if options.self_residual_coefficient != 1.0 or options.cross_residual_coefficient != 1.0:
        raise ValueError("The final model uses fixed unit residual coefficients")
    if options.no_coarse_mapper or options.no_medium_mapper or options.no_fine_mapper:
        print("Warning: the released paper configuration enables all three levels")
    return options


def train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the final FPEdit mapper")
    parser.add_argument("--domain", choices=("ffhq", "cat", "church"), required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-latents", type=Path, required=True)
    parser.add_argument("--val-latents", type=Path, required=True)
    parser.add_argument("--stylegan-ckpt", "--stylegan_ckpt", type=Path, required=True)
    parser.add_argument(
        "--identity-ckpt", "--moco-ckpt", "--moco_ckpt",
        dest="identity_ckpt", type=Path, required=True,
    )
    parser.add_argument("--stylegan-size", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--clip-model", default="ViT-B/32")
    parser.add_argument("--learning-rate", type=float, default=0.5)
    parser.add_argument("--optimizer", choices=("ranger", "adam"), default="ranger")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--val-batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--val-interval", type=int, default=100)
    parser.add_argument("--image-interval", type=int, default=100)
    parser.add_argument("--validation-batches", type=int, default=200)
    parser.add_argument("--id-lambda", type=float, default=0.1)
    parser.add_argument("--clip-lambda", type=float, default=1.0)
    parser.add_argument("--latent-l2-lambda", type=float, default=0.8)
    parser.add_argument("--edit-strength", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    add_mapper_options(parser)
    return parser


def inference_parser(domain: str, stylegan_size: int) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"FPEdit inference for {domain}")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reconstruction-dir", type=Path)
    parser.add_argument("--mapper-ckpt", type=Path, required=True)
    parser.add_argument("--stylegan-ckpt", "--stylegan_ckpt", type=Path, required=True)
    parser.add_argument("--e4e-ckpt", "--e4e_ckpt", type=Path, required=True)
    parser.add_argument(
        "--moco-ckpt", "--moco_ckpt", type=Path,
        default=Path("/path/to/moco_v2_800ep_pretrain.pt"),
        help="Reserved Cat/Church feature-model path; not used during inference",
    )
    parser.add_argument("--prompt-key", required=True)
    parser.add_argument("--prompts-yaml", type=Path, default=Path("configs/prompts.yaml"))
    parser.add_argument("--stylegan-size", type=int, default=stylegan_size)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--clip-model", default="ViT-B/32")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--edit-strength", type=float, default=0.1)
    parser.set_defaults(domain=domain)
    add_mapper_options(parser)
    return parser
