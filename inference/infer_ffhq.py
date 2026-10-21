from fpedit.options import inference_parser
from inference.common import run_inference


if __name__ == "__main__":
    run_inference(inference_parser("ffhq", 1024).parse_args())
