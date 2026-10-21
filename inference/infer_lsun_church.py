from fpedit.options import inference_parser
from inference.common import run_inference


if __name__ == "__main__":
    run_inference(inference_parser("church", 256).parse_args())
