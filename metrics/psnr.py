
import argparse
import math
from pathlib import Path

import numpy as np

from metrics.common import image_pairs, load_pil, write_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--edited-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    values = []
    for source, edited in image_pairs(args.source_dir, args.edited_dir):
        x = np.asarray(load_pil(source), dtype=np.float64)
        y = np.asarray(load_pil(edited), dtype=np.float64)
        mse = float(np.mean((x - y) ** 2))
        values.append(10.0 * math.log10((255.0 ** 2) / max(mse, 1e-12)))
    write_result(args.output, "PSNR", values)


if __name__ == "__main__":
    main()
