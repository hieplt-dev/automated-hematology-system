import argparse
import os

import torch

from src.ahs.models.faster_rcnn import BCCD_Model


def main():
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    p = argparse.ArgumentParser()
    p.add_argument(
        "--ckpt",
        type=str,
        default="experiments/outputs/best.pt",
        help="path to original checkpoint (best.pt)",
    )
    p.add_argument(
        "--out",
        type=str,
        default="experiments/outputs/best_qint8.pt",
        help="output quantized checkpoint",
    )
    args = p.parse_args()

    # check ckpt exists
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    # load model
    model = BCCD_Model(num_classes=3).model
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    # dynamic quantization: applies to Linear/LSTM/Embedding etc.
    # For detection models most ops are Conv2d so this may only shrink some parts.
    quantized = torch.ao.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )

    # save quantized model
    torch.save(quantized, args.out)
    print(f"Quantized checkpoint saved to: {args.out}")


if __name__ == "__main__":
    main()
