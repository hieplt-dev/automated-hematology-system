from argparse import ArgumentParser

from src.ahs.training.train import train
from src.ahs.utils.load_config import load_config

if __name__ == "__main__":
    """Parse CLI arguments for training"""
    parser = ArgumentParser(description="args for training")
    parser.add_argument("--epochs", type=int, default=None, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="batch size")
    parser.add_argument("--img_size", type=int, default=None, help="image size")
    parser.add_argument("--lr", type=float, default=None, help="learning rate")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="experiments/checkpoint.pt",
        help="path to save checkpoint",
    )

    parser.add_argument(
        "--ds_cfg_path",
        type=str,
        default="config/data.yaml",
        help="path to dataset config",
    )
    parser.add_argument(
        "--aug_cfg_path",
        type=str,
        default="config/transform.yaml",
        help="path to aug config",
    )
    parser.add_argument(
        "--tr_cfg_path",
        type=str,
        default="config/train.yaml",
        help="path to training config",
    )

    args = parser.parse_args()

    # load dataset, augmentation and training config
    ds_cfg = load_config(args.ds_cfg_path)
    aug_cfg = load_config(args.aug_cfg_path)
    tr_cfg = load_config(args.tr_cfg_path)

    # train merge
    tr_cfg["epochs"] = args.epochs or tr_cfg["epochs"]
    tr_cfg["batch_size"] = args.epochs or tr_cfg["batch_size"]
    tr_cfg["lr"] = args.epochs or tr_cfg["lr"]

    # dataset merge
    tr_cfg["img_size"] = args.epochs or ds_cfg["image_size"]

    # call main train func
    train(tr_cfg, ds_cfg, aug_cfg)
