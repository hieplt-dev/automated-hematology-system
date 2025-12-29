import os
import random
from argparse import ArgumentParser

import albumentations as A
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision import transforms
from tqdm import tqdm

from src.ahs.data.bccd_dataset import BCCD_DATASET
from src.ahs.models.faster_rcnn import BCCD_Model
from src.ahs.transforms.transforms_albu import build_train_aug_albu, build_val_aug_albu
from src.ahs.utils.load_config import load_config
from src.ahs.utils.visualize_img import visualize_img


# build optimizer (SGD/Adam)
def build_optimizer(name, params, lr, weight_decay=0.0, momentum=0.9):
    name = name.lower()
    if name == "sgd":
        return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


# collate_fn for DataLoader (to handle list of dicts)
def collate_fn(batch):
    imgs, targets = list(zip(*batch))
    return list(imgs), list(targets)


def train(tr_cfg=None, ds_cfg=None, aug_cfg=None):
    # check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # augmentations
    train_transform = build_train_aug_albu(aug_cfg, ds_cfg["image_size"])
    val_transform = build_val_aug_albu(ds_cfg["image_size"])

    # datasets (train/val/test)
    train_dataset = BCCD_DATASET(
        root=ds_cfg["root"],
        mode="train",
        transform=train_transform,
        label_map=ds_cfg["labels_map"],
    )
    val_dataset = BCCD_DATASET(
        root=ds_cfg["root"],
        mode="val",
        transform=val_transform,
        label_map=ds_cfg["labels_map"],
    )

    # dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=tr_cfg["batch_size"],
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=tr_cfg["batch_size"],
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
    )

    # model & Optimizer
    model = BCCD_Model(num_classes=ds_cfg["num_cls"]).model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = build_optimizer(
        tr_cfg["optimizer"],
        params,
        tr_cfg["lr"],
        tr_cfg["weight_decay"],
        tr_cfg["momentum"],
    )

    # MLflow
    TRACKING_URI = os.getenv("TRACKING_URI")
    EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME")

    mlflow.set_tracking_uri(TRACKING_URI)

    with mlflow.start_run(run_name=EXPERIMENT_NAME):
        mlflow.log_params(
            {
                "epochs": tr_cfg["epochs"],
                "batch_size": tr_cfg["batch_size"],
                "learning_rate": tr_cfg["lr"],
                "optimizer": tr_cfg["optimizer"],
                "weight_decay": tr_cfg["weight_decay"],
                "momentum": tr_cfg["momentum"],
                "image_size": ds_cfg["image_size"],
                "num_classes": ds_cfg["num_cls"],
            }
        )

        # log model summary
        try:
            with open(tr_cfg["model_summary_path"], "w") as f:
                f.write(str(summary(model, verbose=0)))
            mlflow.log_artifact("model_summary.txt")
        except Exception as e:
            print("Summary(model) failed:", e)

        # define best val loss
        best_val_loss = -1
        num_iterations = len(train_loader)

        # train loop -> dict loss
        for epoch in range(tr_cfg["epochs"]):
            model.train()
            running_loss = 0.0

            train_pg_bar = tqdm(train_loader, colour="cyan")

            for step, (images, targets) in enumerate(train_pg_bar):
                images = [img.to(device) for img in images]  # list of tensor
                targets = [
                    {k: v.to(device) for k, v in t.items()} for t in targets
                ]  # list of dict

                loss_dict = model(images, targets)  # forward w/ targets -> loss dict
                losses = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                running_loss += losses.item()

                # update progress bar
                train_pg_bar.set_description(
                    f"Train-epoch {epoch+1}/{tr_cfg['epochs']}.batch_loss: {losses.item():.2f}.avg_loss: {(running_loss/(step+1)):.2f}"
                )

                if step % 5 == 0:
                    mlflow.log_metrics(
                        {"train_batch_loss": losses.item()},
                        step=epoch * len(train_loader) + step,
                    )

            epoch_loss = running_loss / max(1, len(train_loader))
            mlflow.log_metric("train_loss", epoch_loss, step=epoch)

            # validation
            val_loss_sum = 0.0
            with torch.no_grad():
                val_pg_bar = tqdm(val_loader, colour="blue")
                for i, (images, targets) in enumerate(val_pg_bar):
                    images = [img.to(device) for img in images]
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    loss_dict = model(images, targets)
                    batch_loss = sum(loss for loss in loss_dict.values())
                    val_loss_sum += batch_loss

                    val_pg_bar.set_description(
                        f"Val-epoch {epoch+1}/{tr_cfg['epochs']}.batch_loss: {batch_loss.item():.2f}.avg_loss: {(val_loss_sum/(i+1)):.2f}"
                    )

            val_loss = val_loss_sum / max(1, len(val_loader))
            mlflow.log_metric("val_loss", val_loss, step=epoch)

            # save best model
            if best_val_loss < 0 or val_loss < best_val_loss:
                best_val_loss = val_loss

                ckpt_path = f'{tr_cfg["output_path"]}/best.pt'
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "val_loss": best_val_loss,
                    },
                    ckpt_path,
                )
                mlflow.log_artifact(ckpt_path)

            # log last model every 5 epoch
            if (epoch + 1) % 5 == 0:
                last_ckpt = f"{tr_cfg['output_path']}/last.pt"
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                    },
                    last_ckpt,
                )
                mlflow.log_artifact(last_ckpt)
