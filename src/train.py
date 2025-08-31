from argparse import ArgumentParser
import yaml
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision import transforms
import random
import tqdm
from bccd_dataset import BCCD_DATASET
from model import BCCD_Model
from utils.load_config import load_config
from utils.visualize_img import visualize_img
from utils.transforms_albu import build_train_aug_albu, build_val_aug_albu
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2 


def get_args():
    """Parse CLI arguments"""
    parser = ArgumentParser(description='args for training')
    parser.add_argument("--config", type=str, default='config.yaml', help='path to config file')
    parser.add_argument("--epochs", type=int, default=10, help='number of epochs')
    parser.add_argument("--batch_size", type=int, default=4, help='batch size')
    parser.add_argument("--tracking_uri", type=str, default='http://127.0.1:5000', help='MLflow tracking URI')
    parser.add_argument("--checkpoint", "-c", type=str, default='models/checkpoint.pt', help='path to save checkpoint')
    args = parser.parse_args()
    return args

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

if __name__ == "__main__":
    # load CLI & config yaml
    args = get_args()
    config = load_config(args.config )

    ds_cfg = config['dataset']
    tr_cfg = config['training']
    mdl_cfg = config['model']
    aug_cfg = config['augmentation']
    log_cfg = config['logging']

    root = ds_cfg['root']  # "BCCD_Dataset"
    image_size = ds_cfg['image_size']
    
    # use CLI if different from YAML
    epochs = args.epochs if args.epochs != tr_cfg['epochs'] else tr_cfg['epochs']
    batch_size = args.batch_size if args.batch_size != tr_cfg['batch_size'] else tr_cfg['batch_size']
    tracking_uri = log_cfg['tracking_uri']
    
    lr = tr_cfg['learning_rate']
    optimizer_nm = tr_cfg['optimizer']
    weight_decay = tr_cfg['weight_decay']
    momentum = tr_cfg['momentum']

    num_classes  = mdl_cfg['num_classes']  # including background

    # check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # augmentations
    train_transform = build_train_aug_albu(aug_cfg, image_size)
    val_transform   = build_val_aug_albu(image_size)

    # datasets (train/val/test)
    train_dataset = BCCD_DATASET(root=root, mode="train", transform=train_transform)
    val_dataset   = BCCD_DATASET(root=root, mode="val",   transform=val_transform)
    test_dataset  = BCCD_DATASET(root=root, mode="test",  transform=val_transform)

    # dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=2, collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                              num_workers=2, collate_fn=collate_fn)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False,
                              num_workers=2, collate_fn=collate_fn)

    # model & Optimizer
    model = BCCD_Model(num_classes=num_classes).model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = build_optimizer(optimizer_nm, params, lr, weight_decay, momentum)

    # MLflow
    if log_cfg['use_mlflow']:
        mlflow.set_tracking_uri(tracking_uri)   

    with mlflow.start_run(run_name=config['experiment_name']):
        mlflow.log_params({
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "optimizer": optimizer_nm,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "image_size": image_size,
            "num_classes": num_classes
        })
        
        # log model summary
        try:
            with open("model_summary.txt", "w") as f:
                f.write(str(summary(model)))
            mlflow.log_artifact("model_summary.txt")
        except Exception as e:
            print("Summary(model) failed:", e)

        # define best val loss
        best_val_loss = -1
        num_iterations = len(train_loader)
        
        # train loop -> dict loss
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            
            train_pg_bar = tqdm.tqdm(train_loader, colour="cyan",
                         desc=f"Train - Epoch {epoch+1}/{epochs}")

            for step, (images, targets) in enumerate(train_pg_bar):
                images  = [img.to(device) for img in images]    # list of tensor
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]    # list of dict

                loss_dict = model(images, targets)         # forward w/ targets -> loss dict
                losses = sum(loss for loss in loss_dict.values())
            
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                running_loss += losses.item()

                train_pg_bar.set_postfix(batch_loss=f"{losses.item():.4f}",
                             avg_loss=f"{(running_loss/(step+1)):.4f}")
                
                if step % 5 == 0:
                    mlflow.log_metrics({"train_batch_loss": losses.item()},
                                       step=epoch * len(train_loader) + step)

            epoch_loss = running_loss / max(1, len(train_loader))
            mlflow.log_metric("train_loss", epoch_loss, step=epoch)

            # validation
            val_loss_sum = 0.0
            with torch.no_grad():
                val_pg_bar = tqdm.tqdm(val_loader, colour="blue")
                for i, (images, targets) in enumerate(val_loader):
                    images  = [img.to(device) for img in images]
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    loss_dict = model(images, targets)
                    batch_loss = sum(loss for loss in loss_dict.values())
                    val_loss_sum += batch_loss
                    
                    val_pg_bar.set_postfix(batch_loss=f"{batch_loss:.4f}",
                               avg_loss=f"{(val_loss_sum/i):.4f}")
                    
            val_loss = val_loss_sum / max(1, len(val_loader))
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            
            # save best model
            if best_val_loss < 0 or val_loss < best_val_loss:
                best_val_loss = val_loss

                # save low checkpoint
                ckpt_path = f"outputs/best_epoch_{epoch+1}.pt"
                torch.save({
                    "epoch": epoch+1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "config": config,
                }, ckpt_path)
                mlflow.log_artifact(ckpt_path)

            # log last model every 5 epoch
            if (epoch + 1) % 5 == 0:
                last_ckpt = f"outputs/last_epoch.pt"
                torch.save({"epoch": epoch+1, "model_state": model.state_dict(),
                            "optimizer_state": optimizer.state_dict()}, last_ckpt)
                mlflow.log_artifact(last_ckpt)

        # test
        model.eval()
        all_preds = []
        with torch.no_grad():
            for images, _ in test_loader:
                images = [img.to(device) for img in images]
                preds = model(images) # list[dict]: boxes, labels, scores
                all_preds.extend([{k: v.cpu() for k, v in p.items()} for p in preds])

        print(f"Done. Collected {len(all_preds)} test predictions.")