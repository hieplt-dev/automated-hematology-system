import albumentations as A
from albumentations.pytorch import ToTensorV2


def build_train_aug_albu(aug_cfg, image_size):
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=aug_cfg.get("hflip_p", 0.5)),
            A.VerticalFlip(p=aug_cfg.get("vflip_p", 0.5)),
            A.RandomRotate90(p=aug_cfg.get("rotate90_p", 0.5)),
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(0.0, 0.05),
                rotate=0, shear=0,
                p=aug_cfg.get("affine_p", 0.4),
            ),

            A.RandomBrightnessContrast(
                brightness_limit=aug_cfg.get("brightness", 0.2),
                contrast_limit=aug_cfg.get("contrast", 0.15),
                p=0.4
            ),
            A.ColorJitter(p=0.2),
            A.GaussianBlur(blur_limit=(3,5), sigma_limit=(0.1,1.0), p=0.1),

            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
            min_visibility=0.3,
            clip=True,
        ),
    )

def build_val_aug_albu(image_size):
    return A.Compose(
        [A.Resize(image_size, image_size), ToTensorV2()],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"], clip=True),
    )
