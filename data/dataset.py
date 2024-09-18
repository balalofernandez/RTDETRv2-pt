

from data.collator import BatchImageCollateFuncion, batch_image_collate_fn
from .transforms import (
    Compose,
    RandomPhotometricDistort,
    RandomZoomOut,
    RandomIoUCrop,
    SanitizeBoundingBoxes,
    RandomHorizontalFlip,
    Resize,
    ConvertPILImage,
    ConvertBoxes,
)
from data.coco_dataset import CocoDetection
from torch.utils.data import DataLoader, random_split


def get_dataloaders(args):
    tile_size = args.size
    train_transforms = [
        RandomPhotometricDistort(p=0.5),
        RandomZoomOut(fill=0),
        RandomIoUCrop(p=0.8),
        SanitizeBoundingBoxes(min_size=1),
        RandomHorizontalFlip(),
        Resize(size=[tile_size, tile_size]),
        SanitizeBoundingBoxes(min_size=1),
        ConvertPILImage(dtype="float32", scale=True),
        ConvertBoxes(fmt="cxcywh", normalize=True),
    ]
    train_policy = {
        "name": "default",
        "epoch": 71,
        "ops": [RandomPhotometricDistort(), RandomZoomOut(), RandomIoUCrop()],
    }
    val_transforms = [
        Resize(size=[tile_size, tile_size]),
        ConvertPILImage(dtype="float32", scale=True),
        ConvertBoxes(fmt="cxcywh", normalize=True),
    ]
    val_policy = None
    train_composed_transforms = Compose(
                transforms=train_transforms, policy=train_policy
            )
    val_composed_transforms = Compose(
                transforms=val_transforms, policy=val_policy
            )

    train_dataset = CocoDetection(
        args.train_img_root,args.train_annot_root, transforms=train_composed_transforms
    )

    val_dataset = CocoDetection(args.val_img_root,args.val_annot_root, transforms=val_composed_transforms)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=True,
        collate_fn=batch_image_collate_fn
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        drop_last=True,
        collate_fn=batch_image_collate_fn
    )

    return train_dataloader, val_dataloader