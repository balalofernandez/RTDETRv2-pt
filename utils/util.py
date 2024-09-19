import torch
import torch.nn as nn
import numpy as np
import random
from torchvision.ops.boxes import box_area
import copy
import re
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/rtdetrv2.yml",
        help="path to config file",
    )
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--train', action='store_true', default=True)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--checkpoint_path', default="./ckpts/rtdetrv2_r50vd_6x_coco_ema.pth", type=str)
    parser.add_argument('--train_img_root', default="/home/balalo/AIPlayground/data/COCO/val2017/", type=str)
    parser.add_argument('--train_annot_root', default="/home/balalo/AIPlayground/data/COCO/annotations/instances_val2017.json", type=str)
    parser.add_argument('--val_img_root', default="/home/balalo/AIPlayground/data/COCO/val2017/", type=str)
    parser.add_argument('--val_annot_root', default="/home/balalo/AIPlayground/data/COCO/annotations/instances_val2017.json", type=str)
    parser.add_argument('--size', default=256, type=int) #640
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return args



def get_optim_params(cfg: dict, model: nn.Module):
    """
    E.g.:
        ^(?=.*a)(?=.*b).*$  means including a and b
        ^(?=.*(?:a|b)).*$   means including a or b
        ^(?=.*a)(?!.*b).*$  means including a, but not b
    """
    assert "type" in cfg, ""
    cfg = copy.deepcopy(cfg)

    if "params" not in cfg:
        return model.parameters()

    assert isinstance(cfg["params"], list), ""

    param_groups = []
    visited = []
    for pg in cfg["params"]:
        pattern = pg["params"]
        params = {
            k: v
            for k, v in model.named_parameters()
            if v.requires_grad and len(re.findall(pattern, k)) > 0
        }
        pg["params"] = params.values()
        param_groups.append(pg)
        visited.extend(list(params.keys()))
        # print(params.keys())

    names = [k for k, v in model.named_parameters() if v.requires_grad]

    if len(visited) < len(names):
        unseen = set(names) - set(visited)
        params = {
            k: v for k, v in model.named_parameters() if v.requires_grad and k in unseen
        }
        param_groups.append({"params": params.values()})
        visited.extend(list(params.keys()))
        # print(params.keys())

    assert len(visited) == len(names), ""

    return param_groups



def setup_seed(val=0):
    """
    Setup random seed.
    """
    np.random.seed(val)
    random.seed(val)
    torch.manual_seed(val)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
