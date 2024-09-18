"""
Extracted from RTDETRv2 original repo: 
    https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch
"""
import torch.nn as nn
import torch

from nets.backbone import PResNet
from nets.encoder import HybridEncoder
from nets.decoder import RTDETRTransformerv2

def get_model(args, config):
    backbone_params = config["PResNet"]
    encoder_params = config["HybridEncoder"]
    decoder_params = config["RTDETRTransformerv2"]
    num_classes = config["num_classes"]

    backbone = PResNet(
        depth=backbone_params["depth"],
        variant=backbone_params["variant"],
        num_stages=backbone_params["num_stages"],
        return_idx=backbone_params["return_idx"],
        freeze_at=backbone_params["freeze_at"],
        freeze_norm=backbone_params["freeze_norm"],
        pretrained=backbone_params["pretrained"],
    )
    encoder = HybridEncoder(
        in_channels=encoder_params["in_channels"],
        feat_strides=encoder_params["feat_strides"],
        hidden_dim=encoder_params["hidden_dim"],
        nhead=encoder_params["nhead"],
        dim_feedforward=encoder_params["dim_feedforward"],
        dropout=encoder_params["dropout"],
        enc_act=encoder_params["enc_act"],
        use_encoder_idx=encoder_params["use_encoder_idx"],
        num_encoder_layers=encoder_params["num_encoder_layers"],
        expansion=encoder_params["expansion"],
        depth_mult=encoder_params["depth_mult"],
        act=encoder_params["act"],
    )
    decoder = RTDETRTransformerv2(
        num_classes=num_classes,
        hidden_dim=decoder_params["hidden_dim"],
        num_queries=decoder_params["num_queries"],
        feat_channels=decoder_params["feat_channels"],
        feat_strides=decoder_params["feat_strides"],
        num_levels=decoder_params["num_levels"],
        num_points=decoder_params["num_points"],
        num_layers=decoder_params["num_layers"],
        num_denoising=decoder_params["num_denoising"],
        label_noise_ratio=decoder_params["label_noise_ratio"],
        box_noise_scale=decoder_params["box_noise_scale"],
        eval_idx=decoder_params["eval_idx"],
        cross_attn_method=decoder_params["cross_attn_method"],
        query_select_method=decoder_params["query_select_method"],
    )

    model = RTDETR(backbone=backbone, encoder=encoder, decoder=decoder)
    if args.checkpoint_path != "":
        state_dict = torch.load(args.checkpoint_path, weights_only=True)['ema']['module']
    
        filtered_dict = {k: v for k, v in state_dict.items() if k in model.state_dict() and v.shape == model.state_dict()[k].shape}
    
        model.load_state_dict(filtered_dict, strict=False)
    return model

class RTDETR(nn.Module):

    __inject__ = [
        "backbone",
        "encoder",
        "decoder",
    ]

    def __init__(
        self,
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder

    def forward(self, x, targets=None):
        x = self.backbone(x)
        x = self.encoder(x)
        x = self.decoder(x, targets)

        return x

    def deploy(
        self,
    ):
        self.eval()
        for m in self.modules():
            if hasattr(m, "convert_to_deploy"):
                m.convert_to_deploy()
        return self

