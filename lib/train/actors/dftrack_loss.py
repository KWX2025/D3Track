

from torch import nn, Tensor
from torch.nn.functional import l1_loss, mse_loss
import torch


class BaseLoss():
    NAME = None

    def __init__(self, content_level="channel", style_level="channel") -> None:
        self.content_level = content_level
        self.style_level = style_level

    def content_distill(self):
        raise ImportError

    def style_distill(self):
        raise ImportError


class DFTrack_loss(BaseLoss):
    NAME = "DFTrack"

    def __init__(self, num_features=768, content_level="channel", style_level="channel"):
        super().__init__(content_level, style_level)
        self.instanceNorm = nn.InstanceNorm1d(num_features)
        self.dualDistill_loss = mse_loss

    def content_distill(self, x: Tensor, y: Tensor, **arg_dict):
        if self.content_level == "channel":
            x = self.instanceNorm(x.transpose(-1, -2))
            y = self.instanceNorm(y.transpose(-1, -2))
        elif self.content_level == "token":
            x = self.instanceNorm(x)
            y = self.instanceNorm(y)
        return self.dualDistill_loss(x, y)
    def cosine_distill(self, x: torch.Tensor, y: torch.Tensor):
        
        x = x.view(x.size(0), x.size(1), -1)
        y = y.view(y.size(0), y.size(1), -1)
        x = nn.functional.normalize(x, dim=1)
        y = nn.functional.normalize(y, dim=1)
        return 1 - (x * y).sum(1).mean()

    def cross_distill_high(self, x: torch.Tensor, y: torch.Tensor, score_map_gt: torch.Tensor, **arg_dict):
        
        x = self.instanceNorm(x.transpose(-1, -2))  
        y = self.instanceNorm(y.transpose(-1, -2))  

        return self.cosine_distill(x, y)


    def cross_distill_low(self, x: Tensor, y: Tensor, **arg_dict):
        if self.style_level == "channel":
            mx = x.mean(-2)
            my = y.mean(-2)
            stdx = x.std(-2)
            stdy = y.std(-2)
        elif self.style_level == "token":
            mx = x.mean(-1)
            my = y.mean(-1)
            stdx = x.std(-1)
            stdy = y.std(-1)
        return ((mx - my) ** 2 + (stdx - stdy) ** 2).mean()




def get_dftrack_loss(cfg):
    name = cfg.TRAIN.DFTrack_LOSS
    content_level = cfg.TRAIN.CONTENT_LOSS_TYPE
    style_level = cfg.TRAIN.STYLE_LOSS_TYPE
    if name == None or name == "":
        name = DFTrack_loss.NAME

    if name == DFTrack_loss.NAME:
        return DFTrack_loss(content_level=content_level, style_level=style_level)


    raise "error dftrack loss type."