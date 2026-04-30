from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct3d(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: Tuple[int, int, int] = (3, 3, 3),
        s: Tuple[int, int, int] = (1, 1, 1),
        p: Tuple[int, int, int] = (1, 1, 1),
        act: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm3d(out_ch)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ResBlock3d(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: Tuple[int, int, int] = (1, 1, 1),
    ):
        super().__init__()
        self.conv1 = ConvBNAct3d(in_ch, out_ch, k=(3, 3, 3), s=stride, p=(1, 1, 1), act=True)
        self.conv2 = ConvBNAct3d(out_ch, out_ch, k=(3, 3, 3), s=(1, 1, 1), p=(1, 1, 1), act=False)
        if in_ch != out_ch or stride != (1, 1, 1):
            self.short = ConvBNAct3d(in_ch, out_ch, k=(1, 1, 1), s=stride, p=(0, 0, 0), act=False)
        else:
            self.short = nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + self.short(x)
        return self.act(out)


class TemporalAlignFastToSlow(nn.Module):
    def forward(self, x_fast: torch.Tensor, t_slow: int) -> torch.Tensor:
        if x_fast.shape[2] == t_slow:
            return x_fast
        return F.adaptive_avg_pool3d(x_fast, output_size=(t_slow, x_fast.shape[3], x_fast.shape[4]))


class FastToSlowFuse(nn.Module):
    def __init__(self, c_fast: int, c_slow: int, mode: str = "add"):
        super().__init__()
        mode = str(mode).lower()
        if mode not in ("add", "concat"):
            raise ValueError(f"Unsupported fuse mode: {mode}")
        self.mode = mode
        self.align = TemporalAlignFastToSlow()
        self.proj = ConvBNAct3d(c_fast, c_slow, k=(1, 1, 1), s=(1, 1, 1), p=(0, 0, 0), act=False)
        self.post = ConvBNAct3d(c_slow * 2, c_slow, k=(1, 1, 1), s=(1, 1, 1), p=(0, 0, 0), act=True) if mode == "concat" else None
        self.act = nn.ReLU(inplace=True)

    def forward(self, x_slow: torch.Tensor, x_fast: torch.Tensor) -> torch.Tensor:
        x_fast = self.align(x_fast, x_slow.shape[2])
        x_fast = self.proj(x_fast)
        if self.mode == "add":
            return self.act(x_slow + x_fast)
        x = torch.cat([x_slow, x_fast], dim=1)
        return self.post(x)


class EventSlowFastBackbone(nn.Module):
    def __init__(
        self,
        in_ch_slow: int = 2,
        in_ch_fast: int = 2,
        slow_base: int = 32,
        fast_base: int = 16,
        fuse_mode: str = "add",
    ):
        super().__init__()

        self.slow_stem = nn.Sequential(
            ConvBNAct3d(in_ch_slow, slow_base, k=(3, 5, 5), s=(1, 2, 2), p=(1, 2, 2)),
            ConvBNAct3d(slow_base, slow_base, k=(3, 3, 3), s=(1, 1, 1), p=(1, 1, 1)),
        )
        self.fast_stem = nn.Sequential(
            ConvBNAct3d(in_ch_fast, fast_base, k=(3, 5, 5), s=(1, 2, 2), p=(1, 2, 2)),
            ConvBNAct3d(fast_base, fast_base, k=(3, 3, 3), s=(1, 1, 1), p=(1, 1, 1)),
        )
        self.fuse1 = FastToSlowFuse(c_fast=fast_base, c_slow=slow_base, mode=fuse_mode)

        self.slow_stage1 = nn.Sequential(
            ResBlock3d(slow_base, slow_base * 2, stride=(1, 2, 2)),
            ResBlock3d(slow_base * 2, slow_base * 2, stride=(1, 1, 1)),
        )
        self.fast_stage1 = nn.Sequential(
            ResBlock3d(fast_base, fast_base * 2, stride=(1, 2, 2)),
            ResBlock3d(fast_base * 2, fast_base * 2, stride=(1, 1, 1)),
        )
        self.fuse2 = FastToSlowFuse(c_fast=fast_base * 2, c_slow=slow_base * 2, mode=fuse_mode)

        self.slow_stage2 = nn.Sequential(
            ResBlock3d(slow_base * 2, slow_base * 4, stride=(1, 2, 2)),
            ResBlock3d(slow_base * 4, slow_base * 4, stride=(1, 1, 1)),
        )
        self.fast_stage2 = nn.Sequential(
            ResBlock3d(fast_base * 2, fast_base * 4, stride=(2, 2, 2)),
            ResBlock3d(fast_base * 4, fast_base * 4, stride=(1, 1, 1)),
        )
        self.fuse3 = FastToSlowFuse(c_fast=fast_base * 4, c_slow=slow_base * 4, mode=fuse_mode)

        self.slow_stage3 = nn.Sequential(
            ResBlock3d(slow_base * 4, slow_base * 8, stride=(1, 2, 2)),
            ResBlock3d(slow_base * 8, slow_base * 8, stride=(1, 1, 1)),
        )
        self.fast_stage3 = nn.Sequential(
            ResBlock3d(fast_base * 4, fast_base * 8, stride=(2, 2, 2)),
            ResBlock3d(fast_base * 8, fast_base * 8, stride=(1, 1, 1)),
        )
        self.fuse4 = FastToSlowFuse(c_fast=fast_base * 8, c_slow=slow_base * 8, mode=fuse_mode)

        self.out_channels_slow = slow_base * 8
        self.out_channels_fast = fast_base * 8

    def forward(self, x_list: List[torch.Tensor]) -> List[torch.Tensor]:
        x_slow, x_fast = x_list

        x_slow = self.slow_stem(x_slow)
        x_fast = self.fast_stem(x_fast)
        x_slow = self.fuse1(x_slow, x_fast)

        x_slow = self.slow_stage1(x_slow)
        x_fast = self.fast_stage1(x_fast)
        x_slow = self.fuse2(x_slow, x_fast)

        x_slow = self.slow_stage2(x_slow)
        x_fast = self.fast_stage2(x_fast)
        x_slow = self.fuse3(x_slow, x_fast)

        x_slow = self.slow_stage3(x_slow)
        x_fast = self.fast_stage3(x_fast)
        x_slow = self.fuse4(x_slow, x_fast)

        return [x_slow, x_fast]


class EventSlowFastClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        slow_base: int = 32,
        fast_base: int = 16,
        fuse_mode: str = "add",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.backbone = EventSlowFastBackbone(
            in_ch_slow=2,
            in_ch_fast=2,
            slow_base=slow_base,
            fast_base=fast_base,
            fuse_mode=fuse_mode,
        )
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(p=float(dropout)) if float(dropout) > 0 else nn.Identity()
        self.emb_dim = self.backbone.out_channels_slow + self.backbone.out_channels_fast
        self.classifier = nn.Linear(self.emb_dim, int(num_classes))

    def forward_embed(self, x_list: List[torch.Tensor]) -> torch.Tensor:
        feat_s, feat_f = self.backbone(x_list)
        slow = self.pool(feat_s).flatten(1)
        fast = self.pool(feat_f).flatten(1)
        emb = torch.cat([slow, fast], dim=1)
        emb = self.dropout(emb)
        return emb

    def forward(self, x_list: List[torch.Tensor]) -> torch.Tensor:
        emb = self.forward_embed(x_list)
        return self.classifier(emb)
