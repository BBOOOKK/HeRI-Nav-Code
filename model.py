"""PyTorch definition of the exported HeRI-Nav mean predictor."""
from pathlib import Path
import torch
from torch import nn

class Chomp1d(nn.Module):
    """裁剪因果卷积的填充"""
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TCNBlock(nn.Module):
    """TCN残差块"""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int,
                 dilation: int, dropout: float = 0.1):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation),
            Chomp1d(pad),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation),
            Chomp1d(pad),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.act = nn.GELU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.act(out + res)

class TCN(nn.Module):
    """时序卷积网络"""
    def __init__(self, in_ch: int, channels: list, kernel_size: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(channels):
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, 2 ** i, dropout))
            in_ch = out_ch
        self.network = nn.Sequential(*layers)
        self.out_ch = channels[-1]

    def forward(self, x):
        # x: (B, T, C) -> (B, C, T) -> (B, T, C)
        return self.network(x.permute(0, 2, 1)).permute(0, 2, 1)

class SpatioTemporalNet(nn.Module):
    """时空特征提取骨干网络

    架构：TCN -> BiLSTM -> Projection -> Multi-Head Attention
    """
    def __init__(self,
                 n_features: int = 14,
                 tcn_channels: list = None,
                 tcn_kernel: int = 3,
                 lstm_hidden: int = 128,
                 lstm_layers: int = 2,
                 mha_heads: int = 4,
                 mha_dim: int = 256,
                 attention_enabled: bool = True,
                 attention_bypass_mode: str = "mean_pooling",
                 dropout: float = 0.3):
        super().__init__()
        if tcn_channels is None:
            tcn_channels = [64, 64, 128, 128]
        if attention_bypass_mode not in {"mean_pooling", "last_hidden_state"}:
            raise ValueError(f"Unsupported attention_bypass_mode: {attention_bypass_mode}")
        self.attention_enabled = bool(attention_enabled)
        self.attention_bypass_mode = str(attention_bypass_mode)

        self.tcn = TCN(n_features, tcn_channels, tcn_kernel, dropout)
        tcn_out = tcn_channels[-1]

        self.bilstm = nn.LSTM(
            input_size=tcn_out,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )
        lstm_out = lstm_hidden * 2

        self.proj = nn.Linear(lstm_out, mha_dim)
        if self.attention_enabled:
            self.mha = nn.MultiheadAttention(
                embed_dim=mha_dim,
                num_heads=mha_heads,
                dropout=dropout,
                batch_first=True
            )
        else:
            self.mha = None
        self.out_dim = mha_dim

    def forward(self, x):
        """
        Args:
            x: (B, T, C) 输入特征

        Returns:
            out: (B, D) 输出特征
        """
        x = self.tcn(x)
        x, _ = self.bilstm(x)
        x = self.proj(x)
        if not self.attention_enabled:
            if self.attention_bypass_mode == "last_hidden_state":
                return x[:, -1, :]
            return x.mean(dim=1)
        q = x[:, -1:, :]
        out, _ = self.mha(q, x, x)
        return out.squeeze(1)

class MeanPredictor(nn.Module):
    """Physical-unit residual mean with embedded source normalization."""
    def __init__(self):
        super().__init__()
        self.backbone = SpatioTemporalNet()
        self.shared_head = nn.Sequential(nn.Linear(256, 128), nn.GELU(),
                                         nn.Linear(128, 128), nn.GELU())
        self.mu_head = nn.Linear(128, 2)
        self.register_buffer("x_mean", torch.zeros(14))
        self.register_buffer("x_std", torch.ones(14))
        self.register_buffer("y8_mean", torch.zeros(2))
        self.register_buffer("y8_std", torch.ones(2))

    def forward(self, x):
        hidden = self.shared_head(self.backbone((x-self.x_mean)/self.x_std))
        return self.mu_head(hidden)*self.y8_std+self.y8_mean


def load_model(path=None):
    path = Path(path) if path else Path(__file__).with_name("model.pt")
    exported = torch.jit.load(str(path), map_location="cpu").eval()
    model = MeanPredictor().eval()
    model.load_state_dict(exported.state_dict(), strict=True)
    return model
