import torch
import torch.nn as nn


class MotionStyleDiscriminator(nn.Module):
    """Conditional sequence discriminator for generated motion rollouts.

    Inputs are expected as batched sequences:
      motion_sequence: (B, T, motion_dim)
      content_condition/style_condition: optional (B, C) or (B, T, C)

    The module returns one logit per sequence. Training code can interpret larger
    logits as more likely to be real/reference motion.
    """

    def __init__(
        self,
        motion_dim,
        condition_dim=0,
        hidden_dim=256,
        num_heads=4,
        num_layers=2,
        dim_feedforward=None,
        dropout=0.1,
        max_sequence_length=64,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            hidden_dim = ((hidden_dim + num_heads - 1) // num_heads) * num_heads

        self.motion_dim = motion_dim
        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim
        self.max_sequence_length = max_sequence_length

        self.motion_proj = nn.Linear(motion_dim, hidden_dim)
        self.condition_proj = nn.Linear(condition_dim, hidden_dim) if condition_dim > 0 else None
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_sequence_length + 3, hidden_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward or hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def _condition_token(self, condition):
        if condition is None or self.condition_proj is None:
            return None
        if condition.dim() == 3:
            condition = condition.mean(dim=1)
        return self.condition_proj(condition).unsqueeze(1)

    def forward(self, motion_sequence, content_condition=None, style_condition=None, padding_mask=None):
        batch_size, sequence_length, _ = motion_sequence.shape
        tokens = [self.cls_token.expand(batch_size, -1, -1)]

        content_token = self._condition_token(content_condition)
        if content_token is not None:
            tokens.append(content_token)

        style_token = self._condition_token(style_condition)
        if style_token is not None:
            tokens.append(style_token)

        tokens.append(self.motion_proj(motion_sequence))
        x = torch.cat(tokens, dim=1)
        if x.shape[1] > self.pos_embedding.shape[1]:
            raise ValueError(
                f"Sequence too long for discriminator: got {x.shape[1]} tokens, "
                f"max {self.pos_embedding.shape[1]}"
            )
        x = x + self.pos_embedding[:, : x.shape[1]]

        if padding_mask is not None:
            prefix_length = x.shape[1] - sequence_length
            prefix_mask = torch.zeros(batch_size, prefix_length, dtype=torch.bool, device=padding_mask.device)
            padding_mask = torch.cat([prefix_mask, padding_mask.bool()], dim=1)

        encoded = self.encoder(x, src_key_padding_mask=padding_mask)
        return self.head(encoded[:, 0]).squeeze(-1)
