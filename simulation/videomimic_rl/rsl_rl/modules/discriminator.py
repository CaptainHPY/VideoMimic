import torch
import torch.nn as nn


class LatentStyleDiscriminator(nn.Module):
    """Multi-head discriminator for generated policy latents."""

    def __init__(self, latent_dim, num_style_labels, hidden_dim=256):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.num_style_labels = int(num_style_labels)
        self.hidden_dim = int(hidden_dim)
        if self.num_style_labels <= 1:
            raise ValueError("num_style_labels must be greater than one")

        self.input_norm = nn.LayerNorm(self.latent_dim)
        self.backbone = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(0.2),
        )
        self.head = nn.Linear(self.hidden_dim, self.num_style_labels)

    def forward_all_logits(self, latent):
        return self.head(self.backbone(self.input_norm(latent)))

    def forward(self, latent, style_label):
        all_logits = self.forward_all_logits(latent)
        style_label = style_label.long().reshape(-1)
        if style_label.shape[0] != latent.shape[0]:
            raise ValueError(
                f"Expected {latent.shape[0]} style labels, got {style_label.shape[0]}"
            )
        if (style_label < 0).any() or (style_label >= self.num_style_labels).any():
            raise ValueError(
                f"style_label must be in [0, {self.num_style_labels - 1}]"
            )
        return all_logits.gather(1, style_label.unsqueeze(1)).squeeze(1)


class MotionStyleDiscriminator(nn.Module):
    """Discrete-style conditional, multi-head sequence discriminator.

    Inputs are expected as batched sequences:
      motion_sequence: (B, T, motion_dim)

    The final head emits one real/fake logit per style category. ``style_label``
    selects the relevant head for each sequence, matching the category-specific
    discriminator used by the original style-transfer model.
    """

    def __init__(
        self,
        motion_dim,
        num_style_labels,
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
        self.num_style_labels = int(num_style_labels)
        if self.num_style_labels <= 0:
            raise ValueError("num_style_labels must be positive")
        self.hidden_dim = hidden_dim
        self.max_sequence_length = max_sequence_length

        self.motion_proj = nn.Linear(motion_dim, hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_sequence_length + 1, hidden_dim) * 0.02)

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
            nn.Linear(hidden_dim, self.num_style_labels),
        )

    def forward_all_logits(self, motion_sequence, padding_mask=None):
        batch_size, sequence_length, _ = motion_sequence.shape
        tokens = [self.cls_token.expand(batch_size, -1, -1)]
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
        return self.head(encoded[:, 0])

    def forward(self, motion_sequence, style_label, padding_mask=None):
        all_logits = self.forward_all_logits(motion_sequence, padding_mask=padding_mask)
        batch_size = motion_sequence.shape[0]
        style_label = style_label.long().reshape(-1)
        if style_label.shape[0] != batch_size:
            raise ValueError(
                f"Expected {batch_size} style labels, got {style_label.shape[0]}"
            )
        if (style_label < 0).any() or (style_label >= self.num_style_labels).any():
            raise ValueError(
                f"style_label must be in [0, {self.num_style_labels - 1}]"
            )
        return all_logits.gather(1, style_label.unsqueeze(1)).squeeze(1)
