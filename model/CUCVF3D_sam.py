"""
SAM-Med3D + residual 3D U-Net, AGE, hierarchical variational fusion, and
evidential uncertainty refinement (EUR). The uncertainty feedback is a
training-time KL constraint; it does not require a recurrent forward pass.

    model = CUCVF3D_sam(classes=2, channels=1)
    logits = model(image)                         # [B,K,D,H,W]
    output = model(image, return_aux=True)
    losses = CUCVFCompositeLoss(...)(output, label, epoch=epoch)
    losses["total"].backward()
"""

from pathlib import Path
import math

import torch
from torch import nn
import torch.nn.functional as F


def downsample():
    return nn.MaxPool3d(kernel_size=2, stride=2)


def deconv(in_channels, out_channels):
    return nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)


def initialize_weights(*models):
    """He-initialize new modules only; never pass the pretrained SAM here."""
    for model in models:
        for module in model.modules():
            if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)


def _resize(x, size):
    if tuple(x.shape[-3:]) == tuple(size):
        return x
    return F.interpolate(x, size=tuple(size), mode="trilinear", align_corners=False)


def _sam_attention_3d(attention, x):
    """SAM self-attention with correctly aligned D/H/W relative position axes.

    Some bundled SAM-Med3D versions broadcast relative positions on the wrong
    axes and fail on rectangular grids. Use the original parameters locally,
    without modifying the SAM package, its state dict, or any global method.
    """
    batch, depth, height, width, channels = x.shape
    shape = (depth, height, width)
    tokens = depth * height * width
    heads = attention.num_heads
    qkv = attention.qkv(x).reshape(batch, tokens, 3, heads, -1)
    qkv = qkv.permute(2, 0, 3, 1, 4).reshape(3, batch * heads, tokens, -1)
    query, key, value = qkv.unbind(0)
    scores = (query * attention.scale) @ key.transpose(-2, -1)
    if getattr(attention, "use_rel_pos", False):
        relative = []
        volume_query = query.reshape(batch * heads, *shape, -1)
        for size, name, axis in zip(shape, ("rel_pos_d", "rel_pos_h", "rel_pos_w"),
                                    ("d", "h", "w")):
            table = getattr(attention, name)
            length = 2 * size - 1
            if table.shape[0] != length:
                table = F.interpolate(table.T[None], size=length, mode="linear",
                                      align_corners=False)[0].T
            coordinate = torch.arange(size, device=x.device)
            index = coordinate[:, None] - coordinate[None, :] + size - 1
            positions = table[index]
            relative.append(torch.einsum("bdhwc," + axis + "kc->bdhwk",
                                         volume_query, positions))
        scores = scores.reshape(batch * heads, *shape, *shape)
        scores = (scores + relative[0][..., :, None, None]
                  + relative[1][..., None, :, None] + relative[2][..., None, None, :])
        scores = scores.reshape(batch * heads, tokens, tokens)
    weights = scores.softmax(dim=-1)
    context = (weights @ value).reshape(batch, heads, *shape, -1)
    context = context.permute(0, 2, 3, 4, 1, 5).reshape(batch, *shape, channels)
    return attention.proj(context)


def _sam_block_3d(block, x):
    """Execute standard SAM blocks while keeping the relative-position fix local."""
    attention = getattr(block, "attn", None)
    required = ("qkv", "proj", "num_heads", "scale")
    if attention is None or not all(hasattr(attention, name) for name in required):
        return block(x)
    shortcut = x
    x = block.norm1(x)
    window = getattr(block, "window_size", 0)
    if window:
        batch, depth, height, width, channels = x.shape
        pad_d, pad_h, pad_w = (-depth) % window, (-height) % window, (-width) % window
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))
        dp, hp, wp = depth + pad_d, height + pad_h, width + pad_w
        x = x.reshape(batch, dp // window, window, hp // window, window,
                      wp // window, window, channels)
        x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).reshape(-1, window, window, window, channels)
    x = _sam_attention_3d(attention, x)
    if window:
        x = x.reshape(batch, dp // window, hp // window, wp // window,
                      window, window, window, channels)
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).reshape(batch, dp, hp, wp, channels)
        x = x[:, :depth, :height, :width].contiguous()
    x = shortcut + x
    return x + block.mlp(block.norm2(x))


def _attention_grid(shape, max_tokens):
    """Bound quadratic attention storage while preserving volume aspect ratio."""
    shape = tuple(int(s) for s in shape)
    if max_tokens is None or math.prod(shape) <= max_tokens:
        return shape
    scale = (max_tokens / math.prod(shape)) ** (1.0 / 3.0)
    grid = [max(1, int(s * scale)) for s in shape]
    while math.prod(grid) > max_tokens:
        axis = max(range(3), key=lambda i: grid[i])
        grid[axis] -= 1
    return tuple(grid)


class ResEncoder3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.conv1x1 = nn.Conv3d(in_channels, out_channels, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.conv1x1(x)
        x = self.relu(self.bn1(self.conv1(x)))
        return self.relu(self.bn2(self.conv2(x)) + residual)


class Decoder3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class AttentionGuidedEnhancement3d(nn.Module):
    """Fig. 2(b): sigmoid(Q^T K), value aggregation, residual, 384 -> 256."""

    def __init__(self, in_channels=384, out_channels=256, max_tokens=1024):
        super().__init__()
        if max_tokens is not None and max_tokens < 1:
            raise ValueError("AGE max_tokens must be positive or None")
        self.query = nn.Conv3d(in_channels, in_channels, 1)
        self.key = nn.Conv3d(in_channels, in_channels, 1)
        self.value = nn.Conv3d(in_channels, in_channels, 1)
        self.projection = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.max_tokens = max_tokens

    def forward(self, x):
        shape = x.shape[-3:]
        grid = _attention_grid(shape, self.max_tokens)
        pooled = x if tuple(shape) == grid else F.adaptive_avg_pool3d(x, grid)
        query = self.query(pooled).flatten(2).transpose(1, 2)
        key = self.key(pooled).flatten(2)
        # Fig. 2 specifies sigmoid rather than a softmax over spatial tokens.
        attention = torch.sigmoid(torch.bmm(query.float(), key.float()))
        value = self.value(pooled).flatten(2)
        context = torch.bmm(value.float(), attention.transpose(1, 2))
        context = context.reshape_as(pooled).to(x.dtype)
        return self.projection(x + _resize(context, shape))


class SpatialAttentionBlock3d(nn.Module):
    """CS2Net spatial affinity residual used by EUR (Eq. 18).

    The existing repository's query/key/judge spatial affinity is retained.
    Attention runs on a bounded grid by default: a full-volume N x N matrix
    at 128 x 160 x 160 is impractical. Set max_tokens=None for unpooled SAB.
    This method returns the residual contribution only, so EUR adds x once.
    """

    def __init__(self, channels, max_tokens=512):
        super().__init__()
        if max_tokens is not None and max_tokens < 1:
            raise ValueError("SAB max_tokens must be positive or None")
        hidden = max(1, channels // 8)
        self.query = nn.Conv3d(channels, hidden, (1, 3, 1), padding=(0, 1, 0))
        self.key = nn.Conv3d(channels, hidden, (3, 1, 1), padding=(1, 0, 0))
        self.judge = nn.Conv3d(channels, hidden, (1, 1, 3), padding=(0, 0, 1))
        self.value = nn.Conv3d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.max_tokens = max_tokens

    def forward(self, x):
        shape = x.shape[-3:]
        grid = _attention_grid(shape, self.max_tokens)
        pooled = x if tuple(shape) == grid else F.adaptive_avg_pool3d(x, grid)
        query = self.query(pooled).flatten(2).transpose(1, 2).float()
        key = self.key(pooled).flatten(2).float()
        judge = self.judge(pooled).flatten(2).transpose(1, 2).float()
        # Q K J K = Q (K J) K avoids multiplying two N x N matrices.
        scores = torch.bmm(torch.bmm(query, torch.bmm(key, judge)), key)
        attention = scores.softmax(dim=-1)
        value = self.value(pooled).flatten(2).float()
        context = torch.bmm(value, attention).reshape_as(pooled).to(x.dtype)
        return self.gamma * _resize(context, shape)


class GaussianEncoder3d(nn.Module):
    """Eqs. 2-5; sigma is a standard deviation, never a variance/log-variance."""

    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.mean = nn.Conv3d(channels, channels, 1)
        self.scale = nn.Conv3d(channels, channels, 1)
        self.eps = eps

    def forward(self, x, sample):
        mean = self.mean(x)
        std = F.softplus(self.scale(x).float()) + self.eps
        z = mean.float() + torch.randn_like(std) * std if sample else mean
        return z.to(x.dtype), {"mean": mean, "std": std}


class ClosedLoopVariationalFusion3d(nn.Module):
    """Eqs. 1-11. Eq. 11 regularizes feature posteriors of both branches.

    Attention logits are stochastic, but the manuscript does not specify a
    separate attention KL penalty, so no additional loss is introduced.
    """

    def __init__(self, channels=256, sigma0=1.0, gamma=2.0, prior_eps=1e-6):
        super().__init__()
        if sigma0 <= 0 or gamma < 0 or prior_eps <= 0:
            raise ValueError("sigma0/prior_eps must be positive; gamma must be nonnegative")
        self.vfm_posterior = GaussianEncoder3d(channels)
        self.cnn_posterior = GaussianEncoder3d(channels)
        self.vfm_attention = GaussianEncoder3d(channels)
        self.cnn_attention = GaussianEncoder3d(channels)
        self.transform = nn.Conv3d(channels, channels, 1)
        self.sigma0, self.gamma, self.prior_eps = sigma0, gamma, prior_eps

    def forward(self, vfm, cnn):
        sample = self.training
        zv, pv = self.vfm_posterior(vfm, sample)
        zc, pc = self.cnn_posterior(cnn, sample)
        av, pav = self.vfm_attention(zv, sample)
        ac, pac = self.cnn_attention(zc, sample)
        # Axis 1 is branch, axis 2 remains feature channel.
        weights = torch.stack((av, ac), dim=1).float().softmax(dim=1).to(zv.dtype)
        fused = self.transform(weights[:, 0] * zv + weights[:, 1] * zc)
        return fused, {
            "vfm": pv, "cnn": pc,
            "attention_vfm": pav, "attention_cnn": pac,
            "weights": weights,
        }

    def uncertainty_guided_kl(self, posteriors, uncertainty):
        mean = posteriors["vfm"]["mean"]
        # Detach only this feedback path. EUR rectification/evidence retain
        # gradients to the uncertainty estimator through their own objectives.
        feedback = _resize(uncertainty.detach().float(), mean.shape[-3:])
        prior_std = self.sigma0 * torch.exp(-self.gamma * feedback) + self.prior_eps
        terms = []
        for branch in ("vfm", "cnn"):
            mu = posteriors[branch]["mean"].float()
            std = posteriors[branch]["std"].float()
            log_ratio = std.log() - prior_std.log()
            kl = 0.5 * ((std / prior_std).square() + (mu / prior_std).square()
                        - 1.0 - 2.0 * log_ratio)
            # Mean over batch, channels, and spatial elements; SUM branches.
            terms.append(kl.mean())
        return sum(terms), prior_std


class EvidentialUncertaintyRefinement3d(nn.Module):
    """EUR, Eqs. 12-23 and Fig. 4 (four projections to eight channels)."""

    def __init__(self, classes, decoder_channels=(16, 32, 64, 128),
                 scale_channels=8, sab_max_tokens=512):
        super().__init__()
        self.classes = classes
        self.evidence_head = nn.Conv3d(decoder_channels[0], classes, 1)
        self.scale_projections = nn.ModuleList([
            nn.Conv3d(c, scale_channels, 1) for c in decoder_channels
        ])
        self.sab = SpatialAttentionBlock3d(4 * scale_channels, sab_max_tokens)
        self.context_projection = nn.Sequential(
            nn.Conv3d(4 * scale_channels, decoder_channels[0], 1, bias=False),
            nn.BatchNorm3d(decoder_channels[0]),
            nn.ReLU(inplace=True),
            nn.Conv3d(decoder_channels[0], classes, 1),
        )
        # One blending weight per voxel, broadcast over classes (Eq. 21).
        self.blend = nn.Conv3d(classes, 1, 1)

    def forward(self, decoder_features, initial_logits):
        evidence = F.softplus(self.evidence_head(decoder_features[0]).float())
        alpha = evidence + 1.0
        strength = alpha.sum(dim=1, keepdim=True)
        uncertainty = self.classes / strength
        projected = [conv(x) for conv, x in zip(self.scale_projections, decoder_features)]
        pyramid = [None] * 4
        pyramid[3] = projected[3]
        for i in (2, 1, 0):
            pyramid[i] = projected[i] + _resize(pyramid[i + 1], projected[i].shape[-3:])
        finest_size = decoder_features[0].shape[-3:]
        multiscale = torch.cat([_resize(x, finest_size) for x in pyramid], dim=1)
        enhanced = multiscale + self.sab(multiscale)
        context_logits = self.context_projection(enhanced)
        reliability_feature = ((initial_logits + context_logits).float()
                               * torch.exp(-uncertainty))
        blend_weight = torch.sigmoid(self.blend(F.relu(reliability_feature).to(initial_logits.dtype)))
        logits = blend_weight * initial_logits + (1.0 - blend_weight) * context_logits
        return {
            "logits": logits,
            "initial_logits": initial_logits,
            "context_logits": context_logits,
            "evidence": evidence,
            "alpha": alpha,
            "uncertainty": uncertainty,
            "evidential_probability": alpha / strength,
            "blend_weight": blend_weight,
        }


class CUCVF3D_sam(nn.Module):
    """Parallel SAM-Med3D/CNN with CU-CVF and EUR.

    Only the last two ViT blocks are trainable in SAM by default. The neck,
    patch/position embeddings, prompt encoder and SAM mask decoder stay frozen.
    Constructor injection of sam_model permits offline testing with a stub.
    In eval mode both levels of stochastic fusion use their posterior means.
    """

    def __init__(self, classes, channels, ckpt_path=None, sam_model=None,
                 sam_channels=384, sam_trainable_blocks=2, prior_sigma0=1.0,
                 prior_gamma=2.0, prior_eps=1e-6, age_max_tokens=1024,
                 sab_max_tokens=512):
        super().__init__()
        if classes < 2 or channels < 1:
            raise ValueError("classes must be >= 2 and channels must be positive")
        if sam_trainable_blocks < 0:
            raise ValueError("sam_trainable_blocks must be nonnegative")
        self.classes, self.channels = int(classes), int(channels)
        default_ckpt = (Path(__file__).resolve().parent / "SAM-Med3D-main"
                        / "ckpt" / "sam_med3d_turbo.pth")
        self.sam_ckpt = str(default_ckpt if ckpt_path is None else ckpt_path)
        if sam_model is None:
            if not Path(self.sam_ckpt).is_file():
                raise FileNotFoundError("SAM-Med3D checkpoint not found: " + self.sam_ckpt)
            try:
                import medim
            except ImportError as exc:
                raise ImportError("Install medim or pass an initialized sam_model.") from exc
            sam_model = medim.create_model(
                "SAM-Med3D", pretrained=True, checkpoint_path=self.sam_ckpt)
        if not hasattr(sam_model, "image_encoder"):
            raise ValueError("sam_model must expose image_encoder")
        self.sam_model = sam_model
        for parameter in self.sam_model.parameters():
            parameter.requires_grad_(False)
        blocks = getattr(self.sam_model.image_encoder, "blocks", None)
        if sam_trainable_blocks:
            if blocks is None or len(blocks) < sam_trainable_blocks:
                raise ValueError("SAM image_encoder has fewer blocks than sam_trainable_blocks")
            for block in list(blocks)[-sam_trainable_blocks:]:
                for parameter in block.parameters():
                    parameter.requires_grad_(True)

        self.enc_input_1 = ResEncoder3d(channels, 16)
        self.encoder1_1 = ResEncoder3d(16, 32)
        self.encoder2_1 = ResEncoder3d(32, 64)
        self.encoder3_1 = ResEncoder3d(64, 128)
        self.encoder4_1 = ResEncoder3d(128, 256)
        self.downsample = downsample()
        self.age = AttentionGuidedEnhancement3d(sam_channels, 256, age_max_tokens)
        self.fusion = ClosedLoopVariationalFusion3d(256, prior_sigma0, prior_gamma, prior_eps)
        self.deconv4, self.decoder4 = deconv(256, 128), Decoder3d(256, 128)
        self.deconv3, self.decoder3 = deconv(128, 64), Decoder3d(128, 64)
        self.deconv2, self.decoder2 = deconv(64, 32), Decoder3d(64, 32)
        self.deconv1, self.decoder1 = deconv(32, 16), Decoder3d(32, 16)
        self.final = nn.Conv3d(16, classes, 1)
        self.eur = EvidentialUncertaintyRefinement3d(classes, sab_max_tokens=sab_max_tokens)
        initialize_weights(*[module for name, module in self.named_children()
                             if name != "sam_model"])

    def _cnn_encode(self, x):
        enc0 = self.enc_input_1(x)
        enc1 = self.encoder1_1(self.downsample(enc0))
        enc2 = self.encoder2_1(self.downsample(enc1))
        enc3 = self.encoder3_1(self.downsample(enc2))
        return self.encoder4_1(self.downsample(enc3)), (enc0, enc1, enc2, enc3)

    def _sam_encode(self, x, target_size):
        encoder = self.sam_model.image_encoder
        # SAM-Med3D's fixed absolute position grid otherwise fails on the
        # manuscript's rectangular 128 x 160 x 160 crops. Interpolate that grid
        # without changing the input image, pretrained weights, or global APIs.
        position = getattr(encoder, "pos_embed", None)
        patch_embed = getattr(encoder, "patch_embed", None)
        if position is not None and patch_embed is not None and hasattr(encoder, "neck"):
            tokens = patch_embed(x)  # SAM uses [B,D,H,W,C] tokens.
            if tokens.ndim != 5 or tokens.shape[-1] != position.shape[-1]:
                raise ValueError("Expected channels-last SAM patch embeddings")
            if tuple(position.shape[1:4]) != tuple(tokens.shape[1:4]):
                position = _resize(position.permute(0, 4, 1, 2, 3), tokens.shape[1:4])
                position = position.permute(0, 2, 3, 4, 1)
            tokens = tokens + position
            for block in encoder.blocks:
                tokens = _sam_block_3d(block, tokens)
            embedding = encoder.neck(tokens.permute(0, 4, 1, 2, 3))
        else:
            embedding = encoder(x)
        if isinstance(embedding, (tuple, list, dict)):
            values = embedding.values() if isinstance(embedding, dict) else embedding
            tensors = [v for v in values if torch.is_tensor(v) and v.ndim == 5]
            if not tensors:
                raise ValueError("SAM encoder did not return a 5D feature tensor")
            embedding = tensors[-1]
        if not torch.is_tensor(embedding) or embedding.ndim != 5:
            raise ValueError("SAM embedding must be [B,C,D,H,W]")
        if embedding.shape[1] != self.age.query.in_channels:
            raise ValueError("SAM output channel mismatch; set sam_channels to the encoder output width")
        return _resize(embedding, target_size)

    def _decode(self, fused, skips):
        enc0, enc1, enc2, enc3 = skips
        dec4 = self.decoder4(torch.cat((enc3, _resize(self.deconv4(fused), enc3.shape[-3:])), 1))
        dec3 = self.decoder3(torch.cat((enc2, _resize(self.deconv3(dec4), enc2.shape[-3:])), 1))
        dec2 = self.decoder2(torch.cat((enc1, _resize(self.deconv2(dec3), enc1.shape[-3:])), 1))
        dec1 = self.decoder1(torch.cat((enc0, _resize(self.deconv1(dec2), enc0.shape[-3:])), 1))
        return dec1, dec2, dec3, dec4

    def forward(self, x, return_aux=False):
        if x.ndim != 5 or x.shape[1] != self.channels or min(x.shape[-3:]) < 16:
            raise ValueError("Input must be [B,channels,D,H,W] with each spatial size >= 16")
        cnn, skips = self._cnn_encode(x)
        vfm = self.age(self._sam_encode(x, cnn.shape[-3:]))
        fused, posteriors = self.fusion(vfm, cnn)
        decoded = self._decode(fused, skips)
        output = self.eur(decoded, self.final(decoded[0]))
        if not return_aux:
            return output["logits"]
        output["ug_kl_loss"], output["prior_std"] = self.fusion.uncertainty_guided_kl(
            posteriors, output["uncertainty"])
        output["fusion_weights"] = posteriors["weights"]
        output["posteriors"] = posteriors
        return output

    def forward_factual_logits(self, x):
        """Compatibility alias for callers of the former logits-only method."""
        return self(x)


class CUCVFCompositeLoss(nn.Module):
    """Eqs. 24-31: Dice + weighted CE + evidence + lambda_ug * UG-KL.

    lambda_ug=1e-4 and annealing_epochs=50 are configurable implementation
    defaults; the supplied manuscript does not report their numerical values.
    Supply class_weights for dataset-specific weighted CE (Eq. 27).
    Epoch is zero based, and Eq. 31 is applied as min(1, epoch / Ta).
    """

    def __init__(self, class_weights=None, lambda_ug=1e-4, annealing_epochs=50,
                 eps=1e-6):
        super().__init__()
        if lambda_ug < 0 or annealing_epochs <= 0 or eps <= 0:
            raise ValueError("lambda_ug must be nonnegative; annealing_epochs/eps positive")
        weights = None if class_weights is None else torch.as_tensor(class_weights, dtype=torch.float32)
        if weights is not None and (weights.ndim != 1 or not torch.isfinite(weights).all()
                                    or (weights < 0).any() or weights.sum() <= 0):
            raise ValueError("class_weights must be a finite, nonnegative 1D vector with positive sum")
        self.register_buffer("class_weights", weights)
        self.lambda_ug = float(lambda_ug)
        self.annealing_epochs = float(annealing_epochs)
        self.eps = float(eps)

    @staticmethod
    def dirichlet_kl_to_uniform(alpha):
        """KL(Dir(alpha) || Dir(1)), summed over classes per voxel."""
        classes = alpha.shape[1]
        strength = alpha.sum(dim=1, keepdim=True)
        log_normalizer = (torch.lgamma(strength) - torch.lgamma(alpha).sum(1, keepdim=True)
                          - torch.lgamma(alpha.new_tensor(float(classes))))
        expectation = ((alpha - 1.0) * (torch.digamma(alpha) - torch.digamma(strength))).sum(1, keepdim=True)
        return log_normalizer + expectation

    def forward(self, output, label, epoch=0):
        logits, alpha = output["logits"].float(), output["alpha"].float()
        if label.ndim == logits.ndim and label.shape[1] == 1:
            label = label[:, 0]
        if tuple(label.shape) != (logits.shape[0], *logits.shape[2:]):
            raise ValueError("label must be [B,D,H,W] or [B,1,D,H,W] class indices")
        if epoch < 0:
            raise ValueError("epoch must be nonnegative")
        label = label.long()
        classes = logits.shape[1]
        if self.class_weights is not None and self.class_weights.numel() != classes:
            raise ValueError("class_weights length must match output classes")
        target = F.one_hot(label, num_classes=classes).movedim(-1, 1).float()
        probability = logits.softmax(dim=1)
        axes = (0, 2, 3, 4)
        intersection = (target * probability).sum(dim=axes)
        denominator = (target + probability).sum(dim=axes)
        dice = 1.0 - ((2.0 * intersection + self.eps) / (denominator + self.eps)).mean()
        # Eq. 27 divides by the number of voxels, not by summed class weights.
        weights = self.class_weights
        if weights is not None:
            weights = weights.to(device=logits.device, dtype=logits.dtype)
        wce = F.cross_entropy(logits, label, weight=weights, reduction="none").mean()
        segmentation = dice + wce
        strength = alpha.sum(dim=1, keepdim=True)
        expected_probability = alpha / strength
        error = (target - expected_probability).square().sum(dim=1)
        variance = (expected_probability * (1.0 - expected_probability)
                    / (strength + 1.0)).sum(dim=1)
        adjusted_alpha = target + (1.0 - target) * alpha
        evidence_kl = self.dirichlet_kl_to_uniform(adjusted_alpha).mean()
        annealing = min(1.0, float(epoch) / self.annealing_epochs)
        evidence = (error + variance).mean() + annealing * evidence_kl
        ug_kl = output["ug_kl_loss"].float()
        total = segmentation + evidence + self.lambda_ug * ug_kl
        return {"total": total, "segmentation": segmentation, "dice": dice,
                "weighted_ce": wce, "evidence": evidence,
                "evidence_kl": evidence_kl, "ug_kl": ug_kl}
