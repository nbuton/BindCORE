"""
Torch implementations of random-coil reference pairwise features.

These matrices depend only on sequence length, device, and dtype, so the
expensive pieces are cached and cloned before use in a forward pass.
"""

from __future__ import annotations

import functools
import math
from typing import Sequence, Union

import torch

Device = Union[torch.device, str]


@functools.lru_cache(maxsize=128)
def _cached_distance_fluctuation(
    size: int, device_str: str, dtype: torch.dtype
) -> torch.Tensor:
    device = torch.device(device_str)
    with torch.no_grad():
        idx = torch.arange(size, device=device)
        diff = (idx[:, None] - idx[None, :]).abs()
        mask = (diff >= 1) & (diff <= 2)

        gamma = torch.zeros((size, size), device=device, dtype=dtype)
        gamma = gamma.masked_fill(mask, -1.0)
        gamma = gamma + torch.diag(-gamma.sum(dim=1))

        covariance = torch.linalg.pinv(gamma)
        diag = torch.diagonal(covariance)
        df_matrix = diag.unsqueeze(1) + diag.unsqueeze(0) - 2.0 * covariance
    return df_matrix


def generate_random_coil_distance_fluctuation(
    size: int,
    device: Device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """GNM-based random-coil distance fluctuation matrix."""
    device_str = str(torch.device(device))
    return _cached_distance_fluctuation(size, device_str, dtype).clone()


@functools.lru_cache(maxsize=128)
def _cached_contact_frequency(
    size: int, cutoff: float, b: float, device_str: str, dtype: torch.dtype
) -> torch.Tensor:
    device = torch.device(device_str)
    with torch.no_grad():
        idx = torch.arange(size, device=device, dtype=dtype)
        i, j = torch.meshgrid(idx, idx, indexing="ij")
        k = (i - j).abs()
        k_safe = torch.where(k == 0, torch.full_like(k, 1e-9), k)

        sigma = torch.sqrt((b**2) * k_safe / 3.0)
        x = cutoff / (math.sqrt(2.0) * sigma)
        frequency = torch.erf(x) - (2.0 / math.sqrt(math.pi)) * x * torch.exp(
            -(x**2)
        )
        frequency = frequency.clone()
        frequency.fill_diagonal_(1.0)
    return frequency


def generate_random_coil_contact_frequency(
    size: int,
    cutoff: float = 8.0,
    b: float = 3.8,
    device: Device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Gaussian-chain contact probability matrix for a random coil."""
    device_str = str(torch.device(device))
    return _cached_contact_frequency(
        size, float(cutoff), float(b), device_str, dtype
    ).clone()


@functools.lru_cache(maxsize=128)
def _cached_rouse_correlation(
    size: int, device_str: str, dtype: torch.dtype
) -> torch.Tensor:
    device = torch.device(device_str)
    with torch.no_grad():
        p = torch.arange(1, size, device=device, dtype=dtype)
        positions = torch.arange(size, device=device, dtype=dtype) + 0.5

        angles = p[:, None] * math.pi * positions[None, :] / size
        modes = torch.cos(angles)
        weights = 1.0 / (p**2)

        weighted_modes = modes * weights[:, None]
        corr = weighted_modes.t() @ modes

        diag = torch.diagonal(corr)
        denom = torch.sqrt(torch.outer(diag, diag))
        corr_norm = corr / denom
    return corr_norm


def rouse_correlation_matrix(
    size: int,
    device: Device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Normalized Rouse-mode correlation matrix for a random coil."""
    if size <= 1:
        return torch.ones((size, size), device=device, dtype=dtype)

    device_str = str(torch.device(device))
    return _cached_rouse_correlation(size, device_str, dtype).clone()


def random_coil_pairwise_baseline(
    size: int,
    device: Device = "cpu",
    dtype: torch.dtype = torch.float32,
    cutoff: float = 8.0,
    b: float = 3.8,
) -> torch.Tensor:
    """
    Return baseline channels in BindCORE pairwise order:
    dccm, contact_map/contact_frequency, distance_fluctuations.
    """
    return torch.stack(
        (
            rouse_correlation_matrix(size, device=device, dtype=dtype),
            generate_random_coil_contact_frequency(
                size, cutoff=cutoff, b=b, device=device, dtype=dtype
            ),
            generate_random_coil_distance_fluctuation(
                size, device=device, dtype=dtype
            ),
        ),
        dim=0,
    )


def subtract_random_coil_pairwise_baseline(
    x_pairwise: torch.Tensor,
    pairwise_features: Sequence[str],
    mask: torch.Tensor | None = None,
    cutoff: float = 8.0,
    b: float = 3.8,
) -> torch.Tensor:
    """
    Convert raw pairwise features to residual features by subtracting the
    matching random-coil baseline from channels named in pairwise_features.

    Known baselines:
    - dccm -> Rouse correlation matrix
    - contact_map/contact_frequency -> Gaussian-chain contact frequency
    - distance_fluctuations -> GNM distance fluctuation matrix

    Unknown feature channels are passed through unchanged.
    """
    baseline_names = {
        "dccm",
        "contact_map",
        "contact_frequency",
        "distance_fluctuations",
        "distance_fluctuation",
    }
    feature_names = [str(name) for name in pairwise_features]
    baseline_channels = [
        channel
        for channel, name in enumerate(feature_names[: x_pairwise.size(1)])
        if name in baseline_names
    ]
    if not baseline_channels:
        return x_pairwise

    B, _, L, _ = x_pairwise.shape
    out = x_pairwise.clone()
    device = x_pairwise.device
    dtype = x_pairwise.dtype

    if mask is None:
        lengths = torch.full((B,), L, device=device, dtype=torch.long)
        pairwise_mask = None
    else:
        lengths = mask.to(device=device).sum(dim=1).to(torch.long)
        m = mask.to(device=device, dtype=dtype)
        pairwise_mask = m.unsqueeze(1).unsqueeze(-1) * m.unsqueeze(1).unsqueeze(2)

    for length in torch.unique(lengths).tolist():
        length = int(length)
        if length <= 0:
            continue

        baseline = random_coil_pairwise_baseline(
            length, device=device, dtype=dtype, cutoff=cutoff, b=b
        )
        batch_idx = lengths == length
        for channel in baseline_channels:
            name = feature_names[channel]
            if name == "dccm":
                baseline_channel = 0
            elif name in {"contact_map", "contact_frequency"}:
                baseline_channel = 1
            elif name in {"distance_fluctuations", "distance_fluctuation"}:
                baseline_channel = 2
            else:
                continue

            out[batch_idx, channel, :length, :length] = (
                out[batch_idx, channel, :length, :length]
                - baseline[baseline_channel].unsqueeze(0)
            )

    if pairwise_mask is not None:
        out = out * pairwise_mask

    return out
