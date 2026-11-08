# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Pure INT8 scaled dot-product attention for NVIDIA tensor-core GPUs."""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass

import torch
from torch.nn import functional

from .backends import cuda as _cuda_backend
from .backends.eager.quantization import DTYPE_TO_CODE

if getattr(torch.version, "hip", None):
    from .backends import hip as _hip_backend
else:
    _hip_backend = None

CTA_K = 64
LARGE_CTA_K = 128
_SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
_NATIVE_MINIMUM_CAPABILITY = (7, 5)

logger = logging.getLogger("comfy_kitchen.sage_attention")

# The fused INT8 path has to quantize Q, K and V before the attention kernel can
# start, and that preamble does not shrink with the sequence. Below some length
# per segment the preamble costs more than the faster kernel saves: on gfx9 at
# q_len == kv_len, 4096 tokens ran 1.81x slower than SDPA and 35880 tokens ran
# 1.28x faster. The default is the first power of two above the measured loss and
# well below the measured win, so it discards the losses without being trusted
# about the wins. A sweep has to place the real crossover, and the variable can
# then raise it without a rebuild.
_INT8_MIN_QK_ENV = "COMFY_KITCHEN_INT8_MIN_QK"
_INT8_MIN_QK_DEFAULT = 8192
_int8_min_qk_cache: tuple[str | None, int] | None = None
_int8_gate_reported = False


@dataclass(frozen=True, slots=True)
class PrequantizedInt8Attention:
    """Packed Q/K/V and immutable launch metadata for split INT8 attention.

    Instances own only the quantized tensors, their scales, and an optional
    attention mask in its active layout: expanded 4D values or a prepared 3D
    buffer containing key values and tile metadata, or a HIP 5D dense tile
    buffer. Prepared masks replace the original mask. Instances never retain
    the floating-point Q, K, or V inputs.
    Create instances with :func:`prequantize_int8_attention` rather than
    constructing them directly.
    """

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    q_scale: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    original_head_dim: int
    input_dtype: torch.dtype
    attention_scale: float
    cta_k: int
    attn_mask: torch.Tensor | None


def _pad_to_cta_k(length: int, cta_k: int = CTA_K) -> int:
    return ((length + cta_k - 1) // cta_k) * cta_k


def _select_cta_k(
    kernel_head_dim: int,
    kv_length: int,
    *,
    has_mask: bool,
) -> int:
    if not has_mask and kernel_head_dim >= 128 and kv_length > 1024:
        return LARGE_CTA_K
    return CTA_K


def _int8_min_qk() -> int:
    """Minimum segment length for the INT8 path, from the environment.

    Read per call rather than at import so that one process can compare the
    gated and ungated paths by flipping the variable between runs; the parse is
    cached against the raw string, which keeps the steady-state cost to a dict
    lookup. A bad value falls back to the default instead of raising: a tuning
    knob is not worth failing an inference over, and 0 -- the documented way to
    restore the old unconditional behaviour -- parses cleanly anyway.
    """
    global _int8_min_qk_cache
    raw = os.environ.get(_INT8_MIN_QK_ENV)
    if _int8_min_qk_cache is None or _int8_min_qk_cache[0] != raw:
        value = _INT8_MIN_QK_DEFAULT
        if raw is not None:
            try:
                value = max(0, int(raw))
            except ValueError:
                logger.warning(
                    "ignoring non-integer %s=%r, using %d",
                    _INT8_MIN_QK_ENV,
                    raw,
                    _INT8_MIN_QK_DEFAULT,
                )
        _int8_min_qk_cache = (raw, value)
    return _int8_min_qk_cache[1]


def _int8_sequence_too_short(q: torch.Tensor, k: torch.Tensor) -> bool:
    """Whether this Q/K pair is shorter than the INT8 path crossover.

    The comparison is q_len * kv_len < threshold ** 2, i.e. the geometric mean
    of the two lengths against a length. Every shape that was measured was
    square, where that reduces to 'sequence length below the threshold'; the mean
    is what keeps the asymmetric case honest. The text segment of a block-causal
    image model is a few dozen queries over a few thousand keys: the product
    alone would put it far above any threshold and keep quantizing thousands of
    V rows to answer forty queries, while min() would gate on the wrong side of
    long-query/short-key attention.
    """
    threshold = _int8_min_qk()
    if threshold <= 0:
        return False
    q_length = q.shape[2]
    kv_length = k.shape[2]
    if q_length * kv_length >= threshold * threshold:
        return False
    global _int8_gate_reported
    if not _int8_gate_reported:
        # Once per process: this fires on every layer of every step, and the
        # decision only changes if the variable does.
        _int8_gate_reported = True
        logger.info(
            "[comfy_kitchen] int8 disabled: q*kv=%d (%d x %d, ~%.0f tokens) < "
            "threshold=%d^2; falling back to SDPA",
            q_length * kv_length,
            q_length,
            kv_length,
            math.sqrt(q_length * kv_length),
            threshold,
        )
    return True


def _int8_gated(q: torch.Tensor, k: torch.Tensor) -> bool:
    """Whether to keep this call off the INT8 path entirely.

    Only the shapes the INT8 path accepts are gated, so anything malformed still
    reaches the validation in _int8_attention_cuda and fails with the error that
    names the real problem. The gate is deliberately HIP-only: the threshold
    comes from gfx9 measurements, and silently moving CUDA's small shapes onto
    SDPA would stop its own kernels being exercised there at all.
    """
    return (
        _hip_backend is not None
        and q.ndim == 4
        and k.ndim == 4
        and q.dtype in _SUPPORTED_DTYPES
        and q.shape[1] % k.shape[1] == 0
        and _int8_sequence_too_short(q, k)
    )


def _sdpa_fallback(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float | None,
    attn_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Run a gated call as plain SDPA.

    This is the call the flash baseline makes, with the mask untouched and the
    same scale, so a gated call is numerically identical to a run that never had
    an INT8 path: the gate can change the timing, never the output.
    """
    head_dim = q.shape[-1]
    return functional.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        scale=head_dim**-0.5 if scale is None else float(scale),
        # The INT8 path takes grouped-query heads as they are; SDPA only
        # broadcasts them when asked (torch >= 2.5).
        enable_gqa=q.shape[1] != k.shape[1],
    )


def _prepare_attn_mask(
    attn_mask: torch.Tensor | None,
    attention_scale: float,
    *,
    fuse_short_key_mask: bool = False,
    fuse_short_dense_mask: bool = False,
) -> torch.Tensor | None:
    """Prepare masks for the native tile layout or let a short fused call do it."""
    if _hip_backend is not None and not _hip_backend._has_rdna_wmma(
        _hip_backend._visible_gfx_arches()
    ):
        # Hygon DCU (gfx9): the ported int8 attention kernel reads the original
        # mask through mask_keep and has no prepared key / dense body, so keep the
        # mask in its native form.
        return attn_mask
    if (
        fuse_short_dense_mask
        and _hip_backend is not None
        and _hip_backend._sage_can_fuse_dense_mask(attn_mask)
    ):
        return attn_mask
    if (
        fuse_short_key_mask
        and _hip_backend is not None
        and attn_mask is not None
        and attn_mask.shape[2] <= 256
        and 64 < attn_mask.shape[3] <= 2048
        and (attn_mask.stride(2) == 0 or attn_mask.shape[2] == 1)
    ):
        return attn_mask
    # HIP retains native 16-bit dense biases in their original domain; FP32
    # biases use the base-two softmax domain. Boolean masks pack one bit per pair.
    # Its floating-score path also supports nonpositive scales.
    # CUDA's prepared layout is limited to broadcast masks and positive scales.
    # Paired HIP measurements: benchmarks/hip_mask_overhead.py.
    if (
        attn_mask is None
        or attn_mask.shape[-1] <= (64 if _hip_backend is not None else 1024)
        or (_hip_backend is None and attn_mask.stride(2) != 0)
        or (_hip_backend is None and attention_scale <= 0)
    ):
        return attn_mask
    mask_batch = 1 if attn_mask.stride(0) == 0 else attn_mask.shape[0]
    mask_heads = 1 if attn_mask.stride(1) == 0 else attn_mask.shape[1]
    if _hip_backend is not None and attn_mask.stride(2) != 0 and attn_mask.shape[2] > 1:
        compact_mask = attn_mask[:mask_batch, :mask_heads]
        packed = _hip_backend._sage_dense_mask_buffer(attn_mask)
        _hip_backend._C.sage_prepare_dense_mask(
            _hip_backend._dl(compact_mask),
            _hip_backend._dl(packed),
            torch.cuda.current_stream(attn_mask.device).cuda_stream,
        )
        return packed
    compact_mask = attn_mask[:mask_batch, :mask_heads, :1, :]
    # HIP retains its 64-key schedule and has its own prepared-mask layout.
    tile_k = _hip_backend._SAGE_CTA_K if _hip_backend is not None else LARGE_CTA_K
    backend = _hip_backend if _hip_backend is not None else _cuda_backend
    wrap = backend._dl if _hip_backend is not None else backend._wrap_for_dlpack
    tiles = (attn_mask.shape[-1] + tile_k - 1) // tile_k
    # Align each packed row for vector loads in the attention kernel.
    packed_width = ((tiles * (tile_k + 1) + 3) // 4) * 4
    packed = torch.empty(
        (mask_batch, mask_heads, packed_width),
        dtype=torch.float32,
        device=attn_mask.device,
    )
    backend._C.sage_prepare_key_mask(
        wrap(compact_mask),
        wrap(packed),
        torch.cuda.current_stream(attn_mask.device).cuda_stream,
    )
    return packed


def is_available(device: torch.device | None = None) -> bool:
    """Return whether the compiled INT8 attention kernel supports this GPU."""
    if device is not None and device.type != "cuda":
        return False
    if not torch.cuda.is_available():
        return False
    if _hip_backend is not None:
        # torch.cuda is the ROCm API here, and get_device_capability reports
        # something SM-shaped for a gfx part, so the compute capability test
        # below would wave AMD hardware through to a CUDA extension that never
        # loaded. Ask the HIP backend instead.
        return _hip_backend.int8_attention_is_available()
    capability = torch.cuda.get_device_capability(device)
    return capability >= _NATIVE_MINIMUM_CAPABILITY and _cuda_backend._EXT_AVAILABLE


def _validate_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor | None,
) -> torch.Tensor | None:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must have shape [batch, heads, sequence, head_dim]")
    if q.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"q, k, and v must be float32, float16, or bfloat16, got {q.dtype}")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise TypeError(
            f"q, k, and v must have the same dtype, got {q.dtype}, {k.dtype}, and {v.dtype}"
        )
    if not q.is_cuda or q.device != k.device or q.device != v.device:
        raise ValueError("q, k, and v must be on the same CUDA device")
    if not is_available(q.device):
        raise RuntimeError(
            "INT8 attention requires the comfy-kitchen CUDA extension on SM75 or newer, "
            "or the HIP extension on an AMD device with matrix cores (RDNA3 or newer)"
        )

    batch, q_heads, q_length, head_dim = q.shape
    k_batch, kv_heads, kv_length, k_head_dim = k.shape
    if v.shape != (batch, kv_heads, kv_length, head_dim):
        raise ValueError(
            f"v must have shape [q.batch, k.heads, k.sequence, q.head_dim], got {tuple(v.shape)}"
        )
    if k_batch != batch or k_head_dim != head_dim:
        raise ValueError(
            f"q and k batch/head dimensions must match, got {tuple(q.shape)} and {tuple(k.shape)}"
        )
    if batch == 0 or q_heads == 0 or kv_heads == 0 or q_length == 0 or kv_length == 0:
        raise ValueError("batch, head counts, and sequence lengths must be positive")
    if q_heads % kv_heads != 0:
        raise ValueError(
            f"q head count ({q_heads}) must be divisible by k/v head count ({kv_heads})"
        )
    if head_dim <= 0 or head_dim > 256:
        raise ValueError(f"head_dim must be in [1, 256], got {head_dim}")
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise ValueError("the last dimension of q, k, and v must be contiguous")
    if attn_mask is None:
        return None
    if attn_mask.device != q.device:
        raise ValueError("attn_mask must be on the same CUDA device as q, k, and v")
    if attn_mask.dtype not in (torch.bool, torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("attn_mask must be bool, float16, bfloat16, or float32")
    try:
        return torch.broadcast_to(attn_mask, (batch, q_heads, q_length, kv_length))
    except RuntimeError as error:
        raise ValueError(
            "attn_mask must be broadcastable to "
            f"[{batch}, {q_heads}, {q_length}, {kv_length}], got {tuple(attn_mask.shape)}"
        ) from error


def _int8_attention_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float | None = None,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    # Ahead of validation on purpose: a gated call does no padding, no mask
    # preparation and no quantization, and SDPA validates what it is given.
    if _int8_gated(q, k):
        return _sdpa_fallback(q, k, v, scale=scale, attn_mask=attn_mask)
    attn_mask = _validate_inputs(q, k, v, attn_mask)

    original_head_dim = q.shape[-1]
    if original_head_dim <= 64:
        kernel_head_dim = 64
    elif original_head_dim <= 128:
        kernel_head_dim = 128
    else:
        kernel_head_dim = 256
    if kernel_head_dim != original_head_dim:
        padding = (0, kernel_head_dim - original_head_dim)
        q = functional.pad(q, padding)
        k = functional.pad(k, padding)
        v = functional.pad(v, padding)

    attention_scale = original_head_dim**-0.5 if scale is None else float(scale)
    if not math.isfinite(attention_scale):
        raise ValueError(f"scale must be finite, got {attention_scale}")

    attn_mask = _prepare_attn_mask(
        attn_mask,
        attention_scale,
        fuse_short_key_mask=kernel_head_dim <= 128,
        fuse_short_dense_mask=kernel_head_dim <= 128,
    )
    if _hip_backend is not None:
        output = _hip_backend.sage_int8_sdpa(
            q,
            k,
            v,
            attention_scale=attention_scale,
            attn_mask=attn_mask,
        )
        output = output[..., :original_head_dim]
        return output.float() if q.dtype == torch.float32 else output

    batch, q_heads, q_length, _ = q.shape
    _, kv_heads, kv_length, _ = k.shape
    cta_k = _select_cta_k(
        kernel_head_dim,
        kv_length,
        has_mask=attn_mask is not None and attn_mask.ndim == 4,
    )
    padded_k_length = _pad_to_cta_k(kv_length, cta_k)
    q_int8 = torch.empty(q.shape, dtype=torch.int8, device=q.device)
    k_int8 = torch.empty(k.shape, dtype=torch.int8, device=k.device)
    q_scales_per_block = 64 if kernel_head_dim == 256 else 32
    q_scale = torch.empty(
        batch,
        q_heads,
        ((q_length + 127) // 128) * q_scales_per_block,
        dtype=torch.float32,
        device=q.device,
    )
    k_scale = torch.empty(
        batch,
        kv_heads,
        ((kv_length + cta_k - 1) // cta_k) * 4,
        dtype=torch.float32,
        device=q.device,
    )
    v_int8 = torch.empty(
        batch * kv_heads * kernel_head_dim,
        padded_k_length,
        dtype=torch.int8,
        device=q.device,
    )
    v_scale = torch.empty(batch * kv_heads * kernel_head_dim, dtype=torch.float32, device=q.device)

    output_dtype = torch.bfloat16 if q.dtype == torch.float32 else q.dtype
    output = torch.empty(
        batch, q_heads, q_length, kernel_head_dim, dtype=output_dtype, device=q.device
    )

    anchor_indices = torch.empty(
        batch, kv_heads, dtype=torch.int32, device=q.device
    )
    anchor_indices_ptr = anchor_indices.data_ptr()

    stream_ptr = torch.cuda.current_stream(q.device).cuda_stream
    _cuda_backend._C.sage_sdpa(
        _cuda_backend._wrap_for_dlpack(q),
        _cuda_backend._wrap_for_dlpack(k),
        _cuda_backend._wrap_for_dlpack(v),
        _cuda_backend._wrap_for_dlpack(output),
        _cuda_backend._wrap_for_dlpack(q_int8),
        _cuda_backend._wrap_for_dlpack(q_scale),
        _cuda_backend._wrap_for_dlpack(k_int8),
        _cuda_backend._wrap_for_dlpack(k_scale),
        _cuda_backend._wrap_for_dlpack(v_int8),
        _cuda_backend._wrap_for_dlpack(v_scale),
        attention_scale,
        DTYPE_TO_CODE[q.dtype],
        DTYPE_TO_CODE[output_dtype],
        stream_ptr,
        anchor_indices_ptr,
        _cuda_backend._wrap_for_dlpack(attn_mask) if attn_mask is not None else None,
        cta_k,
    )

    output = output[..., :original_head_dim]
    return output.float() if q.dtype == torch.float32 else output


def prequantize_int8_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float | None = None,
    attn_mask: torch.Tensor | None = None,
) -> PrequantizedInt8Attention:
    """Quantize Q, K, and V without allocating the attention output.

    The returned object does not retain the floating-point inputs, so model
    code can delete those tensors before calling
    :func:`int8_attention_from_prequantized`. Optimized key masks are prepared
    as part of this snapshot; recreate it after changing the mask.
    Quantization and consumption use
    the current CUDA stream and preserve normal PyTorch stream ordering; no
    host synchronization is introduced.

    This is an inference and peak-memory API. The regular :func:`int8_attention`
    remains the lower-overhead single-call path when early Q/K/V release is not
    needed.

    Calling this is an explicit request for the INT8 path, so it is not subject
    to the COMFY_KITCHEN_INT8_MIN_QK short-sequence gate: there is nothing left
    to fall back to once the floating-point inputs are gone.
    """
    attn_mask = _validate_inputs(q, k, v, attn_mask)

    original_head_dim = q.shape[-1]
    input_dtype = q.dtype
    if original_head_dim <= 64:
        kernel_head_dim = 64
    elif original_head_dim <= 128:
        kernel_head_dim = 128
    else:
        kernel_head_dim = 256
    if kernel_head_dim != original_head_dim:
        padding = (0, kernel_head_dim - original_head_dim)
        q = functional.pad(q, padding)
        k = functional.pad(k, padding)
        v = functional.pad(v, padding)

    attention_scale = original_head_dim**-0.5 if scale is None else float(scale)
    if not math.isfinite(attention_scale):
        raise ValueError(f"scale must be finite, got {attention_scale}")

    attn_mask = _prepare_attn_mask(attn_mask, attention_scale)
    if _hip_backend is not None:
        # The packed V row width follows this cta_k, so the value that packed the
        # buffers is the one that has to come back to attend over them. Taking the
        # CUDA-side constant here would only agree with the HIP choice by accident.
        hip_cta_k = _hip_backend._sage_cta_k(
            kernel_head_dim, k.shape[2], attn_mask is not None
        )
        packed = _hip_backend.sage_int8_quantize(q, k, v, cta_k=hip_cta_k)
        return PrequantizedInt8Attention(
            q=packed["q_int8"],
            k=packed["k_int8"],
            v=packed["v_int8"],
            q_scale=packed["q_scale"],
            k_scale=packed["k_scale"],
            v_scale=packed["v_scale"],
            original_head_dim=original_head_dim,
            input_dtype=input_dtype,
            attention_scale=attention_scale,
            cta_k=hip_cta_k,
            attn_mask=attn_mask,
        )

    batch, q_heads, q_length, _ = q.shape
    _, kv_heads, kv_length, _ = k.shape
    cta_k = _select_cta_k(
        kernel_head_dim,
        kv_length,
        has_mask=attn_mask is not None and attn_mask.ndim == 4,
    )
    padded_k_length = _pad_to_cta_k(kv_length, cta_k)
    q_int8 = torch.empty(q.shape, dtype=torch.int8, device=q.device)
    k_int8 = torch.empty(k.shape, dtype=torch.int8, device=k.device)
    q_scales_per_block = 64 if kernel_head_dim == 256 else 32
    q_scale = torch.empty(
        batch,
        q_heads,
        ((q_length + 127) // 128) * q_scales_per_block,
        dtype=torch.float32,
        device=q.device,
    )
    k_scale = torch.empty(
        batch,
        kv_heads,
        ((kv_length + cta_k - 1) // cta_k) * 4,
        dtype=torch.float32,
        device=q.device,
    )
    v_int8 = torch.empty(
        batch * kv_heads * kernel_head_dim,
        padded_k_length,
        dtype=torch.int8,
        device=q.device,
    )
    v_scale = torch.empty(
        batch * kv_heads * kernel_head_dim,
        dtype=torch.float32,
        device=q.device,
    )

    anchor_indices = torch.empty(
        batch, kv_heads, dtype=torch.int32, device=q.device
    )
    anchor_indices_ptr = anchor_indices.data_ptr()

    stream_ptr = torch.cuda.current_stream(q.device).cuda_stream
    _cuda_backend._C.sage_sdpa_quantize(
        _cuda_backend._wrap_for_dlpack(q),
        _cuda_backend._wrap_for_dlpack(k),
        _cuda_backend._wrap_for_dlpack(v),
        _cuda_backend._wrap_for_dlpack(q_int8),
        _cuda_backend._wrap_for_dlpack(q_scale),
        _cuda_backend._wrap_for_dlpack(k_int8),
        _cuda_backend._wrap_for_dlpack(k_scale),
        _cuda_backend._wrap_for_dlpack(v_int8),
        _cuda_backend._wrap_for_dlpack(v_scale),
        cta_k,
        DTYPE_TO_CODE[input_dtype],
        stream_ptr,
        anchor_indices_ptr,
    )

    return PrequantizedInt8Attention(
        q=q_int8,
        k=k_int8,
        v=v_int8,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        original_head_dim=original_head_dim,
        input_dtype=input_dtype,
        attention_scale=attention_scale,
        cta_k=cta_k,
        attn_mask=attn_mask,
    )


def int8_attention_from_prequantized(
    quantized: PrequantizedInt8Attention,
) -> torch.Tensor:
    """Run INT8 attention after the floating-point Q/K/V inputs are released."""
    if not isinstance(quantized, PrequantizedInt8Attention):
        raise TypeError(
            "quantized must be returned by prequantize_int8_attention, got "
            f"{type(quantized).__name__}"
        )

    packed_tensors = (
        quantized.q,
        quantized.k,
        quantized.v,
        quantized.q_scale,
        quantized.k_scale,
        quantized.v_scale,
    )
    if not quantized.q.is_cuda:
        raise ValueError("prequantized INT8 attention tensors must be on a CUDA device")
    if any(tensor.device != quantized.q.device for tensor in packed_tensors[1:]):
        raise ValueError("prequantized INT8 attention tensors must be on the same CUDA device")
    if quantized.attn_mask is not None and quantized.attn_mask.device != quantized.q.device:
        raise ValueError("attn_mask must be on the same CUDA device as the packed tensors")
    if not is_available(quantized.q.device):
        raise RuntimeError(
            "INT8 attention requires the comfy-kitchen CUDA extension on SM75 or newer, "
            "or the HIP extension on an AMD device with matrix cores (RDNA3 or newer)"
        )

    batch, q_heads, q_length, kernel_head_dim = quantized.q.shape
    output_dtype = (
        torch.bfloat16 if quantized.input_dtype == torch.float32 else quantized.input_dtype
    )

    if _hip_backend is not None:
        output = _hip_backend.sage_int8_attend(
            quantized.q,
            quantized.k,
            quantized.v,
            quantized.q_scale,
            quantized.k_scale,
            quantized.v_scale,
            attention_scale=quantized.attention_scale,
            attn_mask=quantized.attn_mask,
            output_dtype=output_dtype,
            cta_k=quantized.cta_k,
        )
        output = output[..., : quantized.original_head_dim]
        return output.float() if quantized.input_dtype == torch.float32 else output

    output = torch.empty(
        batch,
        q_heads,
        q_length,
        kernel_head_dim,
        dtype=output_dtype,
        device=quantized.q.device,
    )

    stream_ptr = torch.cuda.current_stream(quantized.q.device).cuda_stream
    _cuda_backend._C.sage_sdpa_prequantized(
        _cuda_backend._wrap_for_dlpack(quantized.q),
        _cuda_backend._wrap_for_dlpack(quantized.k),
        _cuda_backend._wrap_for_dlpack(quantized.v),
        _cuda_backend._wrap_for_dlpack(output),
        _cuda_backend._wrap_for_dlpack(quantized.q_scale),
        _cuda_backend._wrap_for_dlpack(quantized.k_scale),
        _cuda_backend._wrap_for_dlpack(quantized.v_scale),
        quantized.cta_k,
        quantized.attention_scale,
        DTYPE_TO_CODE[output_dtype],
        stream_ptr,
        (
            _cuda_backend._wrap_for_dlpack(quantized.attn_mask)
            if quantized.attn_mask is not None
            else None
        ),
    )

    output = output[..., : quantized.original_head_dim]
    return output.float() if quantized.input_dtype == torch.float32 else output


@torch.library.custom_op("comfy_kitchen::int8_attention", mutates_args=())
def _op_int8_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None,
) -> torch.Tensor:
    return _int8_attention_cuda(
        q,
        k,
        v,
        scale=scale,
        attn_mask=None,
    )


@_op_int8_attention.register_fake
def _op_int8_attention_fake(
    q,
    k,
    v,
    scale,
):
    return q.new_empty(q.shape)


@torch.library.custom_op("comfy_kitchen::int8_attention_masked", mutates_args=())
def _op_int8_attention_masked(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor,
    scale: float | None,
) -> torch.Tensor:
    return _int8_attention_cuda(
        q,
        k,
        v,
        scale=scale,
        attn_mask=attn_mask,
    )


@_op_int8_attention_masked.register_fake
def _op_int8_attention_masked_fake(
    q,
    k,
    v,
    attn_mask,
    scale,
):
    return q.new_empty(q.shape)


def int8_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float | None = None,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute inference SDPA with signed INT8 Q/K/V and unsigned INT8 P.

    Inputs use ``[batch, heads, sequence, head_dim]`` layout. Grouped-query
    attention and unequal non-causal Q/K sequence lengths are supported. Head
    dimensions are padded to the kernel's 64-, 128-, or 256-wide tile and
    sliced back on return; 64, 128, and 256 take the zero-copy dimension path.
    Q and K receive the same fused block-Hadamard rotation before INT8
    quantization, preserving their exact dot products while reducing
    quantization outliers. K lengths up to 256 use low-overhead H4 blocks and
    longer D64 attention uses H64. The common D128 path uses a fixed signed H128
    transform, while padded D256 uses plain H128. The kernel samples K on the
    GPU and subtracts a representative key only when doing so improves the
    quantization range. This model-independent shift is exactly
    softmax-invariant and uses one int32 of temporary storage per batch/KV-head.
    Softmax score, maximum, exponential, denominator, reciprocal, and V-scale
    arithmetic is FP32. This path does not allocate FP8 tensors or execute FP8
    MMA instructions.

    The path only pays for itself on long sequences: quantizing Q, K and V is a
    fixed cost per call that the faster kernel has to earn back. On the HIP
    backend a call with q_len * kv_len below COMFY_KITCHEN_INT8_MIN_QK squared
    (default 8192 per segment) therefore runs as scaled_dot_product_attention
    with the caller's mask and scale, which is the same thing the flash baseline
    computes. Set that variable to 0 to take the INT8 path unconditionally.
    """
    if attn_mask is None:
        return torch.ops.comfy_kitchen.int8_attention(
            q,
            k,
            v,
            scale,
        )
    return torch.ops.comfy_kitchen.int8_attention_masked(
        q,
        k,
        v,
        attn_mask,
        scale,
    )
