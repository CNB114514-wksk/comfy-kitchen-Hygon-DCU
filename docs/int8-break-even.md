# Choosing the INT8 attention threshold

The fused INT8 attention path in `comfy_kitchen.sage_attention` quantizes Q, K
and V before the kernel can start. That preamble is a fixed cost per call: it
does not shrink with the sequence, so on short segments it costs more than the
faster kernel saves. `COMFY_KITCHEN_INT8_MIN_QK` is the gate that keeps those
calls on SDPA.

## The test that is applied

```python
q_length * kv_length >= threshold ** 2   # INT8
q_length * kv_length <  threshold ** 2   # SDPA (flash) fallback
```

The variable is a **length**, not a product: the gate compares the geometric
mean of the two segment lengths (`sqrt(q_len * kv_len)`) against it. Square
attention therefore reduces to "sequence length below the threshold", and the
asymmetric cases stay sane. The text segment of a block-causal image model is a
few dozen queries over a few thousand keys; a bare product would put it far
above any threshold and quantize thousands of V rows to answer forty queries,
while `min()` would gate on the wrong side of long-query/short-key attention.

`0` disables the gate (unconditional INT8, the old behaviour). A non-integer
value falls back to the default with a warning.

## Measured points

gfx9 DCU (K500SM_AI), `q_len == kv_len`, bf16, against
`torch.nn.functional.scaled_dot_product_attention` as the flash baseline:

| tokens (S) | INT8 vs SDPA | source |
|---|---|---|
| 4096 | 1.81x slower | microbenchmark |
| 16384 | 1.39x slower | microbenchmark |
| 32760 | 1.12x slower | Wan2.1-I2V-14B-480P end to end |
| 35880 | 1.28x faster | microbenchmark |

The Wan figure is the one that matters for video: 832x480, 81 frames, 20 steps,
32,760 self-attention tokens. Steady-state sampling was 1063 s on flash and
1193 s on INT8 (+12.2%); end to end it was 1080.87 s against 1210.63 s
(+12.0%). The microbenchmarks are per-call kernel timings on the same device.

## The chosen default: 35840

The crossover sits between 32760 and 35880, near 35.8k. `35840` is used
because:

- it is above every measured loss, with 9.4% of headroom over 32760, so the
  +12% video regression can no longer be selected by the gate;
- it is still below the only measured win (35880), so the gate does not discard
  it;
- it is a multiple of 512, which keeps the value on a tile boundary if the
  kernel tiling is ever tied to the threshold.

The gap is deliberately biased toward the loss side. Falling back to SDPA at a
length where INT8 would have won costs a fraction of a percent of one
attention call; taking INT8 where it loses cost 12% of the whole model.

A finer sweep over 32760..35880 would place the crossover more precisely, and
the variable can be raised or lowered per workload without a rebuild.

## Reproducing

`measure_int8_gating.py` in the `int8_shortseq_20261108` work directory runs
SDPA / INT8(`COMFY_KITCHEN_INT8_MIN_QK=0`) / INT8(default) at a list of
lengths, with `--profile` for the quantizer's share and `--e2e-server` against
a live ComfyUI.
