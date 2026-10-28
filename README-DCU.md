# comfy-kitchen — Hygon DCU (gfx9) port

This is a fork of [`Comfy-Org/comfy-kitchen`](https://github.com/Comfy-Org/comfy-kitchen)
with a HIP backend port for **Hygon DCU** GPUs of the **gfx9** family
(`gfx906` / `gfx928` / `gfx938`).

Upstream's HIP backend targets AMD's WMMA matrix core (`gfx11` / `gfx12`).
Hygon DCU exposes a different matrix core through the **`v_mmac_*`** instruction
family, so each kernel has to be ported individually. This fork adds that path.

## Baseline

- **Upstream:** comfy-kitchen **v0.2.35** (`b2a2972`)
- The port was originally written against a 0.2.31 baseline and then
  **forward-ported** to v0.2.35. Downgrading the installed package to 0.2.31 is
  *not* an option: ComfyUI calls `gated_delta_decode_fused`, `deltanet_conv_step`
  and `set_allocation_context`, which only exist in 0.2.35.

## What changed

16 files, `+1065 / −131` vs. v0.2.35. Full diff: [`patches/comfy_kitchen_gfx9_0.2.35.patch`](patches/comfy_kitchen_gfx9_0.2.35.patch).

| Area | Files |
|---|---|
| Build / arch plumbing | `CMakeLists.txt`, `architectures.json`, `architecture_config.h.in` |
| Runtime gating | `__init__.py` |
| MMA primitives | `mma.h`, `gemm_wmma.h` |
| Kernels | `ops/{convrot_w4a4,flash_decode,gemm_fp8,gemm_int8,gemv_awq,na3d,svdquant_w4a4,w4a8_gemv}.hip` |
| Attention | `sage_attention/{int8_attn.hip,sage_common.h}` |

### Key points

- **New arch group** `mmac_gfx9` = `gfx906;gfx928;gfx938`, wired through
  `architectures.json` → `CMakeLists.txt` → `architecture_config.h.in`.
- **Instruction features:** the i8/b16 MMAC family needs `+mmop2-insts`; the i4
  family needs `+mmop3-insts`. `gfx928` has `mmop2` on by default, `gfx906` /
  `gfx938` do not. The flags are added only when a gfx9 target is in the build.
- **i4 caveat:** the i4 (`mmop3`) family *compiles* with clang but **traps on the
  silicon**, so int4 ops run through an **int8 fallback**.
- **fp8 caveat:** gfx9 has **no fp8 matrix instruction**. `gemm_fp8` still compiles
  and dispatches, but `mma.h` widens the fp8 operands to bf16, so it is an fp8-*storage* /
  bf16-*compute* path — not fp8 compute. Do not expect a speedup from fp8 over
  bf16 on this target; the quantized path that actually uses the matrix core here
  is int8.
- **Unsigned operands:** `u8` MMAC requires *both* operands unsigned.
- **Wave size:** `kWave` is 32 in host passes on this target; launch
  configurations must use the *device* wave size, not the host value.
- **CMake dedup:** CMake de-duplicates `target_compile_options`, so four separate
  `-Xclang -target-feature` tokens would collapse into one. Each feature pair is
  wrapped in its own `SHELL:` item to keep every argument (and stay unique).
- **Gating for 0.2.35-only ops:** 0.2.35 added `sol_attn`, `fp16_linear` and
  `fp16_conv3d`, which only have AMD-WMMA kernel bodies. These are gated behind
  `_RDNA_WMMA_ONLY_OPS` and dropped on pure-DCU machines. `gated_delta_decode`
  (`gated_delta.hip`) is also gated on RDNA WMMA, since it is written for wave32.

## Build

Requires a HIP toolchain (Hygon DTK) with `clang++` and Ninja.

```sh
cmake -S comfy_kitchen/backends/hip -B build/hip -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_COMPILER=/opt/dtk/llvm/bin/clang++ \
  -DCOMFY_HIP_ARCHS="gfx906;gfx928;gfx938"

cmake --build build/hip
```

Then install the resulting `_C.abi3.so` into
`<site-packages>/comfy_kitchen/backends/hip/_C.abi3.so`.

## Prebuilt binaries

See [`prebuilt/`](prebuilt/):

- `_C.abi3.so.gfx906_928_938` — fat build for all three DCU targets (**recommended**)
- `_C.abi3.so.gfx928` — gfx928-only build (smaller)

## Verification

- **Numerical:** 9/9 kernel-level tests pass.
- **A/B on real hardware** (MiniMax H3 int8-convrot, image-to-video, 20 steps,
  fixed seed; the only variable was the HIP `_C.abi3.so`):

  | | fallback (`has_wmma=False`) | gfx9 `v_mmac` | speedup |
  |---|---|---|---|
  | sampling | 14.51 s/it | **11.62 s/it** | **1.25×** |
  | `Prompt executed` | 538.55 s | **483.27 s** | 1.11× |
  | end-to-end | 542.8 s | **486.0 s** | 1.12× |

  Outputs are semantically consistent; the per-frame divergence is the expected
  cross-kernel diffusion spread, and the fallback path reproduces the historical
  baseline bit-for-bit.

## Known limitations

- `w4a8_gemv` (the 0.2.35 codebook GEMV) compiles on gfx9 via a scalar
  `dot4_i8` fallback, but is **not numerically verified**. It is reached by
  `w4a8_int8_linear` at `m <= 8` with fp8 scales.
- Only the gfx9 gating was touched; the AMD WMMA paths are unchanged.

## License

Apache-2.0 — see [`LICENSE`](LICENSE). This fork redistributes upstream
comfy-kitchen source; upstream copyright notices are retained (see
[`NOTICE`](NOTICE)).
