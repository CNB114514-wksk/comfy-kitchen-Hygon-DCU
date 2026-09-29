// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Pure INT8 attention on RDNA matrix cores. The CUDA backend's sage_attention/
// sources are the reference for what this computes; how it computes it differs,
// because the RDNA fragment layout does.
//
// The score matrix is evaluated TRANSPOSED, S^T = K @ Q^T, and the output with
// it, O^T = V^T @ P^T. The WMMA accumulator hands a lane one column and eight
// rows, so under S^T a lane owns one query and eight keys: the online softmax
// becomes per-lane scalar work with a single shuffle to fold the half-waves, and
// the query stays on the lane holding its denominator. On gfx12 the score
// accumulator is then already the P operand of the second MMA, element for
// element, so P never goes through LDS; gfx11 interleaves the accumulator rows
// across half-waves and pays two shuffles and two byte permutes for it.
//
// Tiling: 128 queries per block, 16 per wave, 64 keys per iteration; LDS holds
// only K and V. A wider query tile, a 128-key tile, staging the tiles through
// registers across the barrier, and folding the mask into the score loop were
// all tried and are all slower or spill. Do not re-try them blind.
#include <hip/hip_runtime.h>

#include <stdexcept>
#include <string>

#include "../mma.h"
#include "sage_common.h"

namespace comfy::hip_backend::sage {

// Mirrors MaskMode in the CUDA backend's attn_utils.cuh.
enum class MaskMode { kNone = 0, kCustom = 2 };

namespace {

// The RDNA body below is wave32 and WMMA-shaped. The Hygon DCU is wave64 with
// Hygon's v_mmac_* matrix core and a different accumulator layout, so it gets its
// own body rather than a policy swap. Both define launch_one with the same
// signature; everything after the anonymous namespace is shared.

__forceinline__ __device__ int imin(int a, int b) { return a < b ? a : b; }

constexpr int div_ceil_host(int a, int b) { return (a + b - 1) / b; }

// Round to nearest even and saturate, matching cvt.rni.sat.u8.f32 in the CUDA
// backend's pack_u8x4.
__forceinline__ __device__ uint32_t prob_to_u8(float p) {
    return static_cast<uint32_t>(fminf(255.0f, fmaxf(0.0f, rintf(p))));
}

// Reads one mask element as raw bits. -ffast-math folds isfinite() away, so the
// non-finite test that decides whether an additive bias means "drop this key"
// has to be an exponent bit test on the value as loaded, never a float compare.
// See the fast-math note in hip/mma.h's sibling kernels.
__forceinline__ __device__ bool mask_keep(const void* mask, int64_t offset, int dtype_code,
                                          float& bias) {
    bias = 0.0f;
    if (dtype_code == 3) {
        return static_cast<const uint8_t*>(mask)[offset] != 0;
    }
    if (dtype_code == 0) {
        const uint32_t bits = static_cast<const uint32_t*>(mask)[offset];
        if ((bits & 0x7F800000u) == 0x7F800000u) return false;
        bias = __uint_as_float(bits);
        return true;
    }
    const uint16_t bits = static_cast<const uint16_t*>(mask)[offset];
    if (dtype_code == 1) {
        if ((bits & 0x7C00u) == 0x7C00u) return false;
        bias = __half2float(*reinterpret_cast<const __half*>(&bits));
        return true;
    }
    if ((bits & 0x7F80u) == 0x7F80u) return false;
    bias = static_cast<float>(*reinterpret_cast<const __bf16*>(&bits));
    return true;
}

#if !defined(COMFY_MMA_GFX9)

constexpr int kCtaQ = 128;
// gfx11 reads a whole 16-byte K-step per lane, so a 16-byte row keeps that to one
// LDS instruction; the wider pad costs a two-way bank conflict and buys back four
// times the issue slots. gfx12 reads 8 bytes, where an 8-byte pad is already one
// instruction and keeps 16 rows on 16 distinct banks.
#if defined(COMFY_MMA_GFX11)
constexpr int kLdsPad = 16;
#else
constexpr int kLdsPad = 8;
#endif

// 16 queries per wave at every head_dim. The CUDA launcher widens D64 to 32 to
// halve its shared-memory traffic; here the output accumulator is transposed, so
// a wider query tile grows the live FP32 set instead and spills on both families.
template <int HD>
struct Shape {
    static constexpr int kWarpQ = 16;
    static constexpr int kWaves = kCtaQ / kWarpQ;
    static constexpr int kThreads = kWaves * 32;
    static constexpr int kQueryTiles = kWarpQ / 16;
    static constexpr int kDimTiles = HD / 16;
};

template <int HD, int CTA_K, MaskMode kMask, typename OutT>
__global__ __launch_bounds__(Shape<HD>::kThreads) void int8_attn_kernel(
    const int8_t* __restrict__ q, const int8_t* __restrict__ k, const int8_t* __restrict__ v,
    OutT* __restrict__ o, const float* __restrict__ q_scale, const float* __restrict__ k_scale,
    const float* __restrict__ v_scale, const void* __restrict__ mask, int64_t mask_stride_b,
    int64_t mask_stride_h, int64_t mask_stride_q, int64_t mask_stride_k, int mask_dtype_code,
    int qo_len, int kv_len, int qo_len_padded, int k_groups_per_head, int num_kv_groups,
    int64_t q_stride_b, int64_t q_stride_h, int64_t k_stride_b, int64_t k_stride_h,
    int64_t v_stride_b, int64_t v_stride_h, int64_t v_stride_d, int64_t o_stride_b,
    int64_t o_stride_h, int64_t o_stride_n, float sm_scale) {

    using S = Shape<HD>;
    constexpr int QT = S::kQueryTiles;
    constexpr int KT = CTA_K / 16;
    constexpr int DT = S::kDimTiles;
    constexpr int kStrideK = HD + kLdsPad;
    constexpr int kStrideV = CTA_K + kLdsPad;
    constexpr bool kCustomMask = kMask == MaskMode::kCustom;

    __shared__ __attribute__((aligned(16))) int8_t smem_k[CTA_K * kStrideK];
    __shared__ __attribute__((aligned(16))) int8_t smem_v[HD * kStrideV];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wave = tid >> 5;
    const int r = frag_row(lane);

    const int block_q = blockIdx.x * kCtaQ;
    const int head = blockIdx.y;
    const int batch = blockIdx.z;
    const int num_qo_heads = gridDim.y;
    const int kv_head = head / num_kv_groups;

    const int8_t* q_head = q + static_cast<int64_t>(batch) * q_stride_b +
                           static_cast<int64_t>(head) * q_stride_h;
    const int8_t* k_head = k + static_cast<int64_t>(batch) * k_stride_b +
                           static_cast<int64_t>(kv_head) * k_stride_h;
    const int8_t* v_head = v + static_cast<int64_t>(batch) * v_stride_b +
                           static_cast<int64_t>(kv_head) * v_stride_h;

    // Base 2 throughout: the probabilities come out of exp2, not exp.
    sm_scale *= kLog2e;

    // Q is loop invariant and worth holding in registers, but only while it fits:
    // gfx11 fragments are twice as wide, and 16 of them at D256 sit on top of an
    // output accumulator that is already half the file. Past that the wave
    // reloads Q from global, which beats spilling the accumulator.
    constexpr bool kCacheQ = DT * static_cast<int>(sizeof(typename MmaInt8::Frag)) / 4 <= 32;
    typename MmaInt8::Frag q_frag[QT][kCacheQ ? DT : 1];
    const int8_t* q_rows[QT];
    float q_lane_scale[QT];
    int q_lane_idx[QT];
#pragma unroll
    for (int fq = 0; fq < QT; ++fq) {
        // A row past the end reads a clamped one; its output is never stored and
        // its scale is finite, so nothing downstream sees it.
        const int q_idx = block_q + wave * S::kWarpQ + fq * 16 + r;
        q_lane_idx[fq] = q_idx;
        q_rows[fq] = q_head + static_cast<int64_t>(imin(q_idx, qo_len - 1)) * HD;
        if constexpr (kCacheQ) {
#pragma unroll
            for (int dt = 0; dt < DT; ++dt) {
                q_frag[fq][dt] = MmaInt8::load_aligned(q_rows[fq], 0, dt * 16, 0, lane);
            }
        }
        q_lane_scale[fq] =
            q_scale[(static_cast<int64_t>(batch) * num_qo_heads + head) * qo_len_padded +
                    imin(q_idx, qo_len_padded - 1)];
    }

    float o_val[QT][DT][8];
#pragma unroll
    for (int fq = 0; fq < QT; ++fq) {
#pragma unroll
        for (int dt = 0; dt < DT; ++dt) {
#pragma unroll
            for (int e = 0; e < 8; ++e) o_val[fq][dt][e] = 0.0f;
        }
    }

    // Finite sentinels, not -inf: every masked score still goes through fma and
    // exp2 before anything looks at it. Matches the CUDA initial state, d
    // included, so a fully masked row divides by one instead of zero.
    float m_run[QT], d_run[QT];
    bool row_valid[QT];
#pragma unroll
    for (int fq = 0; fq < QT; ++fq) {
        m_run[fq] = kMaskedScore;
        d_run[fq] = 1.0f;
        row_valid[fq] = !kCustomMask;
    }

    // An interior tile has every key in bounds, so nothing needs testing per
    // element: its maximum comes off the INT32 accumulator and exp2 reads the
    // accumulator too, which keeps a tile of scaled scores out of registers. A
    // custom mask has no interior.
    const int num_iters = (kv_len + CTA_K - 1) / CTA_K;

    // D256 keeps one code path: its output accumulator is 128 live floats, and
    // emitting the body twice there spills on both families.
    constexpr bool kSplitLoop = DT <= 8;

    int interior_iters = 0;
    if constexpr (!kCustomMask && kSplitLoop) {
        // Taking the maximum in INT32 assumes the dequantization scale is
        // positive. A negative softmax scale is legal and finite, so it gives up
        // the fast path rather than reporting the wrong maximum.
        if (sm_scale >= 0.0f) {
            interior_iters = kv_len / CTA_K;
        }
    }

    constexpr int kUnitsK = HD / 8;
    constexpr int kUnitsV = CTA_K / 8;

    // `boundary` is a compile-time tag, so the interior instance carries no mask
    // test and no per-key bounds check at all.
    auto run_iter = [&](auto boundary_tag, int iter) {
        constexpr bool kBoundary = decltype(boundary_tag)::value;
        const int kb0 = iter * CTA_K;

        __syncthreads();
        // K tile: [key][d]. A row past the end is zeroed; its score is masked
        // below. Staged straight to LDS: prefetching across the barrier was tried
        // and loses, because this kernel waits on LDS, not on global.
        for (int idx = tid; idx < CTA_K * kUnitsK; idx += S::kThreads) {
            const int row = idx / kUnitsK;
            const int unit = idx % kUnitsK;
            const int key = kb0 + row;
            uint2 val = {0u, 0u};
            if (!kBoundary || key < kv_len) {
                val = *reinterpret_cast<const uint2*>(k_head + static_cast<int64_t>(key) * HD +
                                                      unit * 8);
            }
            *reinterpret_cast<uint2*>(smem_k + row * kStrideK + unit * 8) = val;
        }
        // V tile: [d][key], already transposed and zero padded by the quantizer,
        // so a whole CTA_K tile is always in bounds.
        for (int idx = tid; idx < HD * kUnitsV; idx += S::kThreads) {
            const int row = idx / kUnitsV;
            const int unit = idx % kUnitsV;
            *reinterpret_cast<uint2*>(smem_v + row * kStrideV + unit * 8) =
                *reinterpret_cast<const uint2*>(v_head + static_cast<int64_t>(row) * v_stride_d +
                                                kb0 + unit * 8);
        }
        __syncthreads();

        float key_scale[KT];
#pragma unroll
        for (int fk = 0; fk < KT; ++fk) {
            key_scale[fk] =
                k_scale[(static_cast<int64_t>(batch) * (num_qo_heads / num_kv_groups) + kv_head) *
                            k_groups_per_head +
                        iter * KT + fk];
        }

#pragma unroll
        for (int fq = 0; fq < QT; ++fq) {
            typename MmaInt8::Acc s_acc[KT];
#pragma unroll
            for (int fk = 0; fk < KT; ++fk) {
                s_acc[fk] = MmaInt8::zero();
#pragma unroll
                for (int dt = 0; dt < DT; ++dt) {
                    const typename MmaInt8::Frag qf =
                        kCacheQ ? q_frag[fq][kCacheQ ? dt : 0]
                                : MmaInt8::load_aligned(q_rows[fq], 0, dt * 16, 0, lane);
                    s_acc[fk] = MmaInt8::mma(
                        MmaInt8::load_aligned(smem_k, fk * 16 + r, dt * 16, kStrideK, lane), qf,
                        s_acc[fk]);
                }
            }

            const int q_idx = q_lane_idx[fq];
            float tile_scale_k[KT];
#pragma unroll
            for (int fk = 0; fk < KT; ++fk) {
                tile_scale_k[fk] = sm_scale * q_lane_scale[fq] * key_scale[fk];
            }

            // A boundary tile dequantizes into a live array, because the mask has
            // to land on the scaled score. Folding the mask into the score loop
            // instead, to retire the accumulator earlier, spills D256.
            float prob[kBoundary ? KT : 1][8];
            float tile_m;

            if constexpr (kBoundary) {
                // Hoisted: only the key term varies per element, and eight live
                // 64-bit addresses in the inner loop cost registers this kernel
                // does not have. A key-only mask needs no separate instantiation:
                // it arrives with mask_stride_q at zero, so the query term drops
                // out here at runtime and the two kernels were byte-identical.
                int64_t mask_row = 0;
                if constexpr (kCustomMask) {
                    mask_row = static_cast<int64_t>(batch) * mask_stride_b +
                               static_cast<int64_t>(head) * mask_stride_h +
                               static_cast<int64_t>(imin(q_idx, qo_len - 1)) * mask_stride_q;
                }
#pragma unroll
                for (int fk = 0; fk < KT; ++fk) {
#pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        const int key = kb0 + fk * 16 + acc_row(lane, e);
                        float score = static_cast<float>(s_acc[fk][e]) * tile_scale_k[fk];
                        bool keep = key < kv_len;
                        if constexpr (kCustomMask) {
                            keep = keep && q_idx < qo_len;
                            if (keep) {
                                float bias;
                                keep = mask_keep(
                                    mask,
                                    mask_row + static_cast<int64_t>(key) * mask_stride_k,
                                    mask_dtype_code, bias);
                                score += bias * kLog2e;
                            }
                            row_valid[fq] = row_valid[fq] || keep;
                        }
                        prob[fk][e] = keep ? score : kMaskedScore;
                    }
                }
                tile_m = prob[0][0];
#pragma unroll
                for (int fk = 0; fk < KT; ++fk) {
#pragma unroll
                    for (int e = 0; e < 8; ++e) tile_m = fmaxf(tile_m, prob[fk][e]);
                }
            } else {
                tile_m = kMaskedScore;
#pragma unroll
                for (int fk = 0; fk < KT; ++fk) {
                    int key_max = s_acc[fk][0];
#pragma unroll
                    for (int e = 1; e < 8; ++e) {
                        key_max = key_max > s_acc[fk][e] ? key_max : s_acc[fk][e];
                    }
                    tile_m = fmaxf(tile_m, static_cast<float>(key_max) * tile_scale_k[fk]);
                }
            }

            // The tile maximum is shifted so exp2 fills the unsigned INT8 range.
            // Its gap to the running maximum rides on the accumulator side as
            // tile_scale, which underflows to zero for a fully masked tile.
            tile_m = fmaxf(tile_m, swap_half_wave(tile_m));
            tile_m -= kProbU8Offset;

            const float m_prev = m_run[fq];
            const float smaller = exp2f(-fabsf(m_prev - tile_m));
            const float o_scale = m_prev < tile_m ? smaller : 1.0f;
            const float tile_scale = tile_m < m_prev ? smaller : 1.0f;
            m_run[fq] = fmaxf(m_prev, tile_m);
            d_run[fq] *= o_scale;

#pragma unroll
            for (int dt = 0; dt < DT; ++dt) {
#pragma unroll
                for (int e = 0; e < 8; ++e) o_val[fq][dt][e] *= o_scale;
            }

            typename MmaInt8::Frag p_frag[KT];
            float tile_sum = 0.0f;
#pragma unroll
            for (int fk = 0; fk < KT; ++fk) {
                uint32_t packed[8];
                const float negative_m = -tile_m;
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const float p =
                        kBoundary
                            ? exp2f(prob[kBoundary ? fk : 0][e] + negative_m)
                            : exp2f(fmaf(static_cast<float>(s_acc[fk][e]), tile_scale_k[fk],
                                         negative_m));
                    tile_sum += p;
                    packed[e] = prob_to_u8(p);
                }
                p_frag[fk] = pack_prob_frag(packed, lane);
            }
            d_run[fq] += tile_sum * tile_scale;

            // The INT32 partial covers one CTA_K tile, which is what keeps the
            // running output in FP32 without a second accumulator set per tile.
#pragma unroll
            for (int dt = 0; dt < DT; ++dt) {
                typename MmaInt8::Acc partial = MmaInt8::zero();
#pragma unroll
                for (int fk = 0; fk < KT; ++fk) {
                    partial = MmaInt8::mma_ub(
                        MmaInt8::load_aligned(smem_v, dt * 16 + r, fk * 16, kStrideV, lane), p_frag[fk],
                        partial);
                }
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    o_val[fq][dt][e] =
                        fmaf(static_cast<float>(partial[e]), tile_scale, o_val[fq][dt][e]);
                }
            }
        }
    };

    int iter = 0;
    for (; iter < interior_iters; ++iter) run_iter(std::false_type{}, iter);
    for (; iter < num_iters; ++iter) run_iter(std::true_type{}, iter);

    // The two half-waves summed different keys of the same query.
#pragma unroll
    for (int fq = 0; fq < QT; ++fq) {
        d_run[fq] += swap_half_wave(d_run[fq]);
        if constexpr (kCustomMask) {
            int any = row_valid[fq] ? 1 : 0;
            any |= static_cast<int>(swap_half_wave_b32(static_cast<uint32_t>(any)));
            row_valid[fq] = any != 0;
        }
    }

    const float* v_scale_head =
        v_scale + (static_cast<int64_t>(batch) * (num_qo_heads / num_kv_groups) + kv_head) * HD;

#pragma unroll
    for (int fq = 0; fq < QT; ++fq) {
        const int q_idx = q_lane_idx[fq];
        if (q_idx >= qo_len) continue;
        const float inv_d = row_valid[fq] ? 1.0f / d_run[fq] : 0.0f;
        OutT* row = o + static_cast<int64_t>(batch) * o_stride_b +
                    static_cast<int64_t>(head) * o_stride_h +
                    static_cast<int64_t>(q_idx) * o_stride_n;
#pragma unroll
        for (int dt = 0; dt < DT; ++dt) {
            float vals[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int d = dt * 16 + acc_row(lane, e);
                vals[e] = o_val[fq][dt][e] * v_scale_head[d] * inv_d;
            }
            store_o_tile<OutT>(row, dt * 16, vals, lane);
        }
    }
}

template <int HD, int CTA_K, MaskMode kMask, typename OutT>
void launch_one(const int8_t* q, const int8_t* k, const int8_t* v, OutT* o, const float* q_scale,
                const float* k_scale, const float* v_scale, const void* mask,
                int64_t mask_stride_b, int64_t mask_stride_h, int64_t mask_stride_q,
                int64_t mask_stride_k, int mask_dtype_code, int batch, int qo_len, int kv_len,
                int qo_len_padded, int num_qo_heads, int k_groups_per_head, int num_kv_groups,
                int64_t q_stride_b, int64_t q_stride_h, int64_t k_stride_b, int64_t k_stride_h,
                int64_t v_stride_b, int64_t v_stride_h, int64_t v_stride_d, int64_t o_stride_b,
                int64_t o_stride_h, int64_t o_stride_n, float sm_scale, hipStream_t stream) {

    const dim3 grid(div_ceil_host(qo_len, kCtaQ), num_qo_heads, batch);
    // The host pass never defines COMFY_MMA_GFX9, so on a gfx9 build it compiles
    // THIS launcher while the device runs the gfx9 body above. That body's
    // staging loops stride by kThreads9 = kWaves9 * 64, so the block has to be
    // sized from the device's wave (64 on gfx9, 32 elsewhere); Shape<HD>::kThreads
    // is the wave32 constant and would leave the upper half of the V tile
    // unstaged, zeroing every output channel past the first 64.
    const int threads = Shape<HD>::kWaves * device_wave_size();
    int8_attn_kernel<HD, CTA_K, kMask, OutT><<<grid, threads, 0, stream>>>(
        q, k, v, o, q_scale, k_scale, v_scale, mask, mask_stride_b, mask_stride_h, mask_stride_q,
        mask_stride_k, mask_dtype_code, qo_len, kv_len, qo_len_padded, k_groups_per_head,
        num_kv_groups, q_stride_b, q_stride_h, k_stride_b, k_stride_h, v_stride_b, v_stride_h,
        v_stride_d, o_stride_b, o_stride_h, o_stride_n, sm_scale);
}

#else  // COMFY_MMA_GFX9

// ---------------------------------------------------------------------------
// Hygon DCU (gfx906..gfx938) int8 attention: wave64, v_mmac_i32_16x16x32.
//
// The RDNA body evaluates the scores TRANSPOSED, S^T = K @ Q^T, because its
// accumulator gives a lane one column and eight rows, which turns the online
// softmax into a per-lane reduction folded by one shuffle. The Hygon
// accumulator gives a lane one ROW and four columns four apart,
// C[l%16][4*i + l/16], so the transposed form would scatter one query across
// sixteen lanes. This body evaluates S = Q @ K^T instead: a lane owns query
// l%16 and the keys {l/16, 4+l/16, 8+l/16, 12+l/16}, and the four lanes that
// share l%16 are one query's group -- they cover its sixteen keys and fold with
// two shuffles.
//
// Tiling: 128 queries per block, 16 per wave (a wave is 64 lanes), 32 keys per
// iteration. 32 is the i8 MMA's K, so one iteration's probabilities are exactly
// one P fragment. K is staged key-major and V channel-major, which is the order
// each already has in global memory, and that makes both MMA B operands eight
// contiguous bytes.
// ---------------------------------------------------------------------------

constexpr int kCtaQ9 = 128;
constexpr int kWarpQ9 = 16;
constexpr int kWaves9 = kCtaQ9 / kWarpQ9;
constexpr int kThreads9 = kWaves9 * 64;
constexpr int kCtaK9 = 32;
constexpr int kLdsPad9 = 8;
// Two 16-key score tiles per iteration; the score accumulator covers one at a
// time.
constexpr int kScoreTiles9 = kCtaK9 / 16;
// P is [query][key] with 32 keys a row. The four-byte pad puts the sixteen rows
// of a wave on sixteen distinct bank pairs.
constexpr int kStrideP9 = kCtaK9 + 4;

// Fold one query's group: the four lanes with the same l%16, i.e. l/16 = 0..3.
// XOR with 16 and 32 stays inside that group because both preserve the low four
// bits, which carry the query index.
__forceinline__ __device__ float query_group_max9(float v) {
    v = fmaxf(v, __shfl_xor(v, 16, 64));
    v = fmaxf(v, __shfl_xor(v, 32, 64));
    return v;
}
__forceinline__ __device__ float query_group_sum9(float v) {
    v += __shfl_xor(v, 16, 64);
    v += __shfl_xor(v, 32, 64);
    return v;
}
__forceinline__ __device__ int query_group_any9(int v) {
    v |= __shfl_xor(v, 16, 64);
    v |= __shfl_xor(v, 32, 64);
    return v;
}

// The score MMA's B operand is K^T, but K is staged key-major, so the fragment
// this lane owns -- K^T[channel 8*(l/16) + p][key] -- runs along the channel
// axis and reads exactly like an A fragment whose row is the key. `key` is the
// logical key of the fragment column, which is 16*t + l%16 for score tile t.
__forceinline__ __device__ long load_frag_kt9(const void* lds, int key, int kbyte, int stride,
                                              int lane) {
    return load_frag_b64_a(lds, key, kbyte, stride, lane);
}

// Output element i of this lane is channel 4*i + l/16, one row per lane.
template <typename OutT>
__forceinline__ __device__ void store_o_tile9(OutT* __restrict__ row, int d_base,
                                              const float* vals, int lane) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        row[d_base + 4 * i + (lane / 16)] = static_cast<OutT>(vals[i]);
    }
}

// Same name and template parameter list as the RDNA body above: the two are
// mutually exclusive per compilation pass, and the host pass never sees
// COMFY_MMA_GFX9, so it registers the RDNA spelling. A different device-side
// name would leave the launch pointing at a symbol that is not in the image.
// CTA_K is ignored here -- see launch_one below.
template <int HD, int CTA_K, MaskMode kMask, typename OutT>
__global__ __launch_bounds__(kThreads9) void int8_attn_kernel(
    const int8_t* __restrict__ q, const int8_t* __restrict__ k, const int8_t* __restrict__ v,
    OutT* __restrict__ o, const float* __restrict__ q_scale, const float* __restrict__ k_scale,
    const float* __restrict__ v_scale, const void* __restrict__ mask, int64_t mask_stride_b,
    int64_t mask_stride_h, int64_t mask_stride_q, int64_t mask_stride_k, int mask_dtype_code,
    int qo_len, int kv_len, int qo_len_padded, int k_groups_per_head, int num_kv_groups,
    int64_t q_stride_b, int64_t q_stride_h, int64_t k_stride_b, int64_t k_stride_h,
    int64_t v_stride_b, int64_t v_stride_h, int64_t v_stride_d, int64_t o_stride_b,
    int64_t o_stride_h, int64_t o_stride_n, float sm_scale) {

    constexpr int kDimTiles = HD / 32;           // 32 channels per score MMA
    constexpr int kOutTiles = HD / 16;           // 16 channels per output MMA
    constexpr int kUnits = HD / 8;
    constexpr int kUnitsV = kCtaK9 / 8;
    constexpr int kStrideK = HD + kLdsPad9;      // [key][d]
    constexpr int kStrideV = kCtaK9 + kLdsPad9;  // [d][key]
    constexpr bool kCustomMask = kMask == MaskMode::kCustom;

    // K key-major, V channel-major -- each the order it already has in global.
    // P is per wave: 16 queries by 32 keys, staged so the second MMA reads a
    // lane's eight consecutive keys in one load.
    __shared__ __attribute__((aligned(16))) int8_t smem_k[kCtaK9 * kStrideK];
    __shared__ __attribute__((aligned(16))) int8_t smem_v[HD * kStrideV];
    __shared__ __attribute__((aligned(16))) uint8_t smem_p[kWaves9 * kWarpQ9 * kStrideP9];

    const int tid = threadIdx.x;
    const int lane = tid & 63;
    const int wave = tid >> 6;
    const int q_lane = lane % 16;   // this lane's query within the wave
    const int g = lane / 16;        // which quarter of the keys it owns

    const int block_q = blockIdx.x * kCtaQ9;
    const int head = blockIdx.y;
    const int batch = blockIdx.z;
    const int num_qo_heads = gridDim.y;
    const int kv_head = head / num_kv_groups;

    const int8_t* q_head = q + static_cast<int64_t>(batch) * q_stride_b +
                           static_cast<int64_t>(head) * q_stride_h;
    const int8_t* k_head = k + static_cast<int64_t>(batch) * k_stride_b +
                           static_cast<int64_t>(kv_head) * k_stride_h;
    const int8_t* v_head = v + static_cast<int64_t>(batch) * v_stride_b +
                           static_cast<int64_t>(kv_head) * v_stride_h;

    // Base 2 throughout: the probabilities come out of exp2, not exp.
    sm_scale *= kLog2e;

    // This lane's query and its Q row. A query past the end reads a clamped row;
    // its output is never stored and its scale is finite, so nothing downstream
    // sees it.
    const int q_idx = block_q + wave * kWarpQ9 + q_lane;
    const int8_t* q_row = q_head + static_cast<int64_t>(imin(q_idx, qo_len - 1)) * HD;
    const float q_lane_scale =
        q_scale[(static_cast<int64_t>(batch) * num_qo_heads + head) * qo_len_padded +
                imin(q_idx, qo_len_padded - 1)];

    float o_val[kOutTiles][4];
#pragma unroll
    for (int dt = 0; dt < kOutTiles; ++dt) {
#pragma unroll
        for (int e = 0; e < 4; ++e) o_val[dt][e] = 0.0f;
    }

    // Finite sentinels, not -inf: a fully masked row divides by one instead of
    // zero. Matches the RDNA body and the CUDA initial state.
    float m_run = kMaskedScore;
    float d_run = 1.0f;
    bool row_valid = !kCustomMask;

    const int num_iters = (kv_len + kCtaK9 - 1) / kCtaK9;

    // Taking the maximum off the INT32 accumulator assumes the dequantization
    // scale is positive. A negative softmax scale is legal and finite, so it
    // gives up the fast path rather than report the wrong maximum.
    int interior_iters = 0;
    if constexpr (!kCustomMask) {
        if (sm_scale >= 0.0f) interior_iters = kv_len / kCtaK9;
    }

    uint8_t* p_wave = smem_p + wave * (kWarpQ9 * kStrideP9);

    // `boundary` is a compile-time tag, so the interior instance carries no mask
    // test and no per-key bounds check at all.
    auto run_iter = [&](auto boundary_tag, int iter) {
        constexpr bool kBoundary = decltype(boundary_tag)::value;
        const int kb0 = iter * kCtaK9;

        __syncthreads();
        // K tile, key-major [key][d]. A key past the end is zeroed and its score
        // is masked below.
        for (int idx = tid; idx < kCtaK9 * kUnits; idx += kThreads9) {
            const int key = idx / kUnits;
            const int unit = idx % kUnits;
            const int gkey = kb0 + key;
            uint2 val = {0u, 0u};
            if (!kBoundary || gkey < kv_len) {
                val = *reinterpret_cast<const uint2*>(k_head +
                                                      static_cast<int64_t>(gkey) * HD + unit * 8);
            }
            *reinterpret_cast<uint2*>(smem_k + key * kStrideK + unit * 8) = val;
        }
        // V tile, channel-major [d][key] -- already so in global. The high bit of
        // every byte is flipped: the second MMA is unsigned on BOTH operands and
        // V is signed, so biasing it by 128 lets the u8 instruction read it, and
        // the rowsum correction below takes the bias back out exactly.
        for (int idx = tid; idx < HD * kUnitsV; idx += kThreads9) {
            const int d = idx / kUnitsV;
            const int unit = idx % kUnitsV;
            const int gkey = kb0 + unit * 8;
            uint2 val = {0x80808080u, 0x80808080u};
            if (!kBoundary || gkey < kv_len) {
                val = *reinterpret_cast<const uint2*>(v_head +
                                                      static_cast<int64_t>(d) * v_stride_d + gkey);
                val.x ^= 0x80808080u;
                val.y ^= 0x80808080u;
            }
            *reinterpret_cast<uint2*>(smem_v + d * kStrideV + unit * 8) = val;
        }
        __syncthreads();

        float key_scale[kScoreTiles9];
#pragma unroll
        for (int t = 0; t < kScoreTiles9; ++t) {
            key_scale[t] =
                k_scale[(static_cast<int64_t>(batch) * (num_qo_heads / num_kv_groups) + kv_head) *
                            k_groups_per_head +
                        iter * kScoreTiles9 + t];
        }

        // Hoisted: only the key term varies per element, and eight live 64-bit
        // addresses in the inner loop cost registers this kernel does not have.
        int64_t mask_row = 0;
        if constexpr (kCustomMask) {
            mask_row = static_cast<int64_t>(batch) * mask_stride_b +
                       static_cast<int64_t>(head) * mask_stride_h +
                       static_cast<int64_t>(imin(q_idx, qo_len - 1)) * mask_stride_q;
        }

        // --- score: two 16-key tiles, S = Q @ K^T ---
        // Tile t's accumulator element e of this lane is its query's key
        // 16*t + 4*e + g.
        typename MmaInt8::Acc s_acc[kScoreTiles9];
        float sc[kScoreTiles9][4];
        float m_lane = kMaskedScore;
#pragma unroll
        for (int t = 0; t < kScoreTiles9; ++t) {
            s_acc[t] = MmaInt8::zero();
#pragma unroll
            for (int dt = 0; dt < kDimTiles; ++dt) {
                s_acc[t] = MmaInt8::mma(
                    MmaInt8::load_aligned(q_row, 0, dt * 32, 0, lane),
                    load_frag_kt9(smem_k, t * 16 + q_lane, dt * 32, kStrideK, lane), s_acc[t]);
            }
            const float tile_scale_k = sm_scale * q_lane_scale * key_scale[t];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int gkey = kb0 + 16 * t + 4 * e + g;
                float score = static_cast<float>(s_acc[t][e]) * tile_scale_k;
                bool keep = !kBoundary || gkey < kv_len;
                if constexpr (kCustomMask) {
                    if (kBoundary) {
                        keep = keep && q_idx < qo_len;
                        if (keep) {
                            float bias;
                            keep = mask_keep(
                                mask, mask_row + static_cast<int64_t>(gkey) * mask_stride_k,
                                mask_dtype_code, bias);
                            score += bias * kLog2e;
                        }
                        row_valid = row_valid || keep;
                    }
                }
                sc[t][e] = keep ? score : kMaskedScore;
                m_lane = fmaxf(m_lane, sc[t][e]);
            }
        }
        // The four lanes of the group hold different keys of the same query, so
        // the iteration maximum has to be folded across them before anything
        // exponentiates.
        const float tile_m = query_group_max9(m_lane) - kProbU8Offset;

        // --- online softmax ---
        // The tile maximum's gap to the running one rides on the accumulator side
        // as p_scale, which underflows to zero for a fully masked tile.
        const float m_prev = m_run;
        const float smaller = exp2f(-fabsf(m_prev - tile_m));
        const float o_scale = m_prev < tile_m ? smaller : 1.0f;
        const float p_scale = tile_m < m_prev ? smaller : 1.0f;
        m_run = fmaxf(m_prev, tile_m);
        d_run *= o_scale;
#pragma unroll
        for (int dt = 0; dt < kOutTiles; ++dt) {
#pragma unroll
            for (int e = 0; e < 4; ++e) o_val[dt][e] *= o_scale;
        }

        // --- probabilities: exp2, then stage P for the second MMA ---
        // The shift by kProbU8Offset fills the unsigned INT8 range, and it cancels
        // between the numerator and the denominator, so both carry it.
        float tile_sum = 0.0f;
        float p_rowsum = 0.0f;
        const float neg_m = -tile_m;
#pragma unroll
        for (int t = 0; t < kScoreTiles9; ++t) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float p = exp2f(sc[t][e] + neg_m);
                tile_sum += p;
                const uint32_t q8 = prob_to_u8(p);
                p_rowsum += static_cast<float>(q8);
                p_wave[q_lane * kStrideP9 + 16 * t + 4 * e + g] = static_cast<uint8_t>(q8);
            }
        }
        d_run += query_group_sum9(tile_sum) * p_scale;
        // The bias the V staging added, in the accumulator's own units: 128 times
        // this query's quantized probabilities summed over all 32 keys.
        const float v_bias = 128.0f * query_group_sum9(p_rowsum);

        // Every lane of the group wrote P at an offset another lane reads, so the
        // wave has to be ordered before the second MMA.
        __syncthreads();

        // --- output: O += P @ V, one MMA per 16-channel tile (K = 32) ---
#pragma unroll
        for (int dt = 0; dt < kOutTiles; ++dt) {
            typename MmaInt8::Acc partial = MmaInt8::mma_ub(
                load_frag_b64_a(p_wave, q_lane, 0, kStrideP9, lane),
                load_frag_b64_a(smem_v, dt * 16 + q_lane, 0, kStrideV, lane),
                MmaInt8::zero());
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                o_val[dt][e] =
                    fmaf(static_cast<float>(partial[e]) - v_bias, p_scale, o_val[dt][e]);
            }
        }
    };

    int iter = 0;
    for (; iter < interior_iters; ++iter) run_iter(std::false_type{}, iter);
    for (; iter < num_iters; ++iter) run_iter(std::true_type{}, iter);

    // d_run was already folded across the query group on every iteration -- the
    // four lanes hold different keys of the same query -- so it needs no final
    // pass. row_valid was not: each lane only saw its own keys.
    if constexpr (kCustomMask) {
        row_valid = query_group_any9(row_valid ? 1 : 0) != 0;
    }

    const float* v_scale_head =
        v_scale + (static_cast<int64_t>(batch) * (num_qo_heads / num_kv_groups) + kv_head) * HD;

    if (q_idx >= qo_len) return;
    const float inv_d = row_valid ? 1.0f / d_run : 0.0f;
    OutT* row = o + static_cast<int64_t>(batch) * o_stride_b +
                static_cast<int64_t>(head) * o_stride_h +
                static_cast<int64_t>(q_idx) * o_stride_n;
#pragma unroll
    for (int dt = 0; dt < kOutTiles; ++dt) {
        float vals[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int d = dt * 16 + 4 * e + g;
            vals[e] = o_val[dt][e] * v_scale_head[d] * inv_d;
        }
        store_o_tile9<OutT>(row, dt * 16, vals, lane);
    }
}

// CTA_K is the caller's requested key tile. The RDNA body honors it; this one
// cannot, because its key tile is the i8 MMA's K of 32 and one iteration has to
// be exactly one P fragment. It is accepted so the shared launcher macro has one
// spelling, and ignored: the kernel above is what fixes the tile.
template <int HD, int CTA_K, MaskMode kMask, typename OutT>
void launch_one(const int8_t* q, const int8_t* k, const int8_t* v, OutT* o, const float* q_scale,
                const float* k_scale, const float* v_scale, const void* mask,
                int64_t mask_stride_b, int64_t mask_stride_h, int64_t mask_stride_q,
                int64_t mask_stride_k, int mask_dtype_code, int batch, int qo_len, int kv_len,
                int qo_len_padded, int num_qo_heads, int k_groups_per_head, int num_kv_groups,
                int64_t q_stride_b, int64_t q_stride_h, int64_t k_stride_b, int64_t k_stride_h,
                int64_t v_stride_b, int64_t v_stride_h, int64_t v_stride_d, int64_t o_stride_b,
                int64_t o_stride_h, int64_t o_stride_n, float sm_scale, hipStream_t stream) {

    const dim3 grid(div_ceil_host(qo_len, kCtaQ9), num_qo_heads, batch);
    int8_attn_kernel<HD, CTA_K, kMask, OutT><<<grid, kThreads9, 0, stream>>>(
        q, k, v, o, q_scale, k_scale, v_scale, mask, mask_stride_b, mask_stride_h, mask_stride_q,
        mask_stride_k, mask_dtype_code, qo_len, kv_len, qo_len_padded, k_groups_per_head,
        num_kv_groups, q_stride_b, q_stride_h, k_stride_b, k_stride_h, v_stride_b, v_stride_h,
        v_stride_d, o_stride_b, o_stride_h, o_stride_n, sm_scale);
}

#endif  // COMFY_MMA_GFX9

}  // namespace
}  // namespace comfy::hip_backend::sage

using namespace comfy::hip_backend::sage;

extern "C" void launch_sage_int8_attn(
    const void* q, const void* k, const void* v, void* o, const void* q_scale,
    const void* k_scale, const void* v_scale, const void* mask, int64_t mask_stride_b,
    int64_t mask_stride_h, int64_t mask_stride_q, int64_t mask_stride_k, int mask_dtype_code,
    int cta_k, int batch, int qo_len, int kv_len, int qo_len_padded, int num_qo_heads,
    int num_kv_heads, int head_dim, int k_groups_per_head, int64_t q_stride_b, int64_t q_stride_h,
    int64_t k_stride_b, int64_t k_stride_h, int64_t v_stride_b, int64_t v_stride_h,
    int64_t v_stride_d, int64_t o_stride_b, int64_t o_stride_h, int64_t o_stride_n,
    float sm_scale, int output_dtype_code, hipStream_t stream) {

    // Q, K and V are read as 16-byte fragments and O is stored 8 elements at a
    // time; every offset the kernel forms is a multiple of those widths, so the
    // base addresses are the whole of the alignment requirement. The V quantizer
    // launcher checks the same thing.
    if (reinterpret_cast<uintptr_t>(q) % 16 != 0 || reinterpret_cast<uintptr_t>(k) % 16 != 0 ||
        reinterpret_cast<uintptr_t>(v) % 16 != 0 || reinterpret_cast<uintptr_t>(o) % 16 != 0) {
        throw std::runtime_error(
            "sage int8 attention: q, k, v and o base pointers must be 16-byte aligned");
    }

    const int num_kv_groups = num_qo_heads / num_kv_heads;
    const auto* qi = static_cast<const int8_t*>(q);
    const auto* ki = static_cast<const int8_t*>(k);
    const auto* vi = static_cast<const int8_t*>(v);
    const auto* qs = static_cast<const float*>(q_scale);
    const auto* ks = static_cast<const float*>(k_scale);
    const auto* vs = static_cast<const float*>(v_scale);

#define SAGE_ATTN_LAUNCH(HD, CK, MM, OUT)                                                        \
    launch_one<HD, CK, MM, OUT>(qi, ki, vi, static_cast<OUT*>(o), qs, ks, vs, mask,              \
                                mask_stride_b, mask_stride_h, mask_stride_q, mask_stride_k,      \
                                mask_dtype_code, batch, qo_len, kv_len, qo_len_padded,           \
                                num_qo_heads, k_groups_per_head, num_kv_groups, q_stride_b,      \
                                q_stride_h, k_stride_b, k_stride_h, v_stride_b, v_stride_h,      \
                                v_stride_d, o_stride_b, o_stride_h, o_stride_n, sm_scale, stream)

    // One key tile, 64 keys. The CUDA backend widens to 128 for long unmasked
    // keys; ported and measured, that is slower on RDNA, where V is staged
    // transposed so the wide tile doubles both LDS tiles and costs more occupancy
    // than it saves. It does not fit at D256 at all, wanting 67 KB against a
    // 64 KB workgroup limit. A caller may still ask for 128 and get the same
    // answer from the narrow tile: tile size is a schedule, not a shape.
    (void)cta_k;
#define SAGE_ATTN_CTA(HD, MM, OUT) SAGE_ATTN_LAUNCH(HD, 64, MM, OUT)

#define SAGE_ATTN_OUT(HD, MM)                                                                    \
    do {                                                                                         \
        if (output_dtype_code == 1) {                                                            \
            SAGE_ATTN_CTA(HD, MM, __half);                                                       \
        } else {                                                                                 \
            SAGE_ATTN_CTA(HD, MM, __bf16);                                                       \
        }                                                                                        \
    } while (0)

#define SAGE_ATTN_MASK(HD)                                                                       \
    do {                                                                                         \
        if (mask != nullptr) {                                                                   \
            SAGE_ATTN_OUT(HD, MaskMode::kCustom);                                                \
        } else {                                                                                 \
            SAGE_ATTN_OUT(HD, MaskMode::kNone);                                                  \
        }                                                                                        \
    } while (0)

    switch (head_dim) {
        case 64: SAGE_ATTN_MASK(64); break;
        case 128: SAGE_ATTN_MASK(128); break;
        case 256: SAGE_ATTN_MASK(256); break;
        default:
            throw std::runtime_error("sage int8 attention: unsupported head_dim " +
                                     std::to_string(head_dim));
    }

#undef SAGE_ATTN_MASK
#undef SAGE_ATTN_OUT
#undef SAGE_ATTN_CTA
#undef SAGE_ATTN_LAUNCH
}
