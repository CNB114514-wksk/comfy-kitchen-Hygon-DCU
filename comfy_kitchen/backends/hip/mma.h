// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// WMMA policies for RDNA3 (gfx11) and RDNA4 (gfx12), wave32.
//
// A policy holds the operand fragment type, the bytes of a K-row one MMA consumes
// (kStepBytes), the LDS read, and the MMA. The tile kernels stay byte-addressed;
// only the policy changes per architecture.
//
// Operand layout differs:
//   gfx12: K is split across the half-waves. A lane holds 8 bytes at byte offset
//          8 * (lane / 16) of its row. Every type is a v2i, which is what lets one
//          kernel serve fp8, iu8 and iu4.
//   gfx11: no K split. A lane holds the whole K-step of its row and both
//          half-waves hold the same data (rocWMMA calls it input duplication),
//          which reading at row = lane % 16 with no half-wave offset gives for
//          free. Fragment width is the K-step: 16B iu8, 8B iu4, 32B bf16 for fp8.
//
// Accumulators are v8f/v8i on both, column lane % 16, but the row differs:
// gfx12 gives each half-wave a contiguous block, D[e + 8 * (lane / 16)]; gfx11
// interleaves the halves, D[2 * e + (lane / 16)]. See acc_row. rocWMMA's "padded
// acc" gfx11 quirk applies to the 16-bit accumulators, not these.
//
// gfx11 has no fp8 WMMA, so MmaFp8/MmaBf8 widen to bf16 there. The widening is
// exact (e4m3 and e5m2 mantissas fit bf16's 7) and both paths accumulate in f32,
// so gfx11 fp8 matches gfx12 numerically, just slower.
#pragma once

#include <hip/hip_runtime.h>

#include <atomic>
#include <cstdint>

#include "architecture_config.h"
#include "fp8_utils.h"

namespace comfy::hip_backend {

typedef int int32_t_v2 __attribute__((ext_vector_type(2)));
typedef int int32_t_v4 __attribute__((ext_vector_type(4)));

typedef int32_t_v2 v2i;
typedef int32_t_v4 v4i;
typedef int v8i __attribute__((ext_vector_type(8)));
typedef float v4f __attribute__((ext_vector_type(4)));
typedef float v8f __attribute__((ext_vector_type(8)));
typedef __bf16 v8bf __attribute__((ext_vector_type(8)));
typedef __bf16 v16bf __attribute__((ext_vector_type(16)));
typedef __bf16 v4bf __attribute__((ext_vector_type(4)));
typedef _Float16 v4h __attribute__((ext_vector_type(4)));
typedef _Float16 v8h __attribute__((ext_vector_type(8)));
typedef _Float16 v16h __attribute__((ext_vector_type(16)));

// RDNA (gfx10/11/12) issues one MMA per wave32. The Hygon DCU (gfx9 family,
// gfx906..gfx938) is wave64 and issues v_mmac_*, so kWave differs per family and
// every lane-derived index in the kernels has to follow it.
#if defined(COMFY_MMA_GFX9)
constexpr int kWave = 64;
#else
constexpr int kWave = 32;
#endif

// The runtime wave size of the device a launch will run on.
//
// kWave above is a *device-pass* constant: __gfx906__/__gfx928__/... is defined
// only while compiling device code, so the host pass always falls through to the
// wave32 value -- even in a gfx928-only build. Every host-side launch
// configuration (block dimensions, the partners of a kernel's __launch_bounds__)
// has to match the wave the kernel will actually run on, and a fat binary can
// hold both wave32 and wave64 images, so the size comes from the device rather
// than from kWave. Cached per ordinal; the fallback only mis-sizes a launch the
// driver would reject, it cannot change a result.
inline int device_wave_size() {
    constexpr int kMaxDevices = 16;
    // A kernel can be launched from several host threads at once. Racing threads
    // write the same value and nothing is published through the cache, so relaxed.
    static std::atomic<int> cache[kMaxDevices] = {};
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess || dev < 0 || dev >= kMaxDevices) return 32;
    int n = cache[dev].load(std::memory_order_relaxed);
    if (n == 0) {
        if (hipDeviceGetAttribute(&n, hipDeviceAttributeWarpSize, dev) != hipSuccess || n <= 0) {
            n = 32;
        }
        cache[dev].store(n, std::memory_order_relaxed);
    }
    return n;
}

// architecture_config.h is generated from architectures.json. __gfx*__ is
// defined only in device passes; the host pass falls through to the stubs below.

// ---------------------------------------------------------------------------
// Fragment addressing, shared by both architectures.
// ---------------------------------------------------------------------------

// Sum a value across the wave. The first offset comes from kWave rather than a
// literal 16: at wave64 a hardcoded 16 would fold only half the lanes and say
// nothing about it.
template <typename T>
__forceinline__ __device__ T wave_reduce_sum(T v) {
    #pragma unroll
    for (int off = kWave / 2; off > 0; off >>= 1) v += __shfl_xor(v, off, kWave);
    return v;
}

// Four int8 products accumulated into a 32-bit sum, the operands packed one per
// byte. RDNA3/4 spell it v_dot4_i32_iu8 with both operands signed; RDNA2 has the
// older v_dot4c_i32_i8 under a different builtin. The Hygon DCU does have the
// instruction family, but clang gates __builtin_amdgcn_sudot4 behind the
// +dot8-insts feature, which the gfx9 targets do not enable here, so the gfx9
// path takes the arithmetic fallback too. The instruction count is the only
// difference; the sum is the same int32.
__forceinline__ __device__ int dot4_i8(int a, int b, int c) {
#if defined(COMFY_HAS_WMMA) && !defined(COMFY_MMA_GFX9)
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
#else
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        c += static_cast<int>(static_cast<int8_t>((a >> (i * 8)) & 0xFF)) *
             static_cast<int>(static_cast<int8_t>((b >> (i * 8)) & 0xFF));
    }
    return c;
#endif
}

__forceinline__ __device__ int frag_row(int lane) { return lane % 16; }

// Row of accumulator element `e` for this lane; the column is lane % 16. The two
// architectures lay the 16 result rows across the wave differently. gfx12 gives
// each half-wave a contiguous block of 8 rows. gfx11 interleaves them: element e
// is row 2e for the low half-wave and 2e + 1 for the high one, so consecutive
// elements are two rows apart. Both are 32-bit accumulators (v8f/v8i), so neither
// has the padded-acc read the 16-bit gfx11 accumulators need.
//
// The Hygon DCU (gfx9) transposes the two: a lane owns one row and four columns
// four apart, so the column is a function of `e` and there is no one-argument
// answer. gfx906..gfx938 register layout, measured on gfx928 (0/256 mismatch):
//   A lane l, byte p -> A[l % 16][8 * (l / 16) + p]
//   B lane l, byte p -> B[8 * (l / 16) + p][l % 16]
//   C lane l, elem i -> C[l % 16][4 * i + (l / 16)]
//
// Both gfx11/gfx12 are 8-element (v8) accumulators with a fixed column; gfx9 is a
// 4-element (v4) one whose column moves with `e`. Consumers that walk all the
// accumulator elements must therefore ask the policy for both the element count
// (Mma::kAccElems) and the row/column, not assume eight or a fixed column.
#if defined(COMFY_MMA_GFX9)
__forceinline__ __device__ int acc_row(int lane, int) { return lane % 16; }
__forceinline__ __device__ int acc_col(int lane, int e) { return 4 * e + (lane / 16); }
// Legacy single-argument form so kernels that have not been ported to the gfx9
// layout still compile. It is not the gfx9 column: do not read results with it.
__forceinline__ __device__ int acc_col(int lane) { return lane / 16; }
#elif defined(COMFY_MMA_GFX11)
__forceinline__ __device__ int acc_row(int lane, int e) { return 2 * e + (lane / 16); }
__forceinline__ __device__ int acc_col(int lane, int) { return lane % 16; }
__forceinline__ __device__ int acc_col(int lane) { return lane % 16; }
#else
__forceinline__ __device__ int acc_row(int lane, int e) { return e + 8 * (lane / 16); }
__forceinline__ __device__ int acc_col(int lane, int) { return lane % 16; }
__forceinline__ __device__ int acc_col(int lane) { return lane % 16; }
#endif

// How many distinct accumulator *rows* one lane holds, and which slot a given
// element belongs to. A row's softmax state (running max and denominator) lives
// in one slot, so a consumer that keeps per-row state has to fold the elements
// that share a slot together before reducing across lanes.
//
// On RDNA each of the eight accumulator elements is a different row, so the
// mapping is the identity. On gfx9 all four of a lane's elements are columns of
// the single row it owns (see acc_row/acc_col above), so they collapse to one.
#if defined(COMFY_MMA_GFX9)
constexpr int kAccRowsPerLane = 1;
__forceinline__ __device__ constexpr int acc_row_slot(int) { return 0; }
#else
constexpr int kAccRowsPerLane = 8;
__forceinline__ __device__ constexpr int acc_row_slot(int e) { return e; }
#endif

// Bytes of a row that live `stride` apart in LDS. `row` is the lane's row (A) or
// column (B); `kbyte` is the byte offset within that row.
__forceinline__ __device__ v2i load_frag_b64(const void* lds, int row, int kbyte, int stride) {
    const char* p = static_cast<const char*>(lds) + row * stride + kbyte;
    return *reinterpret_cast<const v2i*>(p);
}

__forceinline__ __device__ v4i load_frag_b128(const void* lds, int row, int kbyte, int stride) {
    const char* p = static_cast<const char*>(lds) + row * stride + kbyte;
    // kLdsPad breaks 16-byte alignment, so this must not become a single b128 load.
    v4i r;
    const int* q = reinterpret_cast<const int*>(p);
    r[0] = q[0];
    r[1] = q[1];
    r[2] = q[2];
    r[3] = q[3];
    return r;
}

// Same, for a row the caller guarantees is 16-byte aligned: one LDS instruction
// instead of four. gfx11 hands every lane the whole 16-byte K-step, so this is
// the difference between one ds_load_b128 and four ds_load_b32 on every single
// fragment read in an attention inner loop.
__forceinline__ __device__ v4i load_frag_b128_aligned(const void* lds, int row, int kbyte,
                                                      int stride) {
    const char* p = static_cast<const char*>(lds) + row * stride + kbyte;
    return *reinterpret_cast<const v4i*>(p);
}

// 16 fp8 bytes of a row -> the 16 bf16 lanes of one gfx11 K-step.
template <bool E5M2>
__forceinline__ __device__ v16bf load_frag_fp8_as_bf16(const void* lds, int row, int kbyte,
                                                       int stride) {
    const uint8_t* p = static_cast<const uint8_t*>(lds) + row * stride + kbyte;
    v16bf r;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        r[i] = static_cast<__bf16>(E5M2 ? bf8_to_float(p[i]) : fp8_to_float(p[i]));
    }
    return r;
}

// A K-step of a 16-bit row, straight out of whatever address space `row` points
// at. The half-wave slice is the policy's, since gfx11 hands every lane the whole
// K-step and gfx12 splits it. The elements are contiguous, so this folds into
// vector loads.
template <typename Mma>
__forceinline__ __device__ typename Mma::Frag load_frag_16bit(const typename Mma::Elem* row,
                                                              int lane) {
    typename Mma::Frag f;
    const typename Mma::Elem* p = row + Mma::frag_base(lane);
    #pragma unroll
    for (int i = 0; i < Mma::kFragElems; ++i) {
        f[i] = p[i];
    }
    return f;
}

// ---------------------------------------------------------------------------
// Policies
// ---------------------------------------------------------------------------

#if defined(COMFY_MMA_GFX12)

struct MmaFp8 {
    using Acc = v8f;
    using Frag = v2i;
    static constexpr int kStepBytes = 16;
    static constexpr int kAccElems = 8;
    static __forceinline__ __device__ Frag load(const void* lds, int row, int kbyte, int stride,
                                                int lane) {
        return load_frag_b64(lds, row, kbyte + 8 * (lane / 16), stride);
    }
    static __forceinline__ __device__ Acc zero() { return Acc{0, 0, 0, 0, 0, 0, 0, 0}; }
    static __forceinline__ __device__ Acc mma(Frag a, Frag b, Acc c) {
        return __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c);
    }
    static __forceinline__ __device__ float get(Acc c, int e) { return c[e]; }
};

struct MmaBf8 {
    using Acc = v8f;
    using Frag = v2i;
    static constexpr int kStepBytes = 16;
    static constexpr int kAccElems = 8;
    static __forceinline__ __device__ Frag load(const void* lds, int row, int kbyte, int stride,
                                                int lane) {
        return load_frag_b64(lds, row, kbyte + 8 * (lane / 16), stride);
    }
    static __forceinline__ __device__ Acc zero() { return Acc{0, 0, 0, 0, 0, 0, 0, 0}; }
    static __forceinline__ __device__ Acc mma(Frag a, Frag b, Acc c) {
        return __builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_w32_gfx12(a, b, c);
    }
    static __forceinline__ __device__ float get(Acc c, int e) { return c[e]; }
};

struct MmaInt8 {
    using Acc = v8i;
    using Frag = v2i;
    static constexpr int kStepBytes = 16;
    static constexpr int kAccElems = 8;
    static __forceinline__ __device__ Frag load(const void* lds, int row, int kbyte, int stride,
                                                int lane) {
        return load_frag_b64(lds, row, kbyte + 8 * (lane / 16), stride);
    }
    static __forceinline__ __device__ Acc zero() { return Acc{0, 0, 0, 0, 0, 0, 0, 0}; }
    static __forceinline__ __device__ Frag load_aligned(const void* lds, int row, int kbyte,
                                                        int stride, int lane) {
        return load_frag_b64(lds, row, kbyte + 8 * (lane / 16), stride);
    }
    static __forceinline__ __device__ Acc mma(Frag a, Frag b, Acc c) {
        return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, b, c, false);
    }
    // Signedness is an instruction immediate, so an unsigned A needs its own entry.
    static __forceinline__ __device__ Acc mma_ua(Frag a, Frag b, Acc c) {
        return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a, true, b, c, false);
    }
    // ... and so does an unsigned B, which is the P operand of int8 attention.
    static __forceinline__ __device__ Acc mma_ub(Frag a, Frag b, Acc c) {
        return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, false, b, c, false);
    }
    static __forceinline__ __device__ float get(Acc c, int e) { return static_cast<float>(c[e]); }
};

// The one type whose K-step differs: gfx12 does K=32 per MMA, gfx11 does K=16.
struct MmaInt4 {
    using Acc = v8i;
    using Frag = v2i;
    static constexpr int kStepBytes = 16;
    static constexpr int kAccElems = 8;
    static __forceinline__ __device__ Frag load(const void* lds, int row, int kbyte, int stride,
                                                int lane) {
        return load_frag_b64(lds, row, kbyte + 8 * (lane / 16), stride);
    }
    // The iu4 instruction takes A's signedness as an immediate, so the fragment
    // bytes are the same either way; this exists only so the callers can name it
    // on every policy. See the gfx9 policy for where it does differ.
    static __forceinline__ __device__ Frag load_unsigned(const void* lds, int row, int kbyte,
                                                         int stride, int lane) {
        return load(lds, row, kbyte, stride, lane);
    }
    static __forceinline__ __device__ Acc zero() { return Acc{0, 0, 0, 0, 0, 0, 0, 0}; }
    static __forceinline__ __device__ Acc mma(Frag a, Frag b, Acc c) {
        return __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true, a, true, b, c, false);
    }
    static __forceinline__ __device__ Acc mma_ua(Frag a, Frag b, Acc c) {
        return __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(false, a, true, b, c, false);
    }
    static __forceinline__ __device__ float get(Acc c, int e) { return static_cast<float>(c[e]); }
};

// 16-bit float MMA, used by the attention kernels rather than the GEMM core: they
// build fragments from global memory and from LDS scratch, so the policy exposes
// the element type and the half-wave's K slice instead of a byte-addressed load.
struct MmaBf16 {
    using Acc = v8f;
    using Frag = v8bf;
    using Elem = __bf16;
    static constexpr int kFragElems = 8;
    static constexpr int kAccElems = 8;
    static __forceinline__ __device__ int frag_base(int lane) { return 8 * (lane / 16); }
    static __forceinline__ __device__ Acc zero() { return Acc{0, 0, 0, 0, 0, 0, 0, 0}; }
    static __forceinline__ __device__ Acc mma(Frag a, Frag b, Acc c) {
        return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
    }
    static __forceinline__ __device__ float get(Acc c, int e) { return c[e]; }
};

struct MmaF16 {
    using Acc = v8f;
    using Frag = v8h;
    using Elem = _Float16;
    static constexpr int kFragElems = 8;
    static constexpr int kAccElems = 8;
    // The GEMM core's byte-addressed view: 16 fp16 elements are 32 bytes.
    static constexpr int kStepBytes = 32;
    static __forceinline__ __device__ int frag_base(int lane) { return 8 * (lane / 16); }
    static __forceinline__ __device__ Frag load(const void* lds, int row, int kbyte, int stride,
                                                int lane) {
        return load_frag_16bit<MmaF16>(
            reinterpret_cast<const _Float16*>(static_cast<const char*>(lds) + row * stride + kbyte),
            lane);
    }
    static __forceinline__ __device__ Acc zero() { return Acc{0, 0, 0, 0, 0, 0, 0, 0}; }
    static __forceinline__ __device__ Acc mma(Frag a, Frag b, Acc c) {
        return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
    }
    static __forceinline__ __device__ float get(Acc c, int e) { return c[e]; }
};

#elif defined(COMFY_MMA_GFX11)

struct MmaFp8 {
    using Acc = v8f;
    using Frag = v16bf;
    static constexpr int kStepBytes = 16;  // 16 fp8 bytes == K-step of 16
    static constexpr int kAccElems = 8;
    static __forceinline__ __device__ Frag load(const void* lds, int row, int kbyte, int stride,
                                                int lane) {
        return load_frag_fp8_as_bf16<false>(lds, row, kbyte, stride);
    }
    static __forceinline__ __device__ Acc zero() { return Acc{0, 0, 0, 0, 0, 0, 0, 0}; }
    static __forceinline__ __device__ Acc mma(Frag a, Frag b, Acc c) {
        return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
    }
    static __forceinline__ __device__ float get(Acc c, int e) { return c[e]; }
};

struct MmaBf8 {
    using Acc = v8f;
    using Frag = v16bf;
    static constexpr int kStepBytes = 16;
    static constexpr int kAccElems = 8;
    static __forceinline__ __device__ Frag load(const void* lds, int row, int kbyte, int stride,
                                                int lane) {
        return load_frag_fp8_as_bf16<true>(lds, row, kbyte, stride);
    }
    static __forceinline__ __device__ Acc zero() { return Acc{0, 0, 0, 0, 0, 0, 0, 0}; }
    static __forceinline__ __device__ Acc mma(Frag a, Frag b, Acc c) {
        return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
    }
    static __forceinline__ __device__ float get(Acc c, int e) { return c[e]; }
};

struct MmaInt8 {
    using Acc = v8i;
    using Frag = v4i;  // whole K-step of 16 int8, duplicated across half-waves
    static constexpr int kStepBytes = 16;
    static constexpr int kAccElems = 8;
    static __forceinline__ __device__ Frag load(const void* lds, int row, int kbyte, int stride,
                                                int lane) {
        return load_frag_b128(lds, row, kbyte, stride);
    }
    static __forceinline__ __device__ Acc zero() { return Acc{0, 0, 0, 0, 0, 0, 0, 0}; }
    static __forceinline__ __device__ Frag load_aligned(const void* lds, int row, int kbyte,
                                                        int stride, int) {
        return load_frag_b128_aligned(lds, row, kbyte, stride);
    }
    static __forceinline__ __device__ Acc mma(Frag a, Frag b, Acc c) {
        return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c, false);
    }
    static __forceinline__ __device__ Acc mma_ua(Frag a, Frag b, Acc c) {
        return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false, a, true, b, c, false);
    }
    static __forceinline__ __device__ Acc mma_ub(Frag a, Frag b, Acc c) {
        return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, false, b, c, false);
    }
    static __forceinline__ __device__ float get(Acc c, int e) { return static_cast<float>(c[e]); }
};

struct MmaInt4 {
    using Acc = v8i;
    using Frag = v2i;  // K=16 nibbles == 8 bytes, whole step, duplicated
    static constexpr int kStepBytes = 8;
    static constexpr int kAccElems = 8;
    static __forceinline__ __device__ Frag load(const void* lds, int row, int kbyte, int stride,
                                                int lane) {
        return load_frag_b64(lds, row, kbyte, stride);
    }
    // Same fragment bytes: the iu4 instruction carries A's signedness as an
    // immediate, so nothing changes here. See the gfx9 policy.
    static __forceinline__ __device__ Frag load_unsigned(const void* lds, int row, int kbyte,
                                                         int stride, int lane) {
        return load(lds, row, kbyte, stride, lane);
    }
    static __forceinline__ __device__ Acc zero() { return Acc{0, 0, 0, 0, 0, 0, 0, 0}; }
    static __forceinline__ __device__ Acc mma(Frag a, Frag b, Acc c) {
        return __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a, true, b, c, false);
    }
    static __forceinline__ __device__ Acc mma_ua(Frag a, Frag b, Acc c) {
        return __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(false, a, true, b, c, false);
    }
    static __forceinline__ __device__ float get(Acc c, int e) { return static_cast<float>(c[e]); }
};


// gfx11 hands every lane the whole 16-wide K-step, duplicated across half-waves.
struct MmaBf16 {
    using Acc = v8f;
    using Frag = v16bf;
    using Elem = __bf16;
    static constexpr int kFragElems = 16;
    static constexpr int kAccElems = 8;
    static __forceinline__ __device__ int frag_base(int) { return 0; }
    static __forceinline__ __device__ Acc zero() { return Acc{0, 0, 0, 0, 0, 0, 0, 0}; }
    static __forceinline__ __device__ Acc mma(Frag a, Frag b, Acc c) {
        return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
    }
    static __forceinline__ __device__ float get(Acc c, int e) { return c[e]; }
};

struct MmaF16 {
    using Acc = v8f;
    using Frag = v16h;
    using Elem = _Float16;
    static constexpr int kFragElems = 16;
    static constexpr int kAccElems = 8;
    static constexpr int kStepBytes = 32;
    static __forceinline__ __device__ int frag_base(int) { return 0; }
    static __forceinline__ __device__ Frag load(const void* lds, int row, int kbyte, int stride,
                                                int lane) {
        return load_frag_16bit<MmaF16>(
            reinterpret_cast<const _Float16*>(static_cast<const char*>(lds) + row * stride + kbyte),
            lane);
    }
    static __forceinline__ __device__ Acc zero() { return Acc{0, 0, 0, 0, 0, 0, 0, 0}; }
    static __forceinline__ __device__ Acc mma(Frag a, Frag b, Acc c) {
        return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
    }
    static __forceinline__ __device__ float get(Acc c, int e) { return c[e]; }
};

#elif defined(COMFY_MMA_GFX9)

// ---------------------------------------------------------------------------
// Hygon DCU matrix core, gfx906..gfx938, wave64.
//
// The instruction is Hygon's v_mmac_*, NOT AMD's v_mfma_*. The latter is what
// clang emits for +mai-insts, but gfx928 hardware traps it as an illegal
// instruction (measured: all five MFMA shapes VMFault); v_mmac_* runs, needs no
// target feature, and is richer -- bf16, i8 with K=32 and i4 with K=64.
//
// Register layouts, measured on gfx928 with /tmp/hiprun/mmac_layout.hip
// (0/256 mismatch against a CPU reference):
//   A lane l, byte p -> A[l % 16][8 * (l / 16) + p]     (i8; 4 elements for bf16/f16)
//   B lane l, byte p -> B[8 * (l / 16) + p][l % 16]
//   C lane l, elem i -> C[l % 16][4 * i + (l / 16)]
//
// A is read straight out of a natural [row][K] tile: the eight bytes a lane owns
// are contiguous in K, so they are one 8-byte load at the lane's row.
//
// The GEMM stages B the same natural way, as [N][K] rows of K. The hardware B
// operand is [K][N], so lane l's byte p maps to Bs[l % 16][8 * (l / 16) + p] --
// again eight contiguous bytes of one row, i.e. the same loader as A. (A caller
// that instead stored B transposed as [K][N] would need the strided gather this
// used to carry; no caller does.)
//
// gfx9 has no fp8 matrix instruction, so MmaFp8/MmaBf8 widen to bf16 exactly as
// the gfx11 policy does.

// One 8-byte fragment: 8 consecutive bytes of this lane's row, starting at the
// lane's K slice. Used for both A and the naturally-staged B.
__forceinline__ __device__ long load_frag_b64_a(const void* lds, int row, int kbyte, int stride,
                                                int lane) {
    const char* p = static_cast<const char*>(lds) + row * stride + kbyte + 8 * (lane / 16);
    const int* q = reinterpret_cast<const int*>(p);
    // kLdsPad can break 8-byte alignment, so this stays two 32-bit loads.
    const uint64_t lo = static_cast<uint32_t>(q[0]);
    const uint64_t hi = static_cast<uint32_t>(q[1]);
    return static_cast<long>(lo | (hi << 32));
}

// Four fp8 bytes of a row -> the four bf16 of one gfx9 K-step slice. gfx9 has no
// fp8 matrix instruction, so fp8 runs as bf16 exactly as the gfx11 policy does;
// the widening is exact and both accumulate in f32.
template <bool E5M2>
__forceinline__ __device__ v4bf load_frag_fp8x4_as_bf16(const void* lds, int row, int kbyte,
                                                        int stride, int lane) {
    const uint8_t* p =
        static_cast<const uint8_t*>(lds) + row * stride + kbyte + 4 * (lane / 16);
    v4bf r;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        r[i] = static_cast<__bf16>(E5M2 ? bf8_to_float(p[i]) : fp8_to_float(p[i]));
    }
    return r;
}

struct MmaFp8 {
    using Acc = v4f;
    using Frag = v4bf;
    static constexpr int kStepBytes = 16;  // 16 fp8 bytes == K-step of 16
    static constexpr int kAccElems = 4;
    static __forceinline__ __device__ Frag load(const void* lds, int row, int kbyte, int stride,
                                                int lane) {
        return load_frag_fp8x4_as_bf16<false>(lds, row, kbyte, stride, lane);
    }
    static __forceinline__ __device__ Acc zero() { return Acc{0, 0, 0, 0}; }
    static __forceinline__ __device__ Acc mma(Frag a, Frag b, Acc c) {
        return __builtin_amdgcn_mmac_f32_16x16x16bf16(a, b, c);
    }
    static __forceinline__ __device__ float get(Acc c, int e) { return c[e]; }
};

struct MmaBf8 : MmaFp8 {};

struct MmaInt8 {
    using Acc = v4i;
    using Frag = long;  // 8 int8 of the K-step of 32, this lane's slice
    static constexpr int kStepBytes = 32;  // K=32 per MMA
    static constexpr int kAccElems = 4;
    static __forceinline__ __device__ Frag load(const void* lds, int row, int kbyte, int stride,
                                                int lane) {
        return load_frag_b64_a(lds, row, kbyte, stride, lane);
    }
    static __forceinline__ __device__ Frag load_aligned(const void* lds, int row, int kbyte,
                                                        int stride, int lane) {
        return load_frag_b64_a(lds, row, kbyte, stride, lane);
    }
    static __forceinline__ __device__ Acc zero() { return Acc{0, 0, 0, 0}; }
    static __forceinline__ __device__ Acc mma(Frag a, Frag b, Acc c) {
        return __builtin_amdgcn_mmac_i32_16x16x32i8(a, b, c);
    }
    // Hygon exposes whole-operand signedness (i8/u8), not per-side. The attention
    // kernel's unsigned P operand is the one that needs mma_ub; it maps to u8.
    static __forceinline__ __device__ Acc mma_ua(Frag a, Frag b, Acc c) {
        return __builtin_amdgcn_mmac_i32_16x16x32u8(a, b, c);
    }
    static __forceinline__ __device__ Acc mma_ub(Frag a, Frag b, Acc c) {
        return __builtin_amdgcn_mmac_i32_16x16x32u8(a, b, c);
    }
    static __forceinline__ __device__ float get(Acc c, int e) { return static_cast<float>(c[e]); }
};

// Hygon's int4 mmac is not usable: clang emits v_mmac_i32_16x16x64_i4 behind
// +mmop3-insts, but the gfx928 silicon traps it as an illegal instruction
// (measured; the u4 form and every feature combination trap too). Int4 therefore
// runs as int8: each nibble is unpacked to a signed int8 in the fragment loader
// and fed to the i8 MMA.
//
// One i4 MMA would consume K=64 nibbles; two i8 MMAs consume 32 each. kStepBytes
// is therefore 16 packed bytes, not 32 -- the same K-per-MMA the gfx12 iu4 policy
// has, so every caller's group arithmetic (kSvdGroupBytes / kStepBytes) is
// unchanged. The instruction count doubles; the accumulated values are exact
// either way, since both paths accumulate the same int32 dot product.
__forceinline__ __device__ long load_frag_i4(const void* lds, int row, int kbyte, int stride,
                                             int lane, bool sign_extend) {
    // Four packed bytes hold this lane's eight nibbles: byte 4*(lane/16)+b carries
    // nibbles 8*(lane/16)+2b and +1, low nibble of the byte first.
    const uint8_t* p = static_cast<const uint8_t*>(lds) + row * stride + kbyte + 4 * (lane / 16);
    uint64_t out = 0;
#pragma unroll
    for (int b = 0; b < 4; ++b) {
        const int byte = p[b];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int nib = h ? ((byte >> 4) & 0xF) : (byte & 0xF);
            const int val = (sign_extend && nib > 7) ? nib - 16 : nib;
            out |= static_cast<uint64_t>(static_cast<uint8_t>(static_cast<int8_t>(val)))
                   << (8 * (2 * b + h));
        }
    }
    return static_cast<long>(out);
}

struct MmaInt4 {
    using Acc = v4i;
    using Frag = long;  // 8 int8, this lane's slice of a K=32 step
    // 16 packed int4 bytes == 32 nibbles == the i8 MMA's K. See above.
    static constexpr int kStepBytes = 16;
    static constexpr int kAccElems = 4;
    static __forceinline__ __device__ Frag load(const void* lds, int row, int kbyte, int stride,
                                                int lane) {
        return load_frag_i4(lds, row, kbyte, stride, lane, /*sign_extend=*/true);
    }
    // The A operand when the activations are unsigned int4 (0..15). The i8 MMA
    // reads A as signed, but 0..15 is positive in both readings, so leaving the
    // nibbles unextended is exactly an unsigned A.
    static __forceinline__ __device__ Frag load_unsigned(const void* lds, int row, int kbyte,
                                                         int stride, int lane) {
        return load_frag_i4(lds, row, kbyte, stride, lane, /*sign_extend=*/false);
    }
    static __forceinline__ __device__ Acc zero() { return Acc{0, 0, 0, 0}; }
    static __forceinline__ __device__ Acc mma(Frag a, Frag b, Acc c) {
        return __builtin_amdgcn_mmac_i32_16x16x32i8(a, b, c);
    }
    // Same instruction: the unsigned A already came out of load_unsigned, and B
    // (the weight) is signed in every caller.
    static __forceinline__ __device__ Acc mma_ua(Frag a, Frag b, Acc c) {
        return __builtin_amdgcn_mmac_i32_16x16x32i8(a, b, c);
    }
    static __forceinline__ __device__ float get(Acc c, int e) { return static_cast<float>(c[e]); }
};

// 16-bit operands. A lane holds four contiguous elements at K offset
// 4 * (lane / 16); B's four are a row apart, same as the int8 case.
struct MmaBf16 {
    using Acc = v4f;
    using Frag = v4bf;
    using Elem = __bf16;
    static constexpr int kFragElems = 4;
    static constexpr int kAccElems = 4;
    static __forceinline__ __device__ int frag_base(int lane) { return 4 * (lane / 16); }
    static __forceinline__ __device__ Acc zero() { return Acc{0, 0, 0, 0}; }
    static __forceinline__ __device__ Acc mma(Frag a, Frag b, Acc c) {
        return __builtin_amdgcn_mmac_f32_16x16x16bf16(a, b, c);
    }
    static __forceinline__ __device__ float get(Acc c, int e) { return c[e]; }
};

struct MmaF16 {
    using Acc = v4f;
    using Frag = v4h;
    using Elem = _Float16;
    static constexpr int kFragElems = 4;
    static constexpr int kAccElems = 4;
    // 16 fp16 elements are one K-step of the byte-addressed GEMM core.
    static constexpr int kStepBytes = 32;
    static __forceinline__ __device__ int frag_base(int lane) { return 4 * (lane / 16); }
    // The fp16 gemm_wmma path is not ported to the DCU: no kernel here uses this
    // loader, and gemm_fp16/conv3d_fp16 are gated off on gfx9. It exists only so
    // the translation units that instantiate launch_gemm_wmma<MmaF16> still
    // compile. Trap rather than return a plausible wrong fragment, the way the
    // no-matrix-core stub does. na3d's fp16 path uses load_frag_16bit directly and
    // is unaffected.
    static __forceinline__ __device__ Frag load(const void*, int, int, int, int) {
        __builtin_trap();
        return Frag{};
    }
    static __forceinline__ __device__ Acc zero() { return Acc{0, 0, 0, 0}; }
    static __forceinline__ __device__ Acc mma(Frag a, Frag b, Acc c) {
        return __builtin_amdgcn_mmac_f32_16x16x16f16(a, b, c);
    }
    static __forceinline__ __device__ float get(Acc c, int e) { return c[e]; }
};

#else

// No matrix cores (RDNA2 and older) and the host pass. The GEMM kernels are
// instantiated in every device pass so they must still compile, but the backend
// reports no WMMA capability there and never launches them. Trap rather than
// return a plausible wrong answer.
#define COMFY_MMA_STUB_BODY \
    __builtin_trap();       \
    return c;

struct MmaFp8 {
    using Acc = v8f;
    using Frag = v2i;
    static constexpr int kStepBytes = 16;
    static constexpr int kAccElems = 8;
    static __forceinline__ __device__ Frag load(const void*, int, int, int, int) { return Frag{}; }
    static __forceinline__ __device__ Acc zero() { return Acc{0, 0, 0, 0, 0, 0, 0, 0}; }
    static __forceinline__ __device__ Acc mma(Frag, Frag, Acc c) { COMFY_MMA_STUB_BODY }
    static __forceinline__ __device__ float get(Acc c, int e) { return c[e]; }
};

struct MmaBf8 : MmaFp8 {};

struct MmaInt8 {
    using Acc = v8i;
    using Frag = v2i;
    static constexpr int kStepBytes = 16;
    static constexpr int kAccElems = 8;
    static __forceinline__ __device__ Frag load(const void*, int, int, int, int) { return Frag{}; }
    static __forceinline__ __device__ Acc zero() { return Acc{0, 0, 0, 0, 0, 0, 0, 0}; }
    static __forceinline__ __device__ Frag load_aligned(const void*, int, int, int, int) {
        return Frag{};
    }
    static __forceinline__ __device__ Acc mma(Frag, Frag, Acc c) { COMFY_MMA_STUB_BODY }
    static __forceinline__ __device__ Acc mma_ua(Frag, Frag, Acc c) { COMFY_MMA_STUB_BODY }
    static __forceinline__ __device__ Acc mma_ub(Frag, Frag, Acc c) { COMFY_MMA_STUB_BODY }
    static __forceinline__ __device__ float get(Acc c, int e) { return static_cast<float>(c[e]); }
};

struct MmaInt4 : MmaInt8 {
    static __forceinline__ __device__ Frag load_unsigned(const void*, int, int, int, int) {
        return Frag{};
    }
};

struct MmaBf16 {
    using Acc = v8f;
    using Frag = v8bf;
    using Elem = __bf16;
    static constexpr int kFragElems = 8;
    static constexpr int kAccElems = 8;
    static __forceinline__ __device__ int frag_base(int) { return 0; }
    static __forceinline__ __device__ Acc zero() { return Acc{0, 0, 0, 0, 0, 0, 0, 0}; }
    static __forceinline__ __device__ Acc mma(Frag, Frag, Acc c) { COMFY_MMA_STUB_BODY }
    static __forceinline__ __device__ float get(Acc c, int e) { return c[e]; }
};

struct MmaF16 {
    using Acc = v8f;
    using Frag = v8h;
    using Elem = _Float16;
    static constexpr int kFragElems = 8;
    static constexpr int kAccElems = 8;
    static constexpr int kStepBytes = 32;
    static __forceinline__ __device__ int frag_base(int) { return 0; }
    static __forceinline__ __device__ Frag load(const void*, int, int, int, int) { return Frag{}; }
    static __forceinline__ __device__ Acc zero() { return Acc{0, 0, 0, 0, 0, 0, 0, 0}; }
    static __forceinline__ __device__ Acc mma(Frag, Frag, Acc c) { COMFY_MMA_STUB_BODY }
    static __forceinline__ __device__ float get(Acc c, int e) { return c[e]; }
};

#undef COMFY_MMA_STUB_BODY

#endif

}  // namespace comfy::hip_backend
