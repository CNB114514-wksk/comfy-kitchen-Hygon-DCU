// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Fused V to signed INT8 with a per-channel scale, transposed to
// [B * H * D, padded_N] so the attention kernel reads a V fragment as a
// contiguous run of keys along one channel row.
//
// The CUDA backend permutes each group of 16 keys here to match what ldmatrix
// hands its MMA. RDNA needs no permutation: the probability fragment and the V
// fragment both index keys in natural order (see int8_attn.h), so keys are
// written where they belong and the tail is zero filled.
//
// One block per (b, h, d_tile) of kDTile channels. Pass 1 reduces the per-channel
// absmax along N, pass 2 quantizes. Both walk four consecutive keys per thread,
// which is what makes a lane cover whole 32-byte sectors: one key per thread read
// 16 bytes of a sector and wrote one byte per lane, so the quantize pass settled
// 32 bytes per store instruction and the store pipe, not the memory system, set
// the pace. Four keys per thread also let each channel leave as a single 4-byte
// store.
#include <hip/hip_runtime.h>

#include <stdexcept>
#include <string>

#include "sage_common.h"

namespace comfy::hip_backend::sage {
namespace {

// Sixteen channels: 64 bytes of fp32 or 32 of a 16-bit type. Eight channels left
// the 16-bit types loading 16 bytes, half a sector, so every load pass fetched
// twice what it used.
constexpr int kDTile = 16;
// Four keys per thread, so a channel's four quantized bytes leave together.
constexpr int kKeysPerThread = 4;

// kDTile adjacent channels: 64 bytes for fp32, 32 for the 16-bit types. The
// caller guarantees the group is in bounds and 16-byte aligned.
template <typename T>
__forceinline__ __device__ void load16(const T* p, float* out) {
    if constexpr (std::is_same_v<T, float>) {
        const float4* raw = reinterpret_cast<const float4*>(p);
#pragma unroll
        for (int i = 0; i < kDTile / 4; ++i) {
            const float4 part = raw[i];
            out[i * 4 + 0] = part.x;
            out[i * 4 + 1] = part.y;
            out[i * 4 + 2] = part.z;
            out[i * 4 + 3] = part.w;
        }
    } else {
        const uint4* raw = reinterpret_cast<const uint4*>(p);
#pragma unroll
        for (int i = 0; i < kDTile / 8; ++i) {
            const uint4 part = raw[i];
            const T* vals = reinterpret_cast<const T*>(&part);
#pragma unroll
            for (int j = 0; j < 8; ++j) out[i * 8 + j] = to_float(vals[j]);
        }
    }
}

template <typename T, int Threads>
__global__ __launch_bounds__(Threads) void quant_v_int8_kernel(
    const T* __restrict__ v, int8_t* __restrict__ out, float* __restrict__ scale_out, int N,
    int padded_N, int H, int D, int64_t stride_b, int64_t stride_h, int64_t stride_n) {

    constexpr int kWaves = Threads / 32;

    const int d_tiles = D / kDTile;
    const int d_tile = blockIdx.x % d_tiles;
    const int bh = blockIdx.x / d_tiles;
    const int h = bh % H;
    const int b = bh / H;
    const int d0 = d_tile * kDTile;

    const T* base = v + static_cast<int64_t>(b) * stride_b + static_cast<int64_t>(h) * stride_h +
                    d0;

    float mx[kDTile];
#pragma unroll
    for (int i = 0; i < kDTile; ++i) mx[i] = 0.0f;

    // A thread owns one group of kKeysPerThread adjacent keys per step; the four
    // loads it issues are independent, so the memory system has something to
    // overlap. This block is the only one touching these channels.
    const int groups = N / kKeysPerThread;
    const int residue = N - groups * kKeysPerThread;
    for (int g = threadIdx.x; g < groups; g += Threads) {
        const int n = g * kKeysPerThread;
        float t0[kDTile], t1[kDTile], t2[kDTile], t3[kDTile];
        load16<T>(base + static_cast<int64_t>(n) * stride_n, t0);
        load16<T>(base + static_cast<int64_t>(n + 1) * stride_n, t1);
        load16<T>(base + static_cast<int64_t>(n + 2) * stride_n, t2);
        load16<T>(base + static_cast<int64_t>(n + 3) * stride_n, t3);
#pragma unroll
        for (int i = 0; i < kDTile; ++i) {
            mx[i] = fmaxf(mx[i], fmaxf(fmaxf(fabsf(t0[i]), fabsf(t1[i])),
                                       fmaxf(fabsf(t2[i]), fabsf(t3[i]))));
        }
    }
    // The keys past the last whole group: at most three, one per thread, and
    // each of them belongs to exactly one thread because the step above is a
    // whole number of groups.
    if (threadIdx.x < residue) {
        float t[kDTile];
        load16<T>(base + static_cast<int64_t>(groups * kKeysPerThread + threadIdx.x) * stride_n, t);
#pragma unroll
        for (int i = 0; i < kDTile; ++i) mx[i] = fmaxf(mx[i], fabsf(t[i]));
    }

    const int lane = threadIdx.x & 31;
    const int wave = threadIdx.x >> 5;

#pragma unroll
    for (int i = 0; i < kDTile; ++i) mx[i] = row_reduce_fmax<32>(mx[i]);

    __shared__ float wave_max[kDTile][kWaves];
    __shared__ float inv_scale_shared[kDTile];
    if (lane == 0) {
#pragma unroll
        for (int i = 0; i < kDTile; ++i) wave_max[i][wave] = mx[i];
    }
    __syncthreads();

    if (threadIdx.x < kDTile) {
        float val = 0.0f;
#pragma unroll
        for (int w = 0; w < kWaves; ++w) val = fmaxf(val, wave_max[threadIdx.x][w]);
        const float sc = fmaxf(val * (1.0f / kInt8Max), 1e-12f);
        scale_out[(static_cast<int64_t>(b) * H + h) * D + d0 + threadIdx.x] = sc;
        inv_scale_shared[threadIdx.x] = 1.0f / sc;
    }
    __syncthreads();

    float inv_scale[kDTile];
#pragma unroll
    for (int i = 0; i < kDTile; ++i) inv_scale[i] = inv_scale_shared[i];

    const int64_t out_row = (static_cast<int64_t>(b) * H + h) * D + d0;

    // Descending so the tail of pass 1 is still in L2 when pass 2 starts. Four
    // keys per store: a channel row is written 4 bytes per lane, 256 bytes per
    // wave, instead of 1 byte per lane.
    for (int g = groups - 1 - static_cast<int>(threadIdx.x); g >= 0; g -= Threads) {
        const int n = g * kKeysPerThread;
        float t0[kDTile], t1[kDTile], t2[kDTile], t3[kDTile];
        load16<T>(base + static_cast<int64_t>(n) * stride_n, t0);
        load16<T>(base + static_cast<int64_t>(n + 1) * stride_n, t1);
        load16<T>(base + static_cast<int64_t>(n + 2) * stride_n, t2);
        load16<T>(base + static_cast<int64_t>(n + 3) * stride_n, t3);
#pragma unroll
        for (int i = 0; i < kDTile; ++i) {
            store4_i8(&out[(out_row + i) * padded_N + n], quant_int8_rcp(t0[i], inv_scale[i]),
                      quant_int8_rcp(t1[i], inv_scale[i]), quant_int8_rcp(t2[i], inv_scale[i]),
                      quant_int8_rcp(t3[i], inv_scale[i]));
        }
    }
    // The keys past the last whole group cannot form a 4-byte store, so they go
    // out one byte at a time.
    if (threadIdx.x < residue) {
        const int src = groups * kKeysPerThread + threadIdx.x;
        float t[kDTile];
        load16<T>(base + static_cast<int64_t>(src) * stride_n, t);
#pragma unroll
        for (int i = 0; i < kDTile; ++i) {
            out[(out_row + i) * padded_N + src] = quant_int8_rcp(t[i], inv_scale[i]);
        }
    }

    // The attention kernel reads whole CTA_K tiles, so the padding has to be a
    // real zero rather than whatever the allocator left behind.
    for (int src = N + static_cast<int>(threadIdx.x); src < padded_N; src += Threads) {
#pragma unroll
        for (int i = 0; i < kDTile; ++i) out[(out_row + i) * padded_N + src] = 0;
    }
}

template <typename T>
void launch_typed(const void* v, void* out, void* scale, int B, int H, int N, int D, int padded_N,
                  int64_t stride_b, int64_t stride_h, int64_t stride_n, hipStream_t stream) {
    const int blocks = B * H * (D / kDTile);
    if (N <= 256) {
        quant_v_int8_kernel<T, 128><<<blocks, 128, 0, stream>>>(
            static_cast<const T*>(v), static_cast<int8_t*>(out), static_cast<float*>(scale), N,
            padded_N, H, D, stride_b, stride_h, stride_n);
    } else {
        quant_v_int8_kernel<T, 512><<<blocks, 512, 0, stream>>>(
            static_cast<const T*>(v), static_cast<int8_t*>(out), static_cast<float*>(scale), N,
            padded_N, H, D, stride_b, stride_h, stride_n);
    }
}

}  // namespace
}  // namespace comfy::hip_backend::sage

using namespace comfy::hip_backend::sage;

extern "C" void launch_sage_quant_v_int8(const void* v, void* out, void* scale, int B, int H,
                                         int N, int D, int padded_N, int64_t stride_b,
                                         int64_t stride_h, int64_t stride_n, int input_dtype_code,
                                         hipStream_t stream) {
    // The kernel loads V 16 bytes at a time, so the base pointer and every row
    // offset it can form have to be 16-byte aligned. An extent of one forms no
    // offset, and PyTorch rewrites such a stride to 1 on the way through DLPack
    // (ATen/DLConvertor.cpp, gh-83069), so policing it would reject a tensor the
    // caller has no way to hand over differently.
    const size_t element_size = input_dtype_code == 0 ? sizeof(float) : sizeof(__half);
    const auto stride_ok = [element_size](int64_t stride, int extent) {
        return extent < 2 || (static_cast<size_t>(stride) * element_size) % 16 == 0;
    };
    if (reinterpret_cast<uintptr_t>(v) % 16 != 0 || !stride_ok(stride_b, B) ||
        !stride_ok(stride_h, H) || !stride_ok(stride_n, N)) {
        throw std::runtime_error(
            "sage int8 attention: V base pointer and B/H/N strides must be 16-byte aligned");
    }
    if (D <= 0 || D % kDTile != 0) {
        throw std::runtime_error(std::string("sage int8 attention: head_dim must be a positive "
                                                    "multiple of ") +
                                 std::to_string(kDTile));
    }
    // The quantize pass writes four adjacent keys at a time through store4_i8,
    // so the row base has to be 4-byte aligned and a row has to be a whole
    // number of 4-byte groups. Rows are padded_N wide and start at a channel
    // boundary, which leaves the base pointer and the padding to check.
    if (reinterpret_cast<uintptr_t>(out) % 4 != 0 || padded_N % 4 != 0) {
        throw std::runtime_error(
            "sage int8 attention: V int8 output must be 4-byte aligned in both dimensions");
    }
    // Rows are padded_N wide and the quantize pass writes columns up to N, so a
    // short pad would run every row past its end.
    if (N < 0 || padded_N < N) {
        throw std::runtime_error(
            "sage int8 attention: padded key length must be at least the key length");
    }

    switch (input_dtype_code) {
        case 0:
            launch_typed<float>(v, out, scale, B, H, N, D, padded_N, stride_b, stride_h, stride_n,
                                stream);
            break;
        case 1:
            launch_typed<__half>(v, out, scale, B, H, N, D, padded_N, stride_b, stride_h, stride_n,
                                 stream);
            break;
        case 2:
            launch_typed<__bf16>(v, out, scale, B, H, N, D, padded_N, stride_b, stride_h, stride_n,
                                 stream);
            break;
        default:
            throw std::runtime_error(
                "sage int8 attention: input dtype must be fp32, fp16 or bf16");
    }

    const hipError_t error = hipGetLastError();
    if (error != hipSuccess) {
        throw std::runtime_error(std::string("sage int8 V quantize kernel launch failed: ") +
                                 hipGetErrorString(error));
    }
}
