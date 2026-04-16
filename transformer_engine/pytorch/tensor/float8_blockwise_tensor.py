# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tensor class with FP8 data quantized with NxN tiles"""
from __future__ import annotations
from collections.abc import Iterable
import math
import os
import warnings
from typing import Any, Optional, Tuple, Union

import torch
import transformer_engine_torch as tex
from transformer_engine_torch import DType as TE_DType
from transformer_engine.common.recipe import Float8BlockScaling, Recipe
from torch.utils.cpp_extension import load_inline
from .storage.float8_blockwise_tensor_storage import Float8BlockwiseQTensorStorage
from ..quantized_tensor import QuantizedTensor, Quantizer
from ._quantization_helpers import _DequantizeFunc, _IdentityFunc
from ..utils import devices_match, round_up_to_nearest_multiple

aten = torch.ops.aten
_FUSED_ELEMENTWISE_MODULE = None


class _SigmoidToFloat8Func(torch.autograd.Function):
    """Sigmoid autograd bridge that preserves Float8BlockwiseQTensor outputs."""

    @staticmethod
    def forward(ctx, tensor: "Float8BlockwiseQTensor") -> "Float8BlockwiseQTensor":
        quantizer = tensor._get_quantizer().copy()
        quantizer.set_usage(rowwise=True, columnwise=True)
        with torch._C._DisableTorchDispatch():
            out = torch.sigmoid(tensor.dequantize())
        out_q = quantizer.quantize(out, dtype=tensor.dtype)
        ctx.save_for_backward(out_q)
        ctx.input_dtype = tensor.dtype
        return out_q.requires_grad_(out.requires_grad)

    @staticmethod
    def backward(ctx, grad):
        (out_q,) = ctx.saved_tensors
        if (
            isinstance(out_q, Float8BlockwiseQTensor)
            and out_q._rowwise_data is not None
            and out_q._rowwise_scale_inv is not None
            and not out_q._is_2D_scaled
        ):
            target_dtype = ctx.input_dtype
            if isinstance(grad, QuantizedTensor):
                grad = grad.dequantize(dtype=target_dtype)
            elif target_dtype is not None and grad.dtype != target_dtype:
                grad = grad.to(target_dtype)
            grad_f = grad.contiguous().to(dtype=torch.float32)
            mod = _get_fused_elementwise_module()
            grad_input = mod.fused_sigmoid_fp8_backward(
                out_q._rowwise_data,
                out_q._rowwise_scale_inv,
                grad_f,
            )
            if target_dtype is not None and grad_input.dtype != target_dtype:
                grad_input = grad_input.to(target_dtype)
            return grad_input
        out = out_q.dequantize(dtype=ctx.input_dtype)
        if isinstance(grad, QuantizedTensor):
            grad = grad.dequantize(dtype=ctx.input_dtype)
        elif grad.dtype != ctx.input_dtype:
            grad = grad.to(ctx.input_dtype)
        grad_input = grad * out * (1 - out)
        return grad_input


def _dequantize_dispatch_input(arg):
    """Convert quantized inputs to plain tensors inside dispatch handlers."""
    if isinstance(arg, QuantizedTensor):
        with torch._C._DisableTorchDispatch():
            return arg.dequantize()
    return arg


def _dispatch_input_dtype(arg) -> Optional[torch.dtype]:
    """Nominal dtype for dispatch handler gradient outputs."""
    if isinstance(arg, QuantizedTensor):
        return arg.dtype
    if isinstance(arg, torch.Tensor):
        return arg.dtype
    return None


class _ReluToFloat8Func(torch.autograd.Function):
    """ReLU autograd bridge that preserves Float8BlockwiseQTensor outputs."""

    @staticmethod
    def forward(ctx, tensor: "Float8BlockwiseQTensor") -> "Float8BlockwiseQTensor":
        quantizer = tensor._get_quantizer().copy()
        quantizer.set_usage(rowwise=True, columnwise=True)
        out = torch.relu(_dequantize_dispatch_input(tensor))
        out_q = quantizer.quantize(out, dtype=tensor.dtype)
        ctx.save_for_backward(out_q)
        ctx.input_dtype = tensor.dtype
        return out_q

    @staticmethod
    def backward(ctx, grad):
        (out_q,) = ctx.saved_tensors
        if (
            isinstance(out_q, Float8BlockwiseQTensor)
            and out_q._rowwise_data is not None
            and not out_q._is_2D_scaled
        ):
            return fused_relu_backward_fp8(
                out_q._rowwise_data,
                grad,
                input_dtype=ctx.input_dtype,
            )
        out = out_q.dequantize(dtype=ctx.input_dtype)
        if isinstance(grad, QuantizedTensor):
            grad = grad.dequantize(dtype=ctx.input_dtype)
        elif grad.dtype != ctx.input_dtype:
            grad = grad.to(ctx.input_dtype)
        return grad * (out > 0).to(dtype=grad.dtype)


class _MulToFloat8Func(torch.autograd.Function):
    """Elementwise multiply bridge that preserves Float8BlockwiseQTensor outputs."""

    @staticmethod
    def forward(ctx, lhs, rhs):
        lhs_hp = _dequantize_dispatch_input(lhs)
        rhs_hp = _dequantize_dispatch_input(rhs)
        out = lhs_hp * rhs_hp
        quantizer_src = lhs if isinstance(lhs, Float8BlockwiseQTensor) else rhs
        assert isinstance(quantizer_src, Float8BlockwiseQTensor)
        quantizer = quantizer_src._get_quantizer().copy()
        quantizer.set_usage(rowwise=True, columnwise=True)
        lhs_saved = quantizer.quantize(lhs_hp, dtype=lhs_hp.dtype)
        rhs_saved = quantizer.quantize(rhs_hp, dtype=rhs_hp.dtype)
        ctx.save_for_backward(lhs_saved, rhs_saved)
        ctx.lhs_dtype = _dispatch_input_dtype(lhs)
        ctx.rhs_dtype = _dispatch_input_dtype(rhs)
        ctx.lhs_is_tensor = isinstance(lhs, torch.Tensor)
        ctx.rhs_is_tensor = isinstance(rhs, torch.Tensor)
        return quantizer.quantize(out, dtype=quantizer_src.dtype)

    @staticmethod
    def backward(ctx, grad):
        lhs_q, rhs_q = ctx.saved_tensors
        if (
            isinstance(lhs_q, Float8BlockwiseQTensor)
            and isinstance(rhs_q, Float8BlockwiseQTensor)
            and lhs_q._rowwise_data is not None
            and lhs_q._rowwise_scale_inv is not None
            and rhs_q._rowwise_data is not None
            and rhs_q._rowwise_scale_inv is not None
            and not lhs_q._is_2D_scaled
            and not rhs_q._is_2D_scaled
        ):
            target_dtype = ctx.lhs_dtype or ctx.rhs_dtype
            if isinstance(grad, QuantizedTensor):
                grad = grad.dequantize(dtype=target_dtype)
            if target_dtype is not None and grad.dtype != target_dtype:
                grad = grad.to(target_dtype)
            grad_f = grad.contiguous().to(dtype=torch.float32)
            mod = _get_fused_elementwise_module()
            grad_lhs, grad_rhs = mod.fused_mul_fp8_backward(
                lhs_q._rowwise_data,
                lhs_q._rowwise_scale_inv,
                rhs_q._rowwise_data,
                rhs_q._rowwise_scale_inv,
                grad_f,
            )
            if ctx.lhs_is_tensor and ctx.lhs_dtype is not None and grad_lhs.dtype != ctx.lhs_dtype:
                grad_lhs = grad_lhs.to(ctx.lhs_dtype)
            if ctx.rhs_is_tensor and ctx.rhs_dtype is not None and grad_rhs.dtype != ctx.rhs_dtype:
                grad_rhs = grad_rhs.to(ctx.rhs_dtype)
            return grad_lhs if ctx.lhs_is_tensor else None, grad_rhs if ctx.rhs_is_tensor else None
        lhs_hp = lhs_q.dequantize(dtype=ctx.lhs_dtype)
        rhs_hp = rhs_q.dequantize(dtype=ctx.rhs_dtype)
        if isinstance(grad, QuantizedTensor):
            grad = grad.dequantize(dtype=ctx.lhs_dtype or ctx.rhs_dtype)
        target_dtype = ctx.lhs_dtype or ctx.rhs_dtype
        if target_dtype is not None and grad.dtype != target_dtype:
            grad = grad.to(target_dtype)
        grad_lhs = grad * rhs_hp if ctx.lhs_is_tensor else None
        grad_rhs = grad * lhs_hp if ctx.rhs_is_tensor else None
        if grad_lhs is not None and ctx.lhs_dtype is not None and grad_lhs.dtype != ctx.lhs_dtype:
            grad_lhs = grad_lhs.to(ctx.lhs_dtype)
        if grad_rhs is not None and ctx.rhs_dtype is not None and grad_rhs.dtype != ctx.rhs_dtype:
            grad_rhs = grad_rhs.to(ctx.rhs_dtype)
        return grad_lhs, grad_rhs


def _get_fused_elementwise_module():
    """Lazily JIT-compile fused FP8 elementwise kernels used by Variant C."""
    global _FUSED_ELEMENTWISE_MODULE
    if _FUSED_ELEMENTWISE_MODULE is not None:
        return _FUSED_ELEMENTWISE_MODULE

    arch_suffix = "cpu"
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        arch_suffix = f"sm{major}{minor}"
        if "TORCH_CUDA_ARCH_LIST" not in os.environ:
            os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    # Keep the extension name stable so all ranks can reuse the same compiled artifact.
    module_name = f"fused_elementwise_fp8_{arch_suffix}_v2"
    cpp_source = r"""
    #include <torch/extension.h>
    #include <vector>

    std::vector<torch::Tensor> fused_sigmoid_gate_fp8_forward(
        torch::Tensor proj_data,
        torch::Tensor proj_scale_inv,
        torch::Tensor gate_data,
        torch::Tensor gate_scale_inv,
        torch::Tensor mask);

    std::vector<torch::Tensor> fused_sigmoid_gate_fp8_backward(
        torch::Tensor proj_data,
        torch::Tensor proj_scale_inv,
        torch::Tensor saved_g_data,
        torch::Tensor saved_g_scale_inv,
        torch::Tensor grad_out,
        torch::Tensor mask);

    torch::Tensor fused_sigmoid_fp8_backward(
        torch::Tensor saved_out_data,
        torch::Tensor saved_out_scale_inv,
        torch::Tensor grad_out);

    std::vector<torch::Tensor> fused_relu_fp8_forward(
        torch::Tensor rowwise_data,
        torch::Tensor columnwise_data);

    torch::Tensor fused_relu_fp8_backward(
        torch::Tensor saved_rowwise_data,
        torch::Tensor grad_out);

    std::vector<torch::Tensor> fused_mul_fp8_backward(
        torch::Tensor lhs_data,
        torch::Tensor lhs_scale_inv,
        torch::Tensor rhs_data,
        torch::Tensor rhs_scale_inv,
        torch::Tensor grad_out);
    """

    cuda_source = r"""
    #include <torch/extension.h>
    #include <ATen/cuda/CUDAContext.h>
    #include <cuda.h>
    #include <cuda_bf16.h>
    #include <cuda_fp8.h>
    #include <vector>
    #include <cmath>

    namespace {
    constexpr int kBlockLen = 128;
    constexpr float kFp8E4M3Max = 448.0f;

    __device__ inline float fp8_byte_to_float(uint8_t x) {
      __nv_fp8_e4m3 v;
      v.__x = x;
      return static_cast<float>(v);
    }

    __device__ inline uint8_t float_to_fp8_byte(float x) {
      return static_cast<uint8_t>(__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3));
    }

    __device__ inline float maybe_pow2_scale_inv(float scale_inv) {
      if (scale_inv <= 0.0f) {
        return 1.0f;
      }
      return exp2f(ceilf(log2f(scale_inv)));
    }

    __device__ inline float sigmoidf_approx(float x) {
      return 1.0f / (1.0f + expf(-x));
    }

    __global__ void fused_sigmoid_gate_fp8_fwd_kernel(
        const uint8_t* proj_data,
        const float* proj_scale_inv,
        const uint8_t* gate_data,
        const float* gate_scale_inv,
        const float* mask,
        uint8_t* out_data,
        float* out_scale_inv,
        uint8_t* saved_g_data,
        float* saved_g_scale_inv,
        int64_t M,
        int64_t K,
        int64_t scale_stride) {
      const int tile_m = blockIdx.y;
      const int tile_k = blockIdx.x;
      const int64_t m0 = static_cast<int64_t>(tile_m) * kBlockLen;
      const int64_t k0 = static_cast<int64_t>(tile_k) * kBlockLen;
      __shared__ float smax_y[256];
      __shared__ float smax_g[256];

      float local_max_y = 0.0f;
      float local_max_g = 0.0f;
      for (int idx = threadIdx.x; idx < kBlockLen * kBlockLen; idx += blockDim.x) {
        const int dm = idx / kBlockLen;
        const int dk = idx % kBlockLen;
        const int64_t m = m0 + dm;
        const int64_t k = k0 + dk;
        if (m >= M || k >= K) continue;
        const int64_t flat = m * K + k;
        const float proj = fp8_byte_to_float(proj_data[flat]) * proj_scale_inv[tile_m * scale_stride + tile_k];
        const float gate = fp8_byte_to_float(gate_data[flat]) * gate_scale_inv[tile_m * scale_stride + tile_k];
        const float g = sigmoidf_approx(gate);
        float y = proj * g;
        if (mask != nullptr) y *= mask[m];
        local_max_y = fmaxf(local_max_y, fabsf(y));
        local_max_g = fmaxf(local_max_g, fabsf(g));
      }
      smax_y[threadIdx.x] = local_max_y;
      smax_g[threadIdx.x] = local_max_g;
      __syncthreads();
      for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
          smax_y[threadIdx.x] = fmaxf(smax_y[threadIdx.x], smax_y[threadIdx.x + offset]);
          smax_g[threadIdx.x] = fmaxf(smax_g[threadIdx.x], smax_g[threadIdx.x + offset]);
        }
        __syncthreads();
      }
      const float tile_scale_inv_y = threadIdx.x == 0
          ? maybe_pow2_scale_inv(smax_y[0] > 0.0f ? smax_y[0] / kFp8E4M3Max : 1.0f)
          : 0.0f;
      const float tile_scale_inv_g = threadIdx.x == 0
          ? maybe_pow2_scale_inv(smax_g[0] > 0.0f ? smax_g[0] / kFp8E4M3Max : 1.0f)
          : 0.0f;
      __shared__ float scale_inv_y;
      __shared__ float scale_inv_g;
      if (threadIdx.x == 0) {
        scale_inv_y = tile_scale_inv_y;
        scale_inv_g = tile_scale_inv_g;
        out_scale_inv[tile_m * scale_stride + tile_k] = tile_scale_inv_y;
        saved_g_scale_inv[tile_m * scale_stride + tile_k] = tile_scale_inv_g;
      }
      __syncthreads();

      for (int idx = threadIdx.x; idx < kBlockLen * kBlockLen; idx += blockDim.x) {
        const int dm = idx / kBlockLen;
        const int dk = idx % kBlockLen;
        const int64_t m = m0 + dm;
        const int64_t k = k0 + dk;
        if (m >= M || k >= K) continue;
        const int64_t flat = m * K + k;
        const float proj = fp8_byte_to_float(proj_data[flat]) * proj_scale_inv[tile_m * scale_stride + tile_k];
        const float gate = fp8_byte_to_float(gate_data[flat]) * gate_scale_inv[tile_m * scale_stride + tile_k];
        const float g = sigmoidf_approx(gate);
        float y = proj * g;
        if (mask != nullptr) y *= mask[m];
        out_data[flat] = float_to_fp8_byte(y / scale_inv_y);
        saved_g_data[flat] = float_to_fp8_byte(g / scale_inv_g);
      }
    }

    __global__ void fused_sigmoid_gate_fp8_bwd_kernel(
        const uint8_t* proj_data,
        const float* proj_scale_inv,
        const uint8_t* saved_g_data,
        const float* saved_g_scale_inv,
        const float* grad_out,
        const float* mask,
        float* grad_proj,
        float* grad_gate,
        int64_t M,
        int64_t K,
        int64_t scale_stride) {
      const int64_t flat = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
      const int64_t total = M * K;
      if (flat >= total) return;
      const int64_t m = flat / K;
      const int64_t k = flat % K;
      const int64_t tile_m = m / kBlockLen;
      const int64_t tile_k = k / kBlockLen;
      float gout = grad_out[flat];
      if (mask != nullptr) gout *= mask[m];
      const float proj = fp8_byte_to_float(proj_data[flat]) * proj_scale_inv[tile_m * scale_stride + tile_k];
      const float g = fp8_byte_to_float(saved_g_data[flat]) * saved_g_scale_inv[tile_m * scale_stride + tile_k];
      grad_proj[flat] = gout * g;
      grad_gate[flat] = gout * proj * g * (1.0f - g);
    }

    __global__ void fused_relu_fp8_fwd_kernel(
        const uint8_t* input,
        uint8_t* output,
        int64_t n) {
      const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
      if (idx >= n) return;
      const uint8_t x = input[idx];
      const bool positive = ((x & 0x80u) == 0u) && (x != 0u);
      output[idx] = positive ? x : static_cast<uint8_t>(0u);
    }

    __global__ void fused_relu_fp8_bwd_kernel(
        const uint8_t* saved_rowwise_data,
        const float* grad_out,
        float* grad_input,
        int64_t n) {
      const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
      if (idx >= n) return;
      const uint8_t x = saved_rowwise_data[idx];
      const bool positive = ((x & 0x80u) == 0u) && (x != 0u);
      grad_input[idx] = positive ? grad_out[idx] : 0.0f;
    }

    __global__ void fused_sigmoid_fp8_bwd_kernel(
        const uint8_t* saved_out_data,
        const float* saved_out_scale_inv,
        const float* grad_out,
        float* grad_input,
        int64_t M,
        int64_t K,
        int64_t scale_stride) {
      const int64_t flat = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
      const int64_t total = M * K;
      if (flat >= total) return;
      const int64_t m = flat / K;
      const int64_t k = flat % K;
      const int64_t tile_k = k / kBlockLen;
      const float scale = saved_out_scale_inv[tile_k * scale_stride + m];
      const float out = fp8_byte_to_float(saved_out_data[flat]) * scale;
      grad_input[flat] = grad_out[flat] * out * (1.0f - out);
    }

    __global__ void fused_mul_fp8_bwd_kernel(
        const uint8_t* lhs_data,
        const float* lhs_scale_inv,
        const uint8_t* rhs_data,
        const float* rhs_scale_inv,
        const float* grad_out,
        float* grad_lhs,
        float* grad_rhs,
        int64_t M,
        int64_t K,
        int64_t scale_stride) {
      const int64_t flat = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
      const int64_t total = M * K;
      if (flat >= total) return;
      const int64_t m = flat / K;
      const int64_t k = flat % K;
      const int64_t tile_k = k / kBlockLen;
      const float lhs_scale = lhs_scale_inv[tile_k * scale_stride + m];
      const float rhs_scale = rhs_scale_inv[tile_k * scale_stride + m];
      const float lhs = fp8_byte_to_float(lhs_data[flat]) * lhs_scale;
      const float rhs = fp8_byte_to_float(rhs_data[flat]) * rhs_scale;
      const float go = grad_out[flat];
      grad_lhs[flat] = go * rhs;
      grad_rhs[flat] = go * lhs;
    }
    }  // namespace

    std::vector<torch::Tensor> fused_sigmoid_gate_fp8_forward(
        torch::Tensor proj_data,
        torch::Tensor proj_scale_inv,
        torch::Tensor gate_data,
        torch::Tensor gate_scale_inv,
        torch::Tensor mask) {
      TORCH_CHECK(proj_data.is_cuda(), "proj_data must be CUDA");
      TORCH_CHECK(gate_data.is_cuda(), "gate_data must be CUDA");
      auto proj_data_c = proj_data.contiguous();
      auto gate_data_c = gate_data.contiguous();
      auto proj_scale_c = proj_scale_inv.contiguous();
      auto gate_scale_c = gate_scale_inv.contiguous();
      auto mask_c = mask.defined() && mask.numel() > 0 ? mask.contiguous() : torch::Tensor();
      const int64_t K = proj_data_c.size(-1);
      const int64_t M = proj_data_c.numel() / K;
      auto out_data = torch::empty_like(proj_data_c);
      auto out_scale = torch::empty_like(proj_scale_c);
      auto saved_g_data = torch::empty_like(proj_data_c);
      auto saved_g_scale = torch::empty_like(proj_scale_c);
      const dim3 grid((K + kBlockLen - 1) / kBlockLen, (M + kBlockLen - 1) / kBlockLen);
      const dim3 block(256);
      fused_sigmoid_gate_fp8_fwd_kernel<<<grid, block, 0, at::cuda::getDefaultCUDAStream()>>>(
          proj_data_c.data_ptr<uint8_t>(),
          proj_scale_c.data_ptr<float>(),
          gate_data_c.data_ptr<uint8_t>(),
          gate_scale_c.data_ptr<float>(),
          mask_c.defined() && mask_c.numel() > 0 ? mask_c.data_ptr<float>() : nullptr,
          out_data.data_ptr<uint8_t>(),
          out_scale.data_ptr<float>(),
          saved_g_data.data_ptr<uint8_t>(),
          saved_g_scale.data_ptr<float>(),
          M,
          K,
          proj_scale_c.size(1));
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      return {out_data, out_scale, saved_g_data, saved_g_scale};
    }

    std::vector<torch::Tensor> fused_sigmoid_gate_fp8_backward(
        torch::Tensor proj_data,
        torch::Tensor proj_scale_inv,
        torch::Tensor saved_g_data,
        torch::Tensor saved_g_scale_inv,
        torch::Tensor grad_out,
        torch::Tensor mask) {
      TORCH_CHECK(proj_data.is_cuda(), "proj_data must be CUDA");
      auto proj_data_c = proj_data.contiguous();
      auto proj_scale_c = proj_scale_inv.contiguous();
      auto saved_g_data_c = saved_g_data.contiguous();
      auto saved_g_scale_c = saved_g_scale_inv.contiguous();
      auto grad_out_c = grad_out.contiguous();
      auto mask_c = mask.defined() && mask.numel() > 0 ? mask.contiguous() : torch::Tensor();
      const int64_t K = proj_data_c.size(-1);
      const int64_t M = proj_data_c.numel() / K;
      auto grad_proj = torch::empty_like(grad_out_c, grad_out_c.options().dtype(torch::kFloat32));
      auto grad_gate = torch::empty_like(grad_out_c, grad_out_c.options().dtype(torch::kFloat32));
      const int threads = 256;
      const int blocks = (M * K + threads - 1) / threads;
      fused_sigmoid_gate_fp8_bwd_kernel<<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
          proj_data_c.data_ptr<uint8_t>(),
          proj_scale_c.data_ptr<float>(),
          saved_g_data_c.data_ptr<uint8_t>(),
          saved_g_scale_c.data_ptr<float>(),
          grad_out_c.data_ptr<float>(),
          mask_c.defined() && mask_c.numel() > 0 ? mask_c.data_ptr<float>() : nullptr,
          grad_proj.data_ptr<float>(),
          grad_gate.data_ptr<float>(),
          M,
          K,
          proj_scale_c.size(1));
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      return {grad_proj, grad_gate};
    }

    std::vector<torch::Tensor> fused_relu_fp8_forward(
        torch::Tensor rowwise_data,
        torch::Tensor columnwise_data) {
      TORCH_CHECK(rowwise_data.is_cuda(), "rowwise_data must be CUDA");
      auto rowwise_in = rowwise_data.contiguous();
      auto rowwise_out = torch::empty_like(rowwise_in);
      const int threads = 256;
      const int blocks = (rowwise_in.numel() + threads - 1) / threads;
      fused_relu_fp8_fwd_kernel<<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
          rowwise_in.data_ptr<uint8_t>(),
          rowwise_out.data_ptr<uint8_t>(),
          rowwise_in.numel());
      C10_CUDA_KERNEL_LAUNCH_CHECK();

      torch::Tensor columnwise_out;
      if (columnwise_data.defined() && columnwise_data.numel() > 0) {
        auto columnwise_in = columnwise_data.contiguous();
        columnwise_out = torch::empty_like(columnwise_in);
        const int col_blocks = (columnwise_in.numel() + threads - 1) / threads;
        fused_relu_fp8_fwd_kernel<<<col_blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
            columnwise_in.data_ptr<uint8_t>(),
            columnwise_out.data_ptr<uint8_t>(),
            columnwise_in.numel());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      } else {
        columnwise_out = torch::Tensor();
      }
      return {rowwise_out, columnwise_out};
    }

    torch::Tensor fused_relu_fp8_backward(
        torch::Tensor saved_rowwise_data,
        torch::Tensor grad_out) {
      TORCH_CHECK(saved_rowwise_data.is_cuda(), "saved_rowwise_data must be CUDA");
      auto saved_rowwise = saved_rowwise_data.contiguous();
      auto grad_out_c = grad_out.contiguous();
      auto grad_input = torch::empty_like(grad_out_c, grad_out_c.options().dtype(torch::kFloat32));
      const int threads = 256;
      const int blocks = (saved_rowwise.numel() + threads - 1) / threads;
      fused_relu_fp8_bwd_kernel<<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
          saved_rowwise.data_ptr<uint8_t>(),
          grad_out_c.data_ptr<float>(),
          grad_input.data_ptr<float>(),
          saved_rowwise.numel());
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      return grad_input;
    }

    torch::Tensor fused_sigmoid_fp8_backward(
        torch::Tensor saved_out_data,
        torch::Tensor saved_out_scale_inv,
        torch::Tensor grad_out) {
      TORCH_CHECK(saved_out_data.is_cuda(), "saved_out_data must be CUDA");
      auto saved_data_c = saved_out_data.contiguous();
      auto saved_scale_c = saved_out_scale_inv.contiguous();
      auto grad_out_c = grad_out.contiguous();
      const int64_t K = saved_data_c.size(-1);
      const int64_t M = saved_data_c.numel() / K;
      auto grad_input = torch::empty_like(grad_out_c, grad_out_c.options().dtype(torch::kFloat32));
      const int threads = 256;
      const int blocks = (M * K + threads - 1) / threads;
      fused_sigmoid_fp8_bwd_kernel<<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
          saved_data_c.data_ptr<uint8_t>(),
          saved_scale_c.data_ptr<float>(),
          grad_out_c.data_ptr<float>(),
          grad_input.data_ptr<float>(),
          M,
          K,
          saved_scale_c.size(1));
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      return grad_input;
    }

    std::vector<torch::Tensor> fused_mul_fp8_backward(
        torch::Tensor lhs_data,
        torch::Tensor lhs_scale_inv,
        torch::Tensor rhs_data,
        torch::Tensor rhs_scale_inv,
        torch::Tensor grad_out) {
      TORCH_CHECK(lhs_data.is_cuda(), "lhs_data must be CUDA");
      auto lhs_data_c = lhs_data.contiguous();
      auto lhs_scale_c = lhs_scale_inv.contiguous();
      auto rhs_data_c = rhs_data.contiguous();
      auto rhs_scale_c = rhs_scale_inv.contiguous();
      auto grad_out_c = grad_out.contiguous();
      const int64_t K = lhs_data_c.size(-1);
      const int64_t M = lhs_data_c.numel() / K;
      auto grad_lhs = torch::empty_like(grad_out_c, grad_out_c.options().dtype(torch::kFloat32));
      auto grad_rhs = torch::empty_like(grad_out_c, grad_out_c.options().dtype(torch::kFloat32));
      const int threads = 256;
      const int blocks = (M * K + threads - 1) / threads;
      fused_mul_fp8_bwd_kernel<<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
          lhs_data_c.data_ptr<uint8_t>(),
          lhs_scale_c.data_ptr<float>(),
          rhs_data_c.data_ptr<uint8_t>(),
          rhs_scale_c.data_ptr<float>(),
          grad_out_c.data_ptr<float>(),
          grad_lhs.data_ptr<float>(),
          grad_rhs.data_ptr<float>(),
          M,
          K,
          lhs_scale_c.size(1));
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      return {grad_lhs, grad_rhs};
    }
    """

    _FUSED_ELEMENTWISE_MODULE = load_inline(
        name=module_name,
        cpp_sources=[cpp_source],
        cuda_sources=[cuda_source],
        functions=[
            "fused_sigmoid_gate_fp8_forward",
            "fused_sigmoid_gate_fp8_backward",
            "fused_sigmoid_fp8_backward",
            "fused_relu_fp8_forward",
            "fused_relu_fp8_backward",
            "fused_mul_fp8_backward",
        ],
        extra_cuda_cflags=["-O3"],
        with_cuda=True,
        verbose=False,
    )
    return _FUSED_ELEMENTWISE_MODULE


def fused_sigmoid_gate_forward_fp8(
    proj: "Float8BlockwiseQTensor",
    gate: "Float8BlockwiseQTensor",
    mask: Optional[torch.Tensor] = None,
    *,
    preserve_columnwise_output: bool = True,
) -> tuple["Float8BlockwiseQTensor", "Float8BlockwiseQTensor"]:
    """Fused forward for proj * sigmoid(gate), returning output and saved sigmoid state."""
    if (
        not isinstance(proj, Float8BlockwiseQTensor)
        or not isinstance(gate, Float8BlockwiseQTensor)
        or proj._rowwise_data is None
        or proj._rowwise_scale_inv is None
        or gate._rowwise_data is None
        or gate._rowwise_scale_inv is None
        or not proj._is_2D_scaled
        or not gate._is_2D_scaled
        or proj._fp8_dtype != TE_DType.kFloat8E4M3
        or gate._fp8_dtype != TE_DType.kFloat8E4M3
    ):
        raise RuntimeError("fused_sigmoid_gate_forward_fp8 requires rowwise 2D-scaled E4M3 Float8BlockwiseQTensor")

    mod = _get_fused_elementwise_module()
    mask_flat = (
        mask.contiguous().view(-1).to(device=proj.device, dtype=torch.float32)
        if mask is not None
        else torch.empty(0, device=proj.device, dtype=torch.float32)
    )
    out_rowwise_data, out_rowwise_scale_inv, g_rowwise_data, g_rowwise_scale_inv = mod.fused_sigmoid_gate_fp8_forward(
        proj._rowwise_data,
        proj._rowwise_scale_inv,
        gate._rowwise_data,
        gate._rowwise_scale_inv,
        mask_flat,
    )
    out_quantizer = proj._get_quantizer().copy()
    out_quantizer.set_usage(rowwise=True, columnwise=preserve_columnwise_output)
    out_q = Float8BlockwiseQTensor(
        shape=proj.shape,
        dtype=proj.dtype,
        fp8_dtype=proj._fp8_dtype,
        rowwise_data=out_rowwise_data,
        rowwise_scale_inv=out_rowwise_scale_inv,
        columnwise_data=None,
        columnwise_scale_inv=None,
        quantizer=out_quantizer,
        is_2D_scaled=True,
        requires_grad=proj.requires_grad or gate.requires_grad,
    )
    if preserve_columnwise_output:
        out_q.update_usage(rowwise_usage=True, columnwise_usage=True)

    g_quantizer = gate._get_quantizer().copy()
    g_quantizer.set_usage(rowwise=True, columnwise=False)
    g_q = Float8BlockwiseQTensor(
        shape=gate.shape,
        dtype=gate.dtype,
        fp8_dtype=gate._fp8_dtype,
        rowwise_data=g_rowwise_data,
        rowwise_scale_inv=g_rowwise_scale_inv,
        columnwise_data=None,
        columnwise_scale_inv=None,
        quantizer=g_quantizer,
        is_2D_scaled=True,
        requires_grad=False,
    )
    return out_q, g_q


def fused_sigmoid_gate_backward_fp8(
    proj: "Float8BlockwiseQTensor",
    saved_g: "Float8BlockwiseQTensor",
    grad_out: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    *,
    proj_dtype: Optional[torch.dtype] = None,
    gate_dtype: Optional[torch.dtype] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused backward for proj * sigmoid(gate) using saved FP8 sigmoid output."""
    if isinstance(grad_out, QuantizedTensor):
        grad_out = grad_out.dequantize(dtype=proj_dtype or gate_dtype or proj.dtype)
    target_dtype = proj_dtype or gate_dtype or proj.dtype
    grad_out_f = grad_out.contiguous().to(dtype=torch.float32)
    mask_flat = (
        mask.contiguous().view(-1).to(device=proj.device, dtype=torch.float32)
        if mask is not None
        else torch.empty(0, device=proj.device, dtype=torch.float32)
    )
    mod = _get_fused_elementwise_module()
    grad_proj, grad_gate = mod.fused_sigmoid_gate_fp8_backward(
        proj._rowwise_data,
        proj._rowwise_scale_inv,
        saved_g._rowwise_data,
        saved_g._rowwise_scale_inv,
        grad_out_f,
        mask_flat,
    )
    if proj_dtype is not None:
        grad_proj = grad_proj.to(proj_dtype)
    if gate_dtype is not None:
        grad_gate = grad_gate.to(gate_dtype)
    elif target_dtype is not None:
        grad_gate = grad_gate.to(target_dtype)
    return grad_proj, grad_gate


def fused_relu_forward_fp8(
    tensor: "Float8BlockwiseQTensor",
) -> "Float8BlockwiseQTensor":
    """Fused raw-FP8 ReLU forward, preserving scales and layout copies."""
    if (
        not isinstance(tensor, Float8BlockwiseQTensor)
        or tensor._rowwise_data is None
        or tensor._rowwise_scale_inv is None
    ):
        raise RuntimeError("fused_relu_forward_fp8 requires rowwise Float8BlockwiseQTensor")
    mod = _get_fused_elementwise_module()
    columnwise_data = (
        tensor._columnwise_data
        if tensor._columnwise_data is not None
        else torch.empty(0, device=tensor.device, dtype=torch.uint8)
    )
    rowwise_out, columnwise_out = mod.fused_relu_fp8_forward(tensor._rowwise_data, columnwise_data)
    out_q = Float8BlockwiseQTensor(
        shape=tensor.shape,
        dtype=tensor.dtype,
        fp8_dtype=tensor._fp8_dtype,
        rowwise_data=rowwise_out,
        rowwise_scale_inv=tensor._rowwise_scale_inv.detach().clone(),
        columnwise_data=columnwise_out if columnwise_out.numel() > 0 else None,
        columnwise_scale_inv=(
            tensor._columnwise_scale_inv.detach().clone()
            if tensor._columnwise_scale_inv is not None
            else None
        ),
        quantizer=tensor._quantizer,
        is_2D_scaled=tensor._is_2D_scaled,
        requires_grad=tensor.requires_grad,
    )
    if out_q._columnwise_data is None and tensor._columnwise_data is not None:
        out_q.update_usage(rowwise_usage=True, columnwise_usage=True)
    return out_q


def fused_relu_backward_fp8(
    saved_rowwise_data: torch.Tensor,
    grad_out: torch.Tensor,
    *,
    input_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Fused raw-FP8 ReLU backward from saved rowwise bytes."""
    if isinstance(grad_out, QuantizedTensor):
        grad_out = grad_out.dequantize(dtype=input_dtype)
    grad_out_f = grad_out.contiguous().to(dtype=torch.float32)
    mod = _get_fused_elementwise_module()
    grad_input = mod.fused_relu_fp8_backward(saved_rowwise_data, grad_out_f)
    if input_dtype is not None and grad_input.dtype != input_dtype:
        grad_input = grad_input.to(input_dtype)
    return grad_input


class Float8BlockQuantizer(Quantizer):
    """Builder class for tensors quantized with current scaling using
    NxN quantization tilings to choose scale.

    This class is typically used to convert a high-precision tensor
    (e.g. in FP32 or BF16) into a quantized tensor (e.g. in FP8).

    """

    dtype: TE_DType
    block_len: int
    amax_epsilon: float
    force_pow_2_scales: bool
    block_scaling_dim: int

    def __init__(
        self,
        fp8_dtype: TE_DType,
        *,
        rowwise: bool,
        columnwise: bool,
        amax_epsilon: float = 0.0,
        force_pow_2_scales: bool = True,
        block_scaling_dim: int = 2,
    ) -> None:
        super().__init__(rowwise=rowwise, columnwise=columnwise)
        self.dtype = fp8_dtype
        self.block_len = 128
        self.force_pow_2_scales = force_pow_2_scales
        self.amax_epsilon = amax_epsilon
        self.block_scaling_dim = block_scaling_dim

    def copy(self) -> Float8BlockQuantizer:
        """Create shallow copy"""

        quantizer = Float8BlockQuantizer(
            fp8_dtype=self.dtype,
            rowwise=self.rowwise_usage,
            columnwise=self.columnwise_usage,
            block_scaling_dim=self.block_scaling_dim,
            amax_epsilon=self.amax_epsilon,
            force_pow_2_scales=self.force_pow_2_scales,
        )
        quantizer.internal = self.internal
        quantizer.optimize_for_gemm = self.optimize_for_gemm

        return quantizer

    def update_quantized(
        self,
        src: torch.Tensor,
        dst: QuantizedTensor,
        *,
        noop_flag: Optional[torch.Tensor] = None,
    ) -> QuantizedTensor:
        """Update the quantized tensor with data from the source tensor.

        This method quantizes the input tensor and stores the result in the destination tensor.

        Parameters
        ----------
        src : torch.Tensor
            Source tensor containing the data to be quantized
        dst : QuantizedTensor
            Destination tensor where the quantized data will be stored
        noop_flag : Optional[torch.Tensor]
            Optional flag tensor indicating whether to skip the quantization operation

        Returns
        -------
        QuantizedTensor
            The destination tensor containing the quantized data

        Raises
        ------
        AssertionError
            If the destination tensor is not a Float8BlockwiseQTensor
        """
        assert isinstance(
            dst, Float8BlockwiseQTensor
        ), f"Cannot store quantized blockwise tensor in {type(dst)} type."
        # Make sure input is in expected format
        if not devices_match(src.device, dst.device):
            src = src.to(device=dst.device)
        if not src.is_contiguous():
            src = src.contiguous()

        # Launch cast kernel
        tex.quantize(src, self, dst, noop_flag)

        dst._fp8_dtype = self.dtype
        return dst

    def quantize_impl(self, tensor: torch.Tensor) -> QuantizedTensor:
        """Quantize tensor implementation"""
        return tex.quantize(tensor, self)

    def get_scale_shape(self, shape: Iterable[int], columnwise: bool) -> Tuple[int, int]:
        """Scaling tensor shape.

        This method determines the shape of the scaling tensor based
        on the quantizer configuration. The scales are padded to
        multiples of 4 for compatibility with GEMM.

        Parameters
        ----------
        shape : Iterable[int]
            Logical tensor shape.
        columnwise : bool
            Whether the data is scaled column-wise (True) or row-wise (False).

        Returns
        -------
        Tuple[int, int]
            Scaling tensor shape.

        """

        # Flatten tensor to 2D
        dim0 = math.prod(shape[:-1])
        dim1 = shape[-1] if shape else 1

        # Check block dims
        if self.block_scaling_dim not in (1, 2):
            raise RuntimeError(
                "Only 1D or 2D blocks are supported, "
                f"but got block_scaling_dim={self.block_scaling_dim}"
            )

        # 128x128 block scaling
        if self.block_scaling_dim == 2:
            scale_dim0 = (dim0 + self.block_len - 1) // self.block_len
            scale_dim1 = (dim1 + self.block_len - 1) // self.block_len
            if columnwise:
                return (scale_dim1, round_up_to_nearest_multiple(scale_dim0, 4))
            return (scale_dim0, round_up_to_nearest_multiple(scale_dim1, 4))

        # 1x128 block scaling
        if columnwise:
            return (
                (dim0 + self.block_len - 1) // self.block_len,
                round_up_to_nearest_multiple(dim1, 4),
            )
        return (
            (dim1 + self.block_len - 1) // self.block_len,
            round_up_to_nearest_multiple(dim0, 4),
        )

    def get_columnwise_shape(self, shape: Iterable[int]) -> Tuple[int, ...]:
        """Column-wise data shape

        GEMMs expect that the column-wise data is transposed relative
        to the logical tensor shape.

        Parameters
        ----------
        shape : Iterable[int]
            Logical tensor shape.

        Returns
        -------
        Tuple[int, ...]
            Column-wise data shape.
        """
        colwise_shape = []
        if shape:
            colwise_shape.append(shape[-1])
        colwise_shape.extend(shape[:-1])
        return tuple(colwise_shape)

    def is_quantizable(self, inp: torch.Tensor) -> bool:
        """Returns whether or not given inp can be quantized"""
        shape = inp.size()
        if len(shape) < 2:
            return False
        if shape[-1] % self.block_len != 0:
            return False
        if math.prod(shape[:-1]) % self.block_len != 0:
            return False
        return True

    def make_empty(
        self,
        shape: Iterable[int],
        *,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        requires_grad: bool = False,
        pin_memory: bool = False,
    ) -> Float8BlockwiseQTensor:
        """Construct quantized tensor with uninitialized data"""

        tensor_kwargs = {
            "device": torch.device("cuda") if device is None else device,
            "pin_memory": pin_memory,
        }

        # Allocate buffers for row-scaled data
        rowwise_data = None
        rowwise_scale_inv = None
        if self.rowwise_usage:
            rowwise_data = torch.empty(shape, dtype=torch.uint8, **tensor_kwargs)
            rowwise_scale_inv = torch.empty(
                self.get_scale_shape(shape, columnwise=False),
                dtype=torch.float32,
                **tensor_kwargs,
            )

        # Allocate buffers for column-scaled data
        columnwise_data = None
        columnwise_scale_inv = None
        if self.columnwise_usage:
            columnwise_data = torch.empty(
                self.get_columnwise_shape(shape),
                dtype=torch.uint8,
                **tensor_kwargs,
            )
            columnwise_scale_inv = torch.empty(
                self.get_scale_shape(shape, columnwise=True),
                dtype=torch.float32,
                **tensor_kwargs,
            )

        # Construct FP8 tensor
        return Float8BlockwiseQTensor(
            shape=shape,
            dtype=dtype,
            fp8_dtype=self.dtype,
            rowwise_data=rowwise_data,
            rowwise_scale_inv=rowwise_scale_inv,
            columnwise_data=columnwise_data,
            columnwise_scale_inv=columnwise_scale_inv,
            quantizer=self,
            is_2D_scaled=self.block_scaling_dim == 2,
            requires_grad=requires_grad,
        )

    def calibrate(self, tensor: torch.Tensor) -> None:
        # NOTE: This interface is specific to requirements like delayed scaling
        # where state from an estimator influences distribution parameters.
        pass

    def _get_compatible_recipe(self) -> Union[type[Recipe], None]:
        return Float8BlockScaling


class Float8BlockwiseQTensor(Float8BlockwiseQTensorStorage, QuantizedTensor):
    """Tensor class with FP8 data quantized via NxN blocks or 1xN blocks.

    The tensor presents as having a standard, higher-precision dtype,
    but the data itself is (scaled) FP8. For most tensor operations,
    the data will be cast to the nominal dtype before performing the
    operation.

    Parameters
    ----------
    rowwise_data : torch.Tensor
          FP8 data in a uint8 tensor matching shape of dequantized tensor.
    rowwise_scale_inv : torch.Tensor
          FP32 dequantization scales in GEMM format for dequantizing rowwise_data.
    columnwise_data : Optional[torch.Tensor]
          FP8 data in a uint8 tensor matching shape of dequantized tensor transpose.
    columnwise_scale_inv : Optional[torch.Tensor]
          FP32 dequantization scales in GEMM format for dequantizing columnwise_data.

    fp8_dtype : transformer_engine_torch.DType, default = kFloat8E4M3
               FP8 format.
    quantizer : Quantizer - the Float8BlockQuantizer that quantized this tensor and
               holds configuration about quantization and dequantization modes.
    """

    # NOTE: We reorder the *args so that we can instantiate a Float8BlockwiseQTensorStorage with positional args,
    # which significantly reduces the Pybind11 overhead when calling the constructor from C++.
    def __new__(
        cls,
        *args,
        rowwise_data: Optional[torch.Tensor],
        rowwise_scale_inv: Optional[torch.Tensor],
        columnwise_data: Optional[torch.Tensor],
        columnwise_scale_inv: Optional[torch.Tensor],
        fp8_dtype: TE_DType,
        quantizer: Quantizer,
        is_2D_scaled: bool,
        **kwargs,
    ):
        instance = super().__new__(
            cls,
            rowwise_data,
            rowwise_scale_inv,
            columnwise_data,
            columnwise_scale_inv,
            fp8_dtype,
            quantizer,
            is_2D_scaled,
            *args,
            **kwargs,
        )

        return instance

    def __repr__(self, *, tensor_contents=None):
        return (
            f"Float8BlockwiseQTensor(fp8_dtype={self._fp8_dtype},"
            f" is_2D_scaled={self._is_2D_scaled},"
            f" data={self.dequantize()})"
        )

    def quantize_(
        self,
        tensor: torch.Tensor,
        *,
        noop_flag: Optional[torch.Tensor] = None,
    ) -> Float8BlockwiseQTensor:
        """Update FP8 data

        Parameters
        ----------
        tensor: torch.Tensor
            Tensor to copy from
        noop_flag: torch.Tensor, optional
            float32 flag indicating whether to avoid performing update

        """
        if isinstance(tensor, QuantizedTensor):
            return self.quantize_(tensor.dequantize())
        return super().quantize_(tensor, noop_flag=noop_flag)

    def dequantize(self, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """
        Construct plain PyTorch tensor from Float8BlockwiseQTensor

        By default the resulting tensor's dtype is the
        Float8BlockwiseQTensor's pre-quantized dtype.
        """
        if dtype is not None:
            dequant_dtype = dtype
        else:
            dequant_dtype = self.dtype
        return _DequantizeFunc.apply(
            self,
            dequant_dtype,
            Float8BlockwiseQTensorStorage.dequantize,
        )

    def detach(self) -> Float8BlockwiseQTensor:
        # pylint: disable=missing-function-docstring
        return Float8BlockwiseQTensor.make_like(self)

    def clone(self) -> Float8BlockwiseQTensor:
        # pylint: disable=missing-function-docstring
        rowwise_data = None
        if self._rowwise_data is not None:
            rowwise_data = self._rowwise_data.detach().clone()
        columnwise_data = None
        if self._columnwise_data is not None:
            columnwise_data = self._columnwise_data.detach().clone()
        return _IdentityFunc.apply(
            self,
            {
                "rowwise_data": rowwise_data,
                "columnwise_data": columnwise_data,
            },
        )

    def view(self, *shape: Tuple[int]) -> Float8BlockwiseQTensor:
        # pylint: disable=missing-function-docstring
        return _ViewFunc.apply(self, shape)

    def reshape(self, *shape: Tuple[int]) -> Float8BlockwiseQTensor:
        # pylint: disable=missing-function-docstring
        return _ReshapeFunc.apply(self, shape)

    def untyped_storage(self) -> torch.UntypedStorage:
        """Return the underlying UntypedStorage of the FP8 data.

        Note that FP8 block-scaled tensor may involve multiple
        buffers: row-wise FP8 data, row-wise scales, column-wise FP8
        data, column-wise scales. The UntypedStorage of the row-wise
        FP8 data is returned if it exists, and otherwise the
        UntypedStorage of the column-wise FP8 data.

        """
        data = self._rowwise_data if self._rowwise_data is not None else self._columnwise_data
        if data is not None:
            return data.untyped_storage()
        return torch.UntypedStorage(0, device=self.device)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        kwargs = kwargs or {}

        # View op
        if func == aten.view.default:
            tensor = args[0]
            data = tensor._rowwise_data
            if data is None:
                # Columnwise data only.
                super().__torch_dispatch__(func, types, args, kwargs)
            orig_size = data.size()
            out_data = data.__torch_dispatch__(
                func,
                types,
                [data] + list(args[1:]),
                kwargs,
            )
            if orig_size != out_data.size():
                raise NotImplementedError(
                    "Changing shape with view not implemented "
                    " (scales and columnwise data untouched)."
                )
            return Float8BlockwiseQTensor.make_like(tensor)

        # as_strided op — applied by FSDP2 on the unsharded param.
        # When shape and strides match (no-op), return self to preserve the quantized type.
        # If shape differs (e.g. padding needed), fall through to dequantize.
        if func == aten.as_strided.default:
            tensor = args[0]
            shape = args[1]
            strides = args[2]
            if (
                len(shape) == len(strides) == 2
                and tuple(strides) == (shape[-1], 1)
                and tuple(shape) == tuple(tensor.size())
            ):
                return Float8BlockwiseQTensor.make_like(tensor)

        # slice op — applied by FSDP2 when shards need unpadding.
        # When the slice is a no-op (covers entire dimension), return self.
        if func == aten.slice.Tensor:
            tensor = args[0]
            dim = args[1]
            start = args[2]
            length = args[3]
            if start == 0 and length == tensor.size(dim):
                return Float8BlockwiseQTensor.make_like(tensor)

        # record stream op
        if func == torch.ops.aten.record_stream.default:
            qt, stream = args
            for t in (
                qt._rowwise_data,
                qt._columnwise_data,
                qt._rowwise_scale_inv,
                qt._columnwise_scale_inv,
            ):
                if t is not None and t.is_cuda:
                    t.record_stream(stream)
            return None

        # Unary elementwise ops that can stay in blockwise FP8.
        if func == aten.sigmoid.default:
            return _SigmoidToFloat8Func.apply(args[0])
        if func == aten.relu.default:
            return _ReluToFloat8Func.apply(args[0])
        if func == aten.mul.Tensor:
            return _MulToFloat8Func.apply(args[0], args[1])

        # Default case
        return super().__torch_dispatch__(func, types, args, kwargs)

    def contiguous(
        self,
        memory_format: torch.memory_format = torch.contiguous_format,
    ) -> Float8BlockwiseQTensor:
        """Returns tensor with data in provided memory format

        Returns `self` if data is already in correct memory format.

        """
        if (
            self._rowwise_data is not None
            and self._rowwise_data.is_contiguous(memory_format=memory_format)
            and (
                (self._columnwise_data is None)
                or (self._columnwise_data.is_contiguous(memory_format=memory_format))
            )
        ):
            return self
        raise ValueError("Float8BlockwiseQTensor does not support different memory formats!")

    @classmethod
    def _make_in_reduce_ex(
        cls,
        shape: torch.Size,
        rowwise_data: torch.Tensor,
        rowwise_scale_inv: torch.Tensor,
        columnwise_data: torch.Tensor,
        columnwise_scale_inv: torch.Tensor,
        fp8_dtype: TE_DType,
        dtype: torch.dtype,
        quantizer: Quantizer,
        is_2D_scaled: bool,
        data_format: Any = None,  # pylint: disable=unused-argument
    ) -> Float8BlockwiseQTensor:
        """Build Float8BlockwiseQTensor, for use in __reduce__

        __reduce_ex__ assumes object constructor has positional
        arguments.

        """
        return Float8BlockwiseQTensor(
            shape=shape,
            rowwise_data=rowwise_data,
            rowwise_scale_inv=rowwise_scale_inv,
            fp8_dtype=fp8_dtype,
            columnwise_data=columnwise_data,
            columnwise_scale_inv=columnwise_scale_inv,
            dtype=dtype,
            quantizer=quantizer,
            is_2D_scaled=is_2D_scaled,
        )

    def __reduce_ex__(self, protocol: int) -> tuple:
        """Custom pickling to remove references to FP8 metadata objects"""
        return (
            Float8BlockwiseQTensor._make_in_reduce_ex,
            (
                self.shape,
                self._rowwise_data,
                self._rowwise_scale_inv,
                self._columnwise_data,
                self._columnwise_scale_inv,
                self._fp8_dtype,
                self.dtype,
                self._quantizer,
                self._is_2D_scaled,
                None,  # data_format
            ),
        )

    def _get_data(self) -> Float8BlockwiseQTensor:
        """Get tensor data property"""
        return self

    @torch.no_grad()
    def _set_data(self, tensor: torch.Tensor) -> None:
        """Set tensor data property

        Just takes FP8 data if setting from a Float8BlockwiseQTensor. Otherwise
        casts to FP8.

        """
        # Tensor device
        new_device = tensor.device if tensor.is_cuda else self.device

        def _set_from_tensor(dst: Float8BlockwiseQTensor, src: Float8BlockwiseQTensor):
            dst._rowwise_data = src._rowwise_data
            dst._columnwise_data = src._columnwise_data
            dst._quantizer = src._quantizer.copy()
            dst._fp8_dtype = src._fp8_dtype
            dst._rowwise_scale_inv = src._rowwise_scale_inv
            dst._columnwise_scale_inv = src._columnwise_scale_inv

        # Check that tensor dimensions match
        if (
            self.size() != tensor.size()
            or self.stride() != tensor.stride()
            or self.layout != tensor.layout
        ):
            raise ValueError("Invalid tensor for updating Float8BlockwiseQTensor data")

        # Just copy FP8 data if other tensor is Float8BlockwiseQTensor
        if (
            isinstance(tensor, Float8BlockwiseQTensor)
            and self.storage_offset() == tensor.storage_offset()
            and devices_match(self.device, new_device)
        ):
            _set_from_tensor(self, tensor)
            return

        if isinstance(tensor, Float8BlockwiseQTensor):
            assert tensor._quantizer is not None, "Can't quantize without a quantizer"
            quantizer = tensor._quantizer
        else:
            assert self._quantizer is not None, "Can't quantize without a quantizer"
            quantizer = self._quantizer

        # Quantize to FP8
        quantizer.update_quantized(tensor, self)

    # Cast to FP8 when setting Float8BlockwiseQTensor.data
    data = property(_get_data, _set_data)

    @property
    def shape(self):
        """Return the shape of the tensor. Define this to avoid expensive PyObject lookups."""
        if self._rowwise_data is not None:
            return self._rowwise_data.shape
        if self._columnwise_data is not None:
            return self._columnwise_data.shape
        return torch.Tensor.size(self)

    @property
    def is_cuda(self):
        """Return whether the tensor is on a CUDA device."""
        if self._rowwise_data is not None:
            return self._rowwise_data.is_cuda
        if self._columnwise_data is not None:
            return self._columnwise_data.is_cuda
        raise RuntimeError("Float8BlockwiseQTensor has no data!")

    def fsdp_pre_all_gather(self, mesh, orig_size, contiguous_orig_stride, module, mp_policy):
        """Called by FSDP2 before all-gather of weights for forward and backward passes.

        Args:
            mesh: DeviceMesh used by FSDP2 to shard the weights.
            orig_size: Original size of the weight tensor.
            contiguous_orig_stride: Original stride of the weight tensor.
            module: FSDP-wrapped module containing this tensor.
            mp_policy: Mixed precision policy used by FSDP2.

        Returns:
            sharded_tensors: Tuple of tensors to be all-gathered.
            metadata: Metadata needed for reconstructing the tensor after all-gather.
        """
        # pylint: disable=unused-argument
        # PyTorch FSDP2 private API – tested with PyTorch 2.5+;
        from torch.distributed.fsdp._fully_shard._fsdp_common import TrainingState
        from transformer_engine.pytorch.distributed import _get_module_fsdp_state

        if not self._is_2D_scaled:
            raise NotImplementedError(
                "FSDP2 is only supported for Float8BlockwiseQTensors with 2D block scaling "
                "(block_scaling_dim=2). 1D block scaling is not supported because the scale "
                "layout has M in dim1, which is incompatible with FSDP2 dim0 all-gather."
            )

        if self._rowwise_data is None or self._rowwise_scale_inv is None:
            raise RuntimeError(
                "Rowwise data must be available for FSDP2 all-gather with 2D block scaling."
            )

        fsdp_state = _get_module_fsdp_state(module)
        param_group = fsdp_state._fsdp_param_group
        if param_group is None:
            raise RuntimeError(
                "FSDP state for this module has no parameter group; "
                "cannot determine reshard_after_forward."
            )
        reshard_after_forward = param_group._reshard_after_forward

        # If weights are resharded after forward pass, only the relevant usage
        # is needed based on whether it's a forward or backward pass.
        # If not resharded, the same all-gathered weights are reused in backward,
        # so both usages may be needed.
        if reshard_after_forward:
            training_state = param_group._training_state
            is_backward_pass = training_state == TrainingState.PRE_BACKWARD
            rowwise_usage = not is_backward_pass
            columnwise_usage = is_backward_pass
        else:
            rowwise_usage = True
            columnwise_usage = self._quantizer.columnwise_usage

        # For 2D block scaling (128x128 blocks), columnwise data and scales are
        # the transpose of rowwise data and scales. Only all-gather the rowwise
        # tensors; columnwise will be derived locally via _create_columnwise()
        # in post_all_gather, halving all-gather communication volume.
        sharded_tensors = (self._rowwise_data, self._rowwise_scale_inv)
        metadata = (self._fp8_dtype, self._is_2D_scaled, rowwise_usage, columnwise_usage)
        return sharded_tensors, metadata

    def fsdp_post_all_gather(
        self,
        all_gather_outputs: Tuple[torch.Tensor, ...],
        metadata: Any,
        param_dtype: torch.dtype,
        *,
        out: Optional[Float8BlockwiseQTensor] = None,
    ):
        """Called by FSDP2 after all-gather of weights for forward and backward passes.

        Args:
            all_gather_outputs: All-gathered tensors from fsdp_pre_all_gather.
            metadata: Metadata from fsdp_pre_all_gather.
            param_dtype: High-precision dtype of the tensor.
            out: Existing tensor to update in-place (None on first iteration).

        Returns:
            Tuple of (Float8BlockwiseQTensor, all_gather_outputs).
        """
        fp8_dtype, is_2D_scaled, rowwise_usage, columnwise_usage = metadata

        # Only rowwise data+scales were all-gathered (columnwise is derived locally).
        rowwise_data, rowwise_scale_inv = all_gather_outputs[:2]
        data_shape = rowwise_data.shape

        if out is not None:
            out._rowwise_data = rowwise_data
            out._rowwise_scale_inv = rowwise_scale_inv
        else:
            out = Float8BlockwiseQTensor(
                shape=data_shape,
                dtype=param_dtype,
                fp8_dtype=fp8_dtype,
                rowwise_data=rowwise_data,
                rowwise_scale_inv=rowwise_scale_inv,
                columnwise_data=None,
                columnwise_scale_inv=None,
                quantizer=self._quantizer,
                is_2D_scaled=is_2D_scaled,
            )

        # For 2D block scaling, derive columnwise data and scales from rowwise
        # via local fp8 transpose.
        if columnwise_usage:
            out._create_columnwise()
        # remove usages if not needed.
        out.update_usage(
            rowwise_usage=rowwise_usage,
            columnwise_usage=columnwise_usage,
        )
        out._quantizer.set_usage(rowwise=rowwise_usage, columnwise=columnwise_usage)
        return out, all_gather_outputs


class _ViewFunc(torch.autograd.Function):
    """View function

    View the Float8BlockwiseQTensor using the provided shape.

    """

    @staticmethod
    def forward(
        ctx,
        tensor: Float8BlockwiseQTensor,
        shape: Optional[list[int]] = None,
    ) -> Float8BlockwiseQTensor:
        # pylint: disable=missing-function-docstring

        # Return input tensor if shape is not provided
        ctx.shape = tensor.shape
        if shape is None:
            return tensor

        # Canonicalize shape
        if not isinstance(shape, Iterable):
            shape = [shape]
        elif len(shape) == 1 and isinstance(shape[0], Iterable):
            shape = shape[0]
        if -1 in shape:
            shape = list(shape)
            d_inferred = -math.prod(ctx.shape) // math.prod(shape)
            for i, d in enumerate(shape):
                if d == -1:
                    shape[i] = d_inferred
                    break

        if tensor._is_2D_scaled:
            # For the case of 2D scaled tensor, the last 2 dimensions should not change
            if shape[-1] != ctx.shape[-1] or shape[-2] != ctx.shape[-2]:
                warnings.warn(
                    "2D scaled Float8BlockwiseQTensor does not support view "
                    "the last 2 dimensions "
                    f"(attempted to view dims={tuple(tensor.shape)} to {tuple(shape)}). "
                    "If you are using this for FSDP2 without compiled_autograd_enabled, "
                    "then ignore this warning since this view is not going to be used anywhere.",
                    stacklevel=2,
                )
                return tensor.dequantize().view(*shape)
        else:
            # For the case of 1D scaled tensor, the last dimension should not change
            if shape[-1] != ctx.shape[-1]:
                warnings.warn(
                    "1D scaled Float8BlockwiseQTensor does not support view "
                    "the last dimension "
                    f"(attempted to view dims={tuple(tensor.shape)} to {tuple(shape)}). "
                    "If you are using this for FSDP2 without compiled_autograd_enabled, "
                    "then ignore this warning since this view is not going to be used anywhere.",
                    stacklevel=2,
                )
                return tensor.dequantize().view(*shape)

        if list(shape) == list(tensor.shape):
            return tensor

        # Construct new tensor if shape is provided
        new_rowwise_data = None
        new_columnwise_data = None
        if tensor._rowwise_data is not None:
            new_rowwise_data = tensor._rowwise_data.view(*shape)
        if tensor._columnwise_data is not None:
            columnwise_shape = [shape[-1]] + list(shape[:-1])
            new_columnwise_data = tensor._columnwise_data.view(columnwise_shape)

        return Float8BlockwiseQTensor(
            shape=shape,
            dtype=tensor.dtype,
            fp8_dtype=tensor._fp8_dtype,
            rowwise_data=new_rowwise_data,
            rowwise_scale_inv=tensor._rowwise_scale_inv,
            columnwise_data=new_columnwise_data,
            columnwise_scale_inv=tensor._columnwise_scale_inv,
            quantizer=tensor._quantizer,
            is_2D_scaled=tensor._is_2D_scaled,
            requires_grad=tensor.requires_grad,
        )

    @staticmethod
    def backward(
        ctx,
        grad: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        # pylint: disable=missing-function-docstring

        if isinstance(grad, Float8BlockwiseQTensor):
            new_data = (
                grad._rowwise_data.view(*ctx.shape) if grad._rowwise_data is not None else None
            )
            if grad._columnwise_data is not None:
                columnwise_shape = [ctx.shape[-1]] + list(ctx.shape[:-1])
                new_columnwise_data = grad._columnwise_data.view(columnwise_shape)
            else:
                new_columnwise_data = None
            dgrad = Float8BlockwiseQTensor(
                shape=ctx.shape,
                dtype=grad.dtype,
                rowwise_data=new_data,
                rowwise_scale_inv=grad._rowwise_scale_inv,
                columnwise_data=new_columnwise_data,
                columnwise_scale_inv=grad._columnwise_scale_inv,
                fp8_dtype=grad._fp8_dtype,
                quantizer=grad._quantizer,
                is_2D_scaled=grad._is_2D_scaled,
                requires_grad=grad.requires_grad,
            )
            return dgrad, None
        return grad.reshape(ctx.shape), None


class _ReshapeFunc(torch.autograd.Function):
    """Reshape function

    Reshape the Float8BlockwiseQTensor using the provided shape.

    """

    @staticmethod
    def forward(
        ctx,
        tensor: Float8BlockwiseQTensor,
        shape: Optional[list[int]] = None,
    ) -> Float8BlockwiseQTensor:
        # pylint: disable=missing-function-docstring

        # Return input tensor if shape is not provided
        ctx.shape = tensor.shape
        if shape is None:
            return tensor

        # Canonicalize shape
        if not isinstance(shape, Iterable):
            shape = [shape]
        elif len(shape) == 1 and isinstance(shape[0], Iterable):
            shape = shape[0]
        if -1 in shape:
            shape = list(shape)
            d_inferred = -math.prod(tensor.shape) // math.prod(shape)
            for i, d in enumerate(shape):
                if d == -1:
                    shape[i] = d_inferred
                    break

        if tensor._is_2D_scaled:
            # For the case of 2D scaled tensor, the last 2 dimensions should not change
            if shape[-1] != ctx.shape[-1] or shape[-2] != ctx.shape[-2]:
                warnings.warn(
                    "2D scaled Float8BlockwiseQTensor does not support reshaping "
                    "the last 2 dimensions "
                    f"(attempted to reshape dims={tuple(tensor.shape)} to {tuple(shape)}). "
                    "If you are using this for FSDP2 without compiled_autograd_enabled, "
                    "then ignore this warning since this view is not going to be used anywhere.",
                    stacklevel=2,
                )
                return tensor.dequantize().reshape(*shape)
        else:
            # For the case of 1D scaled tensor, the last dimension should not change
            if shape[-1] != ctx.shape[-1]:
                warnings.warn(
                    "1D scaled Float8BlockwiseQTensor does not support reshaping "
                    "the last dimension "
                    f"(attempted to reshape dims={tuple(tensor.shape)} to {tuple(shape)}). "
                    "If you are using this for FSDP2 without compiled_autograd_enabled, "
                    "then ignore this warning since this view is not going to be used anywhere.",
                    stacklevel=2,
                )
                return tensor.dequantize().reshape(*shape)
        if list(shape) == list(tensor.shape):
            return tensor

        # Construct new tensor if shape is provided
        new_rowwise_data = None
        new_columnwise_data = None
        if tensor._rowwise_data is not None:
            new_rowwise_data = tensor._rowwise_data.reshape(*shape)
        if tensor._columnwise_data is not None:
            columnwise_shape = [shape[-1]] + list(shape[:-1])
            new_columnwise_data = tensor._columnwise_data.view(columnwise_shape)

        return Float8BlockwiseQTensor(
            shape=shape,
            dtype=tensor.dtype,
            fp8_dtype=tensor._fp8_dtype,
            rowwise_data=new_rowwise_data,
            rowwise_scale_inv=tensor._rowwise_scale_inv,
            columnwise_data=new_columnwise_data,
            columnwise_scale_inv=tensor._columnwise_scale_inv,
            quantizer=tensor._quantizer,
            is_2D_scaled=tensor._is_2D_scaled,
            requires_grad=tensor.requires_grad,
        )

    @staticmethod
    def backward(
        ctx,
        grad: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        # pylint: disable=missing-function-docstring

        if isinstance(grad, Float8BlockwiseQTensor):
            new_rowwise_data = None
            new_columnwise_data = None
            if grad._rowwise_data is not None:
                new_rowwise_data = grad._rowwise_data.view(*ctx.shape)
            if grad._columnwise_data is not None:
                columnwise_shape = [ctx.shape[-1]] + list(ctx.shape[:-1])
                new_columnwise_data = grad._columnwise_data.view(columnwise_shape)
            dgrad = Float8BlockwiseQTensor(
                shape=ctx.shape,
                dtype=grad.dtype,
                rowwise_data=new_rowwise_data,
                rowwise_scale_inv=grad._rowwise_scale_inv,
                columnwise_data=new_columnwise_data,
                columnwise_scale_inv=grad._columnwise_scale_inv,
                fp8_dtype=grad._fp8_dtype,
                quantizer=grad._quantizer,
                is_2D_scaled=grad._is_2D_scaled,
                requires_grad=grad.requires_grad,
            )
            return dgrad, None
        return grad.reshape(ctx.shape), None
