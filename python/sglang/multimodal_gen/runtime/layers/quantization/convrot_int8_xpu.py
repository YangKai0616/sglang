# SPDX-License-Identifier: Apache-2.0
"""Native BMG ConvRot INT8: SYCL rotation/quantization and SYCL-TLA GEMM.

No comfy_kitchen dispatch or eager arithmetic fallback is used. The FP32
butterfly is not bit-exact with dense-matmul rotation at rounding boundaries.
"""

from __future__ import annotations

import torch
from torch.nn.parameter import Parameter

from sglang.multimodal_gen.runtime.layers.linear import (
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.convrot_int8_config import (
    ConvRotInt8Config,
)
from sglang.multimodal_gen.runtime.utils.weight_attrs import set_weight_attrs


def _load_xpu_kernels() -> None:
    try:
        import sgl_kernel
    except ImportError as exc:
        raise ImportError(
            "convrot_int8 backend xpu requires sgl-kernel-xpu built with "
            "convrot_rotate_quantize and convrot_int8_linear_prequant"
        ) from exc
    for name in ("convrot_rotate_quantize", "convrot_int8_linear_prequant"):
        if not hasattr(sgl_kernel, name) or not hasattr(torch.ops.sgl_kernel, name):
            raise RuntimeError(
                f"convrot_int8 backend xpu requires {name}; rebuild and install "
                "sgl-kernel-xpu with ConvRot INT8 support"
            )
    if not torch.xpu.is_available():
        raise RuntimeError("convrot_int8 backend xpu requires an Intel BMG XPU")
    if tuple(torch.ops.sgl_kernel.query_device(torch.xpu.current_device())) != (2, 0):
        raise RuntimeError("convrot_int8 backend xpu requires an Intel BMG/Xe2 XPU")


class ConvRotInt8XpuLinearMethod(LinearMethodBase):
    """Load or create compatible INT8 weights, then call the two native ops."""

    def __init__(
        self,
        quant_config: ConvRotInt8Config,
        *,
        group_size: int,
        is_checkpoint_serialized: bool,
    ) -> None:
        if group_size != 256:
            raise ValueError("convrot_int8 backend xpu requires group_size 256")
        self.quant_config = quant_config
        self.group_size = group_size
        self.is_checkpoint_serialized = is_checkpoint_serialized
        _load_xpu_kernels()

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        if params_dtype != torch.bfloat16:
            raise ValueError(
                "convrot_int8 backend xpu requires BF16 parameters and activations; "
                "use --dit-precision bf16"
            )
        if not (
            256 <= input_size_per_partition <= 8192
            and input_size_per_partition % self.group_size == 0
        ):
            raise ValueError(
                "convrot_int8 backend xpu requires input_size_per_partition "
                f"in [256, 8192], divisible by 256; got {input_size_per_partition}"
            )
        rows = sum(output_partition_sizes)
        if rows <= 0 or rows % 32:
            raise ValueError(
                "convrot_int8 backend xpu requires a positive output size per "
                f"partition divisible by 32; got {rows}"
            )
        UnquantizedLinearMethod.create_weights(
            self,
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            torch.int8 if self.is_checkpoint_serialized else params_dtype,
            **extra_weight_attrs,
        )
        if self.is_checkpoint_serialized:
            scale = Parameter(
                torch.empty(rows, 1, dtype=torch.float32), requires_grad=False
            )
            set_weight_attrs(scale, {"output_dim": 0})
            set_weight_attrs(scale, extra_weight_attrs)
            layer.register_parameter("weight_scale", scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.is_checkpoint_serialized:
            self.quant_config.note_loaded()
            return
        weight = layer.weight.data
        if weight.dtype == torch.int8:
            return
        from sgl_kernel import convrot_rotate_quantize

        home = weight.device
        # Offloaded layers are quantized one at a time on this worker's XPU.
        device = (
            home
            if home.type == "xpu"
            else torch.device("xpu", torch.xpu.current_device())
        )
        qdata, scale = convrot_rotate_quantize(
            weight.to(device=device, non_blocking=True).contiguous(), self.group_size
        )
        layer.weight = Parameter(qdata.to(home), requires_grad=False)
        layer.register_parameter(
            "weight_scale", Parameter(scale.to(home), requires_grad=False)
        )
        self.quant_config.note_quantized(weight.numel() * weight.element_size())

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from sgl_kernel import convrot_int8_linear_prequant, convrot_rotate_quantize

        if x.device.type != "xpu" or x.dtype != torch.bfloat16:
            raise ValueError(
                "convrot_int8 backend xpu requires BF16 activations on XPU, "
                f"got {x.dtype} on {x.device}"
            )
        if x.ndim < 1 or x.shape[-1] != layer.weight.shape[1]:
            raise ValueError(
                "convrot_int8 backend xpu activation width must match weight K"
            )
        rows = x.reshape(-1, x.shape[-1]).contiguous()
        qdata, scale = convrot_rotate_quantize(rows, self.group_size)
        out = convrot_int8_linear_prequant(
            qdata, layer.weight, scale, layer.weight_scale, bias
        )
        return out.reshape(*x.shape[:-1], out.shape[-1])
