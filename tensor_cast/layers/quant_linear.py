from typing import Optional

import torch

from ..model_config import (
    LinearQuantConfig,
    LinearQuantType,
    QuantGranularity,
    QuantScheme,
)


class QuantLinearBase(torch.nn.Module):
    """
    A quantized linear layer that replaces a standard torch.nn.Linear layer.
    It handles different quantization types for weights and activations.
    """

    def __init__(self, linear_layer: torch.nn.Linear, quant_config: LinearQuantConfig):
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.quant_config = quant_config

        # Store bias if it exists
        if linear_layer.bias is not None:
            self.register_buffer("bias", linear_layer.bias.clone())
        else:
            self.register_buffer("bias", None)

        # Determine the quantized weight dtype
        if self.quant_config.quant_type in [
            LinearQuantType.W8A16,
            LinearQuantType.W8A8,
            LinearQuantType.W4A8,
        ]:
            # We use int8 to store int4 data, packing two int4 values into one int8
            # This requires careful handling during dequantization.
            q_weight_dtype = torch.int8
        else:
            raise ValueError(f"Unsupported quant_type: {self.quant_config.quant_type}")

        # Quantize the weight
        quantized_weight = self.quantize(
            linear_layer.weight.data,
            self.quant_config.weight_scale,
            self.quant_config.weight_offset,
            q_weight_dtype,
        )

        if self.quant_config.quant_type == LinearQuantType.W4A8:
            # Pack two 4-bit values into a single int8
            quantized_weight = self.pack_int4(
                quantized_weight, dim=self.quant_config.weight_int4_pack_dim
            )

        quantized_weight = quantized_weight.transpose(0, 1).contiguous().transpose(0, 1)
        self.register_buffer("qweight", quantized_weight)

        # Register scales and offsets
        self.register_buffer("weight_scale", quant_config.weight_scale)
        if quant_config.weight_offset is not None:
            self.register_buffer("weight_offset", quant_config.weight_offset)
        else:
            self.register_buffer("weight_offset", None)

        if quant_config.activation_scale is not None:
            self.register_buffer("activation_scale", quant_config.activation_scale)
        else:
            self.register_buffer("activation_scale", None)

        if quant_config.activation_offset is not None:
            self.register_buffer("activation_offset", quant_config.activation_offset)
        else:
            self.register_buffer("activation_offset", None)

    def quantize(
        self,
        tensor: torch.Tensor,
        scale: torch.Tensor,
        offset: Optional[torch.Tensor],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Generic quantization function."""
        if offset is not None:
            tensor = tensor / scale + offset
        else:  # Symmetric quantization
            tensor = tensor / scale
        return tensor.round().to(dtype)

    def dequantize_weight(self) -> torch.Tensor:
        """Dequantizes the weight tensor."""
        if self.quant_config.quant_type == LinearQuantType.W4A8:
            unpacked_qweight = self.unpack_int4(
                self.qweight, self.quant_config.weight_int4_pack_dim
            )
        else:
            unpacked_qweight = self.qweight

        weight = unpacked_qweight.to(self.weight_scale.dtype)
        if self.weight_offset is not None:
            return (weight - self.weight_offset) * self.weight_scale
        return weight * self.weight_scale

    def pack_int4(self, tensor: torch.Tensor, dim: int = 1) -> torch.Tensor:
        """Packs a tensor of int8 values (where each value is in [-8, 7]) into an int8 tensor."""
        # Ensure values are in the correct range for int4
        tensor = tensor.clamp(-8, 7)
        if dim == 1:
            # Shift to be non-negative for bitwise operations
            high_bits = (tensor[:, ::2] + 8).to(torch.uint8)
            low_bits = (tensor[:, 1::2] + 8).to(torch.uint8)
        else:
            if dim != 0:
                raise ValueError(f"Unsupported dimension for int4 packing: {dim}")
            # Shift to be non-negative for bitwise operations
            high_bits = (tensor[::2, :] + 8).to(torch.uint8)
            low_bits = (tensor[1::2, :] + 8).to(torch.uint8)

        return (high_bits << 4) | low_bits

    def unpack_int4(self, packed_tensor: torch.Tensor, dim: int = 1) -> torch.Tensor:
        """Unpacks an int8 tensor into two int4 tensors (represented as int8)."""
        high_bits = (packed_tensor >> 4).to(torch.int8) - 8
        low_bits = (packed_tensor & 0x0F).to(torch.int8) - 8
        # Interleave the unpacked values
        unpacked = torch.empty(
            self.out_features,
            self.in_features,
            dtype=torch.int8,
            device=packed_tensor.device,
        )
        if dim == 1:
            unpacked[:, ::2] = high_bits
            unpacked[:, 1::2] = low_bits
        else:
            if dim != 0:
                raise ValueError(f"Unsupported dimension for int4 unpacking: {dim}")
            unpacked[::2, :] = high_bits
            unpacked[1::2, :] = low_bits
        return unpacked

    def _calculate_dynamic_qparams(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculates scale and offset for dynamic activation quantization."""
        x = x.reshape(-1, x.shape[-1])
        cfg = self.quant_config
        if cfg.dynamic_quant_dtype == torch.int8:
            q_min, q_max = -128.0, 127.0
        else:
            raise ValueError(
                f"Unsupported dynamic quant dtype: {cfg.dynamic_quant_dtype}"
            )

        # Determine reduction dimensions based on granularity
        if cfg.dynamic_quant_granularity == QuantGranularity.PER_TENSOR:
            dim = None  # Reduce over all dims
        elif cfg.dynamic_quant_granularity == QuantGranularity.PER_SAMPLE:
            dim = tuple(range(1, x.ndim))  # Reduce over all dims except batch
        else:
            raise ValueError(f"Unknown granularity: {cfg.dynamic_quant_granularity}")

        min_val = torch.amin(x, dim=dim)
        max_val = torch.amax(x, dim=dim)

        if cfg.dynamic_quant_scheme == QuantScheme.SYMMETRIC:
            max_abs = torch.maximum(torch.abs(min_val), torch.abs(max_val))
            scale = max_abs / q_max
            offset = torch.zeros_like(scale)
        elif cfg.dynamic_quant_scheme == QuantScheme.ASYMMETRIC:
            scale = (max_val - min_val) / (q_max - q_min)
            offset = q_min - min_val / scale
        else:
            raise ValueError(f"Unknown scheme: {cfg.dynamic_quant_scheme}")

        scale = torch.where(scale == 0, 1.0, scale)  # Avoid division by zero
        return scale, offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the quantized linear layer.
        """
        # Dequantize weights
        dequantized_weight = self.dequantize_weight().to(x.dtype)

        # Only quantize activations for W8A8 or W4A8 types
        if self.quant_config.quant_type in [LinearQuantType.W8A8, LinearQuantType.W4A8]:
            # Use pre-computed static parameters if available
            if self.activation_scale is not None:
                act_scale = self.activation_scale
                act_offset = self.activation_offset
            # Otherwise, compute parameters dynamically
            else:
                act_scale, act_offset = self._calculate_dynamic_qparams(x)

            q_x = torch.round(x / act_scale + act_offset)
            x = (q_x - act_offset) * act_scale

        # Perform linear operation
        output = torch.nn.functional.linear(x, dequantized_weight, self.bias)

        out_dtype = (
            self.quant_config.out_dtype if self.quant_config is not None else x.dtype
        )
        return output.to(out_dtype)

    def __repr__(self):
        return (
            f"QuantLinear(in_features={self.in_features}, out_features={self.out_features}, "
            f"quant_type={self.quant_config.quant_type})"
        )


class TensorCastQuantLinear(QuantLinearBase):
    def __init__(self, linear_layer: torch.nn.Linear, quant_config: LinearQuantConfig):
        super().__init__(linear_layer, quant_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the quantized linear operation using custom tensor_cast ops.

        This method selects the appropriate custom operator based on the
        availability of static activation quantization parameters.
        """
        x_shape = x.shape
        x = x.reshape(-1, x_shape[-1])
        qweight = self.qweight.transpose(0, 1)
        out_dtype = (
            self.quant_config.out_dtype
            if self.quant_config.out_dtype is not None
            else x.dtype
        )
        if self.activation_scale is not None:
            op = (
                torch.ops.tensor_cast.static_quant_linear_int4
                if self.quant_config.quant_type == LinearQuantType.W4A8
                else torch.ops.tensor_cast.static_quant_linear
            )
            if self.activation_scale is not None:
                # TODO: perhaps we can consider to explicitly model dynamic quant outside the quant linear op
                #       so that we can fuse this quant op with previous ops too.
                x = torch.ops.tensor_cast.quantize(
                    x,
                    self.activation_scale,
                    self.activation_offset,
                    out_dtype=torch.int8,
                )
            out = op(
                x,
                qweight,
                self.weight_scale,
                w_offset=self.weight_offset,
                x_scale=self.activation_scale,
                x_offset=self.activation_offset,
                bias=self.bias,
                out_dtype=out_dtype,
            )
        else:
            op = (
                torch.ops.tensor_cast.dynamic_quant_linear_int4
                if self.quant_config.quant_type == LinearQuantType.W4A8
                else torch.ops.tensor_cast.dynamic_quant_linear
            )
            out = op(
                x,
                qweight,
                self.weight_scale,
                w_offset=self.weight_offset,
                bias=self.bias,
                out_dtype=out_dtype,
            )
        return out.reshape(*x_shape[:-1], out.shape[-1]).to(out_dtype)
