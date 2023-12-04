"""Multi-head attention."""
from typing import List, Optional, Dict, Any

import torch
import torch.nn as nn
from xformers import ops as xops
from xformers.ops.fmha.attn_bias import (BlockDiagonalCausalMask,
                                         LowerTriangularMaskWithTensorBias)

from vllm._C import ops
from vllm._C import cache_ops
from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.rotary_embedding import (
    DynamicNTKScalingRotaryEmbedding, LinearScalingRotaryEmbedding,
    RotaryEmbedding)

_SUPPORTED_HEAD_SIZES = [64, 80, 96, 112, 128, 256]
# Should be the same as PARTITION_SIZE in `paged_attention_v2_launcher`.
_PARTITION_SIZE = 512


class PagedAttention(nn.Module):
    """MHA/MQA/GQA layer with PagedAttention.

    This class takes query, key, and value tensors as input. The input tensors
    can either contain prompt tokens or generation tokens.
    The class does the following:

    1. Wait for the cache operations (e.g., swap, copy) to finish. The cache
        operations are issued by the cache engine before executing the forward
        pass of the model, and they are executed asynchronously.
    2. Reshape and store the input key and value tensors in the KV cache.
    3. Perform (multi-head/multi-query/grouped-query) attention using either
        xformers or the PagedAttention custom op.
    4. Return the output tensor.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        sliding_window: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.sliding_window = sliding_window
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.register_buffer("alibi_slopes", alibi_slopes, persistent=False)

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.head_mapping = torch.repeat_interleave(
            torch.arange(self.num_kv_heads, dtype=torch.int32, device="cuda"),
            self.num_queries_per_kv)

        if self.head_size not in _SUPPORTED_HEAD_SIZES:
            raise ValueError(f"head_size ({self.head_size}) is not supported. "
                             f"Supported head sizes: {_SUPPORTED_HEAD_SIZES}.")

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: Optional[torch.Tensor],
        value_cache: Optional[torch.Tensor],
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        """PagedAttention forward pass.

        Args:
            query: shape = [batch_size, seq_len, num_heads * head_size]
            key: shape = [batch_size, seq_len, num_kv_heads * head_size]
            value: shape = [batch_size, num_kv_heads * head_size]
            key_cache: shape = [num_blocks, num_kv_heads, head_size/x,
                block_size, x]
            value_cache: shape = [num_blocks, num_kv_heads, head_size,
                block_size]
            input_metadata: metadata for the inputs.
            cache_event: event to wait for the cache operations to finish.
        Returns:
            shape = [batch_size, seq_len, num_heads * head_size]
        """
        batch_size, seq_len, hidden_size = query.shape
        # Reshape the query, key, and value tensors.
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)
        slot_mapping = input_metadata.slot_mapping.flatten()

        if cache_event is not None:
            cache_event.wait()

        # Reshape the keys and values and store them in the cache.
        # If key_cache and value_cache are not provided, the new key and value
        # vectors will not be cached. This happens during the initial memory
        # profiling run.
        if key_cache is not None and value_cache is not None:
            cache_ops.reshape_and_cache(
                key,
                value,
                key_cache,
                value_cache,
                slot_mapping,
            )

        if input_metadata.is_prompt:
            # Prompt run.
            if self.num_kv_heads != self.num_heads:
                # As of Nov 2023, xformers only supports MHA. For MQA/GQA,
                # project the key and value tensors to the desired number of
                # heads.
                # TODO(woosuk): Use MQA/GQA kernels for higher performance.
                query = query.view(query.shape[0], self.num_kv_heads,
                                   self.num_queries_per_kv, query.shape[-1])
                key = key[:, :,
                          None, :].expand(key.shape[0], self.num_kv_heads,
                                          self.num_queries_per_kv,
                                          key.shape[-1])
                value = value[:, :, None, :].expand(value.shape[0],
                                                    self.num_kv_heads,
                                                    self.num_queries_per_kv,
                                                    value.shape[-1])

            # Set attention bias if not provided. This typically happens at the
            # very attention layer of every iteration.
            # FIXME(woosuk): This is a hack.
            if input_metadata.attn_bias is None:
                if self.alibi_slopes is None:
                    attn_bias = BlockDiagonalCausalMask.from_seqlens(
                        [seq_len] * batch_size)
                    if self.sliding_window is not None:
                        attn_bias = attn_bias.make_local_attention(
                            self.sliding_window)
                    input_metadata.attn_bias = attn_bias
                else:
                    input_metadata.attn_bias = _make_alibi_bias(
                        self.alibi_slopes, batch_size, seq_len, query.dtype)

            # TODO(woosuk): Too many view operations. Let's try to reduce them
            # in the future for code readability.
            if self.alibi_slopes is None:
                query = query.unsqueeze(0)
                key = key.unsqueeze(0)
                value = value.unsqueeze(0)
            else:
                query = query.unflatten(0, (batch_size, seq_len))
                key = key.unflatten(0, (batch_size, seq_len))
                value = value.unflatten(0, (batch_size, seq_len))

            out = xops.memory_efficient_attention_forward(
                query,
                key,
                value,
                attn_bias=input_metadata.attn_bias,
                p=0.0,
                scale=self.scale,
            )
            output = out.view_as(query)
        else:
            # Decoding run.
            output = _paged_attention(
                query,
                key_cache,
                value_cache,
                input_metadata,
                self.head_mapping,
                self.scale,
                self.alibi_slopes,
            )

        # Reshape the output tensor.
        return output.view(batch_size, seq_len, hidden_size)


def _make_alibi_bias(
    alibi_slopes: torch.Tensor,
    batch_size: int,
    seq_len: int,
    dtype: torch.dtype,
) -> LowerTriangularMaskWithTensorBias:
    bias = torch.arange(seq_len, dtype=dtype)
    # NOTE(zhuohan): HF uses
    #     `bias = bias[None, :].repeat(prompt_len, 1)`
    # here. We find that both biases give the same results, but
    # the bias below more accurately follows the original ALiBi
    # paper.
    bias = bias[None, :] - bias[:, None]
    bias = bias.to(alibi_slopes.device)

    # When using custom attention bias, xformers requires the bias to
    # be sliced from a tensor whose length is a multiple of 8.
    padded_len = (seq_len + 7) // 8 * 8
    bias = torch.empty(
        batch_size,
        alibi_slopes.shape[0],
        seq_len,
        padded_len,
        device=alibi_slopes.device,
        dtype=dtype,
    )[:, :, :, :seq_len].copy_(bias)
    bias.mul_(alibi_slopes[:, None, None])
    attn_bias = LowerTriangularMaskWithTensorBias(bias)
    return attn_bias


def _paged_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    input_metadata: InputMetadata,
    head_mapping: torch.Tensor,
    scale: float,
    alibi_slopes: Optional[torch.Tensor],
) -> torch.Tensor:
    output = torch.empty_like(query)

    block_size = value_cache.shape[3]
    num_seqs, num_heads, head_size = query.shape
    max_num_partitions = (
        (input_metadata.max_context_len + _PARTITION_SIZE - 1) //
        _PARTITION_SIZE)
    # NOTE(woosuk): We use a simple heuristic to decide whether to use
    # PagedAttention V1 or V2. If the number of partitions is 1, we use
    # V1 to avoid the overhead of reduction. Also, if the number of
    # sequences or heads is large, we use V1 since there is enough work
    # to parallelize.
    # TODO(woosuk): Tune this heuristic.
    # For context len > 8192, use V2 kernel to avoid shared memory shortage.
    use_v1 = input_metadata.max_context_len <= 8192 and (
        max_num_partitions == 1 or num_seqs * num_heads > 512)
    if use_v1:
        # Run PagedAttention V1.
        ops.paged_attention_v1(
            output,
            query,
            key_cache,
            value_cache,
            head_mapping,
            scale,
            input_metadata.block_tables,
            input_metadata.context_lens,
            block_size,
            input_metadata.max_context_len,
            alibi_slopes,
        )
    else:
        # Run PagedAttention V2.
        assert _PARTITION_SIZE % block_size == 0
        tmp_output = torch.empty(
            size=(num_seqs, num_heads, max_num_partitions, head_size),
            dtype=output.dtype,
            device=output.device,
        )
        exp_sums = torch.empty(
            size=(num_seqs, num_heads, max_num_partitions),
            dtype=torch.float32,
            device=output.device,
        )
        max_logits = torch.empty_like(exp_sums)
        ops.paged_attention_v2(
            output,
            exp_sums,
            max_logits,
            tmp_output,
            query,
            key_cache,
            value_cache,
            head_mapping,
            scale,
            input_metadata.block_tables,
            input_metadata.context_lens,
            block_size,
            input_metadata.max_context_len,
            alibi_slopes,
        )
    return output

class PagedAttentionWithRoPE(PagedAttention):
    """PagedAttention with rotary positional embedding."""
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        rotary_dim: int,
        max_position: int = 8192,
        base: int = 10000,
        num_kv_heads: Optional[int] = None,
        is_neox_style: bool = True,
        rope_scaling: Optional[Dict[str, Any]] = None,
        sliding_window: Optional[int] = None,
    ) -> None:
        super().__init__(num_heads,
                         head_size,
                         scale,
                         num_kv_heads,
                         sliding_window=sliding_window)
        if rope_scaling is None:
            self.rotary_emb = RotaryEmbedding(head_size, rotary_dim,
                                              max_position, base,
                                              is_neox_style)
            #self.rotary_emb = RotaryEmbedding(head_size, rotary_dim,
            #                                  max_position, base,
            #                                  is_neox_style)
            self.rotary_emb = DynamicNTKScalingRotaryEmbedding(head_size, rotary_dim,
                                  max_position, base, is_neox_style, 2)
        else:
            scaling_type = rope_scaling["type"]
            scaling_factor = rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LinearScalingRotaryEmbedding(
                    head_size, rotary_dim, max_position, base, is_neox_style,
                    scaling_factor)
            elif scaling_type == "dynamic":
                self.rotary_emb = DynamicNTKScalingRotaryEmbedding(
                    head_size, rotary_dim, max_position, base, is_neox_style,
                    scaling_factor)
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")
    def forward(
        self,
        positions: torch.Tensor,
        input_true_seq_len: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        """ PagedAttention forward pass with rotary embedding.
        Args:
            positions: shape = [batch_size, seq_len]
            query: shape = [batch_size, seq_len, num_heads * head_size]
            key: shape = [batch_size, seq_len, num_kv_heads * head_size]
            value: shape = [batch_size, seq_len, num_kv_heads * head_size]
            key_cache: shape = [num_blocks, num_kv_heads, head_size/x,
                block_size, x]
            value_cache: shape = [num_blocks, num_kv_heads, head_size,
                block_size]
            input_metadata: metadata for paged attention.
            cache_event: event to wait for the cache operations to finish.
        Returns:
            shape = [batch_size, seq_len, num_heads * head_size]
        """

        # Apply rotary embedding to the query and key before passing them
        # to the attention op.
        query, key = self.rotary_emb(positions, input_true_seq_len, query, key)
        return super().forward(
            query,
            key,
            value,
            key_cache,
            value_cache,
            input_metadata,
            cache_event,
        )
class PagedAttentionWithALiBi(PagedAttention):
    """PagedAttention with ALiBi attention bias."""
    def __init__(self,
                 num_heads: int,
                 head_size: int,
                 scale: float,
                 slopes: List[float],
                 num_kv_heads: Optional[int] = None) -> None:
        super().__init__(num_heads, head_size, scale, num_kv_heads)
        assert len(slopes) == num_heads
        slopes = torch.tensor(slopes, dtype=torch.float32)
        self.register_buffer("alibi_slopes", slopes, persistent=False)
    def set_attn_bias(self, input_metadata: InputMetadata,
                      dtype: torch.dtype) -> None:
        if input_metadata.attn_bias is not None:
            # Already set by a previous layer.
            return
        # Generates ALiBi mask based on the max prompt length.
        max_prompt_len = input_metadata.max_prompt_len
        bias = torch.arange(max_prompt_len, dtype=dtype)
        # NOTE(zhuohan): HF uses
        #     `bias = bias[None, :].repeat(prompt_len, 1)`
        # here. We find that both biases give the same results, but
        # the bias below more accurately follows the original ALiBi
        # paper.
        bias = bias[None, :] - bias[:, None]
        bias = bias.to(self.alibi_slopes.device)
        # When using custom attention bias, xformers requires the bias to
        # be sliced from a tensor whose length is a multiple of 8.
        padded_len = (max_prompt_len + 7) // 8 * 8
        bias = torch.empty(
            input_metadata.num_prompts,
            self.num_heads,
            max_prompt_len,
            padded_len,
            device=self.alibi_slopes.device,
            dtype=dtype,
        )[:, :, :, :max_prompt_len].copy_(bias)
        bias.mul_(self.alibi_slopes[:, None, None])
        attn_bias = LowerTriangularMaskWithTensorBias(bias)
        input_metadata.attn_bias = attn_bias
    def multi_query_kv_attention(
        self,
        output: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        """Attention with ALiBi bias for the prompt tokens.
        Args:
            output: shape = [num_prompt_tokens, num_heads, head_size]
            query: shape = [num_prompt_tokens, num_heads, head_size]
            key: shape = [num_prompt_tokens, num_kv_heads, head_size]
            value: shape = [num_prompt_tokens, num_kv_heads, head_size]
            input_metadata: metadata for paged attention.
        """
        if self.num_kv_heads != self.num_heads:
            # Project the key and value tensors to the desired number of heads.
            key = torch.repeat_interleave(key, self.num_queries_per_kv, dim=1)
            value = torch.repeat_interleave(value,
                                            self.num_queries_per_kv,
                                            dim=1)
        batch_size = input_metadata.num_prompts
        seq_len = input_metadata.max_prompt_len
        out = xops.memory_efficient_attention_forward(
            query.view(batch_size, seq_len, self.num_heads, self.head_size),
            key.view(batch_size, seq_len, self.num_heads, self.head_size),
            value.view(batch_size, seq_len, self.num_heads, self.head_size),
            attn_bias=input_metadata.attn_bias,
            p=0.0,
            scale=self.scale,
        )
        # TODO(woosuk): Unnecessary copy. Optimize.
        output.copy_(out.view(-1, self.num_heads, self.head_size))
        return output
    def get_alibi_slopes(self) -> Optional[torch.Tensor]:
        return self.alibi_slopes