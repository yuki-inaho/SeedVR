# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

import torch
import torch.nn.functional as F


def flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q=None,
    max_seqlen_k=None,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    **kwargs,
):
    """SDPA-based drop-in replacement for flash_attn_varlen_func.

    Avoids the prebuilt flash-attn wheel (which needs glibc>=2.32) on this
    glibc-2.31 host. q/k/v are packed [total_tokens, nheads, head_dim] and
    cu_seqlens_* are int offsets [nseq+1]; runs per-segment torch SDPA, which
    is Blackwell (sm_120) native on torch 2.8.
    """
    cu_q = cu_seqlens_q.tolist()
    cu_k = cu_seqlens_k.tolist()
    outputs = []
    for i in range(len(cu_q) - 1):
        qi = q[cu_q[i] : cu_q[i + 1]].transpose(0, 1).unsqueeze(0).contiguous()
        ki = k[cu_k[i] : cu_k[i + 1]].transpose(0, 1).unsqueeze(0).contiguous()
        vi = v[cu_k[i] : cu_k[i + 1]].transpose(0, 1).unsqueeze(0).contiguous()
        oi = F.scaled_dot_product_attention(
            qi, ki, vi, dropout_p=dropout_p, is_causal=causal, scale=softmax_scale
        )
        outputs.append(oi.squeeze(0).transpose(0, 1))
    return torch.cat(outputs, dim=0)

from torch import nn

class TorchAttention(nn.Module):
    def tflops(self, args, kwargs, output) -> float:
        assert len(args) == 0 or len(args) > 2, "query, key should both provided by args / kwargs"
        q = kwargs.get("query") or args[0]
        k = kwargs.get("key") or args[1]
        b, h, sq, d = q.shape
        b, h, sk, d = k.shape
        return b * h * (4 * d * (sq / 1e6) * (sk / 1e6))

    def forward(self, *args, **kwargs):
        return F.scaled_dot_product_attention(*args, **kwargs)


class FlashAttentionVarlen(nn.Module):
    def tflops(self, args, kwargs, output) -> float:
        cu_seqlens_q = kwargs["cu_seqlens_q"]
        cu_seqlens_k = kwargs["cu_seqlens_k"]
        _, h, d = output.shape
        seqlens_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]) / 1e6
        seqlens_k = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]) / 1e6
        return h * (4 * d * (seqlens_q * seqlens_k).sum())

    def forward(self, *args, **kwargs):
        kwargs["deterministic"] = torch.are_deterministic_algorithms_enabled()
        return flash_attn_varlen_func(*args, **kwargs)
