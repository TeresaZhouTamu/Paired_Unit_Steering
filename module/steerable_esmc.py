"""ESMC forward pass with optional layerwise activation steering.
"""

from dataclasses import dataclass
from typing import Sequence

import torch
from esm.utils.structure.affine3d import Affine3D

try:
    from flash_attn.bert_padding import pad_input, unpad_input
except ImportError:
    pad_input = unpad_input = None


@dataclass
class ESMCOutput:
    sequence_logits: torch.Tensor
    embeddings: torch.Tensor | None
    hidden_states: torch.Tensor | None


def steering_forward(
    self,
    x: torch.Tensor,
    sequence_id: torch.Tensor | None = None,
    affine: Affine3D | None = None,
    affine_mask: torch.Tensor | None = None,
    chain_id: torch.Tensor | None = None,
    steering_vectors: Sequence[torch.Tensor] | torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """Run transformer blocks and return final and per-layer states."""
    *batch_dims, _ = x.shape
    if chain_id is None:
        chain_id = torch.ones(batch_dims, dtype=torch.int64, device=x.device)
    if steering_vectors is not None and len(steering_vectors) != len(self.blocks):
        raise ValueError("Expected one steering vector per transformer block")

    hidden_states = []
    for layer, block in enumerate(self.blocks):
        x = block(x, sequence_id, affine, affine_mask, chain_id)
        if steering_vectors is not None:
            shifted = x + steering_vectors[layer]
            old_norm = x.norm(p=2, dim=-1, keepdim=True).detach()
            shifted_norm = shifted.norm(p=2, dim=-1, keepdim=True).detach()
            x = shifted * (old_norm / shifted_norm.clamp_min(1e-12))
        hidden_states.append(x)
    return self.norm(x), x, hidden_states


def esmc_steering_forward(
    self,
    sequence_tokens: torch.Tensor | None = None,
    sequence_id: torch.Tensor | None = None,
    steering_vectors: Sequence[torch.Tensor] | torch.Tensor | None = None,
) -> ESMCOutput:
    """Return logits and hidden states through the ESMC steering hook."""
    if sequence_tokens is None:
        raise ValueError("sequence_tokens is required")
    if sequence_id is None:
        sequence_id = sequence_tokens != self.tokenizer.pad_token_id

    embeddings = self.embed(sequence_tokens)
    batch_size, sequence_length = embeddings.shape[:2]
    indices = None
    if self._use_flash_attn:
        if sequence_id.dtype != torch.bool or sequence_id.shape != (batch_size, sequence_length):
            raise ValueError("Flash Attention needs a [batch, length] boolean mask")
        if unpad_input is None or pad_input is None:
            raise ImportError("Flash Attention is enabled but flash_attn is unavailable")
        embeddings, indices, *_ = unpad_input(embeddings, sequence_id)

    final_state, _, hidden = self.transformer.steering_forward(
        embeddings, sequence_id=sequence_id, steering_vectors=steering_vectors
    )
    if self._use_flash_attn:
        final_state = pad_input(final_state, indices, batch_size, sequence_length)
        hidden = [pad_input(h, indices, batch_size, sequence_length) for h in hidden]

    return ESMCOutput(
        sequence_logits=self.sequence_head(final_state),
        embeddings=final_state,
        hidden_states=torch.stack(hidden, dim=0),  # [layer, batch, token, hidden]
    )
