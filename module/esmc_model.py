"""ESMC loading and representations shared by extraction and optimization.
"""

from __future__ import annotations

import importlib
import types

import numpy as np


class ESMCModel:
    """Small adapter around the ESMC interface used in the notebooks."""

    def __init__(self, model_name: str = "esmc_600m", device: str = "auto"):
        try:
            import torch
            from esm.models.esmc import ESMC
            from esm.tokenization import get_esmc_model_tokenizers
        except ImportError as exc:
            raise RuntimeError(
                "Install torch and an esm version exposing ESMC and "
                "get_esmc_model_tokenizers"
            ) from exc

        self.torch = torch
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available()
            else "cpu" if device == "auto" else device
        )
        self.tokenizer = get_esmc_model_tokenizers()
        self.model = ESMC.from_pretrained(model_name, device=self.device)
        self.model = self.model.to(torch.float32).to(self.device).eval()

    def encode_sequence(self, sequence: str):
        raw = self.tokenizer.encode(sequence, add_special_tokens=False)
        bos = getattr(self.tokenizer, "bos_token_id", 0) or 0
        eos = getattr(self.tokenizer, "eos_token_id", 2) or 2
        return self.torch.tensor(
            [bos, *raw, eos], device=self.device, dtype=self.torch.long
        )

    def mean_representations(self, sequence: str) -> np.ndarray:
        torch = self.torch
        batch = self.tokenizer.batch_encode_plus(
            [list(sequence)], add_special_tokens=True, padding=True,
            is_split_into_words=True, return_tensors="pt",
        )
        tokens = batch["input_ids"].to(self.device)
        mask = batch["attention_mask"].to(self.device)
        with torch.no_grad():
            states = self.model(sequence_tokens=tokens, sequence_id=mask).hidden_states
        length = int(mask[0].sum())
        if length <= 2:
            raise ValueError("ESMC encoding contains no residue tokens")
        return states[:, 0, 1:length - 1].mean(dim=1).cpu().numpy().astype(np.float32)

    def residue_representations(self, tokens):
        torch = self.torch
        with torch.no_grad():
            states = self.model(
                sequence_tokens=tokens.unsqueeze(0),
                sequence_id=torch.ones_like(tokens).unsqueeze(0),
            ).hidden_states
        return states[:, 0, 1:-1]

    def enable_steering(self, module_name: str = "module.steerable_esmc") -> None:
        try:
            steering = importlib.import_module(module_name)
            transformer_forward = steering.steering_forward
            model_forward = steering.esmc_steering_forward
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                f"Cannot load steering module {module_name!r}; run from the "
                "repository root or pass an importable module name"
            ) from exc
        self.model.transformer.steering_forward = types.MethodType(
            transformer_forward, self.model.transformer
        )
        self.model.steering_forward = types.MethodType(model_forward, self.model)
