# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

import torch


@dataclass
class LayerSpec:
    """
    Defines the KV cache specification for a single transformer layer.

    Used to specify the memory requirements and configuration for storing
    key-value pairs in the attention mechanism of a transformer layer.
    """

    name: str
    num_kv_heads: int
    head_size: int
    dtype: torch.dtype
    sliding_window_size: int | None = None
    chunk_size: int | None = None


@dataclass
class StateLayerSpec:
    """
    Defines the recurrent-state cache specification for a single layer.

    Layers that carry state rather than keys and values -- Mamba, Gated DeltaNet and the rest of the
    linear-attention family -- need a shape per state tensor, not a head count and a head size. A layer
    typically has two: a convolution window and a recurrent state, with different dtypes.

    Kept separate from ``LayerSpec`` rather than added to it. Every model in this repository iterates
    ``KVSpec.layers`` and reads ``num_kv_heads``, so a state layer appearing in that list would be read
    as an attention layer with nonsense dimensions.

    Attributes:
        name: The layer's name, as used in the attention metadata and the cache dict.
        shapes: One shape per state tensor, WITHOUT the leading slot axis. The runner prepends it.
        dtypes: One dtype per state tensor, same order and length as ``shapes``.
        state_kind: Which state family this is, as the NAME of a member of vLLM's
            ``MambaAttentionBackendEnum`` (``"MAMBA2"``, ``"GDN_ATTN"``, ...). Required rather than
            defaulted: the field says what the state IS, and on this platform nothing resolves it to a
            backend class, so a wrong value would never surface as an error. An unknown name raises when
            the runner looks it up.
        num_speculative_blocks: Extra slots a speculative step needs to hold a rejected draft's state.
    """

    name: str
    shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[torch.dtype, ...]
    state_kind: str
    num_speculative_blocks: int = 0

    def __post_init__(self) -> None:
        if len(self.shapes) != len(self.dtypes):
            raise ValueError(
                f"{self.name}: {len(self.shapes)} shape(s) but {len(self.dtypes)} dtype(s); each state "
                "tensor needs both, and a mismatch would allocate the wrong number of bytes."
            )
        if not self.shapes:
            raise ValueError(f"{self.name}: a state layer with no state tensors declares nothing")
        for shape in self.shapes:
            if any(dim <= 0 for dim in shape):
                raise ValueError(f"{self.name}: non-positive dimension in state shape {shape}")


@dataclass
class KVSpec:
    """
    Defines the KV cache needs of a model by specifying all layer configurations.

    Contains a list of LayerSpec objects that collectively define the complete
    KV cache requirements for an entire transformer model, plus -- for hybrid models -- the
    recurrent-state layers, which are described differently and allocated differently.
    """

    layers: list[LayerSpec]
    state_layers: list[StateLayerSpec] = field(default_factory=list)

    def __post_init__(self) -> None:
        names = [layer.name for layer in self.layers] + [s.name for s in self.state_layers]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(
                f"duplicate layer name(s) {duplicates}: the cache is keyed by name, so one entry would "
                "silently shadow the other."
            )
