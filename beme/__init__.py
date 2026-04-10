from .market    import BemeMarket
from .utils     import MultiLabelWrapper
from .templates import (
    DecayMarket,
    AlphaMarket,
    FutureValueMarket,
    ConfidenceMarket,
    AdaptiveEvolutionMarket,
    PruningMarket,
    HybridEngineMarket,
)

__all__ = [
    "BemeMarket",
    "MultiLabelWrapper",
    # Templates
    "DecayMarket",
    "AlphaMarket",
    "FutureValueMarket",
    "ConfidenceMarket",
    "AdaptiveEvolutionMarket",
    "PruningMarket",
    "HybridEngineMarket",
]
__version__ = "1.0.0"
