from .market    import BemeMarket
from .utils     import MultiLabelWrapper
from .autobeme  import AutoBEME
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
    "AutoBEME",
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
