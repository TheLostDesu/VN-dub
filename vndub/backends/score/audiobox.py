"""Audiobox Aesthetics (Meta, CC-BY-4.0): CE / CU / PC / PQ; PQ enters the composite."""

from vndub.backends.base import BackendSpec, WeightSpec

SPECS = [
    BackendSpec(
        name="audiobox",
        role="aesthetics",
        env="score",
        adapter="audiobox",
        license="CC-BY-4.0",
        commercial_ok=True,
        vram_estimate_gb=1.0,
        cpu_ok=True,
        weights=(WeightSpec("hf", repo="facebook/audiobox-aesthetics", size_gb=0.5),),
        mock="mock_aesthetics",
        description="Technical quality (PQ) and enjoyment (CE) predictor",
        homepage="https://github.com/facebookresearch/audiobox-aesthetics",
    ),
]
