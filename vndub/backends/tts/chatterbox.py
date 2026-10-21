"""Chatterbox Multilingual V3 (Resemble AI, MIT), own ``chatterbox`` env.

Emotion via ``exaggeration`` (mapped from arousal into [0.3, 1.2]);
``cfg_weight=0`` for cross-lingual references (documented accent mitigation).
Outputs carry the Perth watermark.
"""

from vndub.backends.base import BackendSpec, WeightSpec

SPECS = [
    BackendSpec(
        name="chatterbox_ml",
        role="tts",
        env="chatterbox",
        adapter="chatterbox_ml",
        license="MIT (Perth watermark embedded)",
        commercial_ok=True,
        vram_estimate_gb=4.0,
        cpu_ok=True,
        weights=(
            WeightSpec(
                "hf",
                repo="ResembleAI/chatterbox",
                allow=("ve.pt", "s3gen.pt", "t3_mtl23ls_v3.safetensors", "t3_mtl23ls_v2.safetensors", "grapheme_mtl_merged_expanded_v1.json", "conds.pt", "Cangjie5_TC.json", "tokenizer.json"),
                size_gb=3.0,
            ),
        ),
        options={"exaggeration_range": [0.3, 1.2], "cfg_weight": 0.5, "cfg_weight_cross_lingual": 0.0, "temperature": 0.8},
        caps={
            "instruct": "none",
            "tags": "none",
            "knobs": "chatterbox",
            "duration": False,
            "stress_format": "plain",
            "needs_ref_text": False,
            "cross_lingual": True,
            "sr": 24000,
            "langs": ["ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it", "ja", "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh"],
        },
        mock="mock_tts",
        description="Chatterbox Multilingual: light cloner with an exaggeration dial",
        homepage="https://github.com/resemble-ai/chatterbox",
    ),
]
