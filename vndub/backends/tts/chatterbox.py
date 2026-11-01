"""Chatterbox Multilingual (Resemble AI, MIT), own ``chatterbox`` env; chatterbox-tts 0.1.7 runs its V2 T3 checkpoint.

Emotion via ``exaggeration`` (mapped from arousal into [0.3, 1.2]);
``cfg_weight=0`` for cross-lingual references (documented accent mitigation).
Russian comes with ruaccent's stress as U+0301 marks: chatterbox-tts wants them from ``russian_text_stresser``, which it does
not depend on, and goes without when that is missing (``synth.backends.chatterbox_ml.stress_format: plain`` asks for that).
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
                # chatterbox-tts 0.1.7 ``from_local(ckpt_dir, device)`` loads t3_mtl23ls_v2 and has no ``t3_model`` argument, so the
                # 2.1 GB v3 checkpoint would only be downloaded to sit unused; measured on the hub (HF tree API): these files sum to 3.21 GB
                allow=("ve.pt", "s3gen.pt", "t3_mtl23ls_v2.safetensors", "grapheme_mtl_merged_expanded_v1.json", "conds.pt", "Cangjie5_TC.json", "tokenizer.json"),
                size_gb=3.2,
            ),
        ),
        options={"exaggeration_range": [0.3, 1.2], "cfg_weight": 0.5, "cfg_weight_cross_lingual": 0.0, "temperature": 0.8},
        caps={
            "instruct": "none",
            "tags": "none",
            "knobs": "chatterbox",
            "duration": False,
            "stress_format": "acute",
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
