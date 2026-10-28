"""Fun-CosyVoice3-0.5B-2512 (Apache-2.0), ``funasr`` env (CosyVoice repo).

``inference_cross_lingual`` for route A; ``inference_instruct2`` with the style
instruction placed before ``<|endofprompt|>``. Cheapest per-character SFT.
"""

from vndub.backends.base import BackendSpec, WeightSpec

SPECS = [
    BackendSpec(
        name="cosyvoice3",
        role="tts",
        env="funasr",
        adapter="cosyvoice3",
        license="Apache-2.0",
        commercial_ok=True,
        vram_estimate_gb=3.8,
        cpu_ok=True,
        # the whole snapshot: 9.75 GB on the hub (HF tree API: llm.pt 2.0 + llm.rl.pt 2.0 + flow.pt 1.3 + flow.decoder onnx 1.3
        # + two speech_tokenizer onnx 1.0 each + CosyVoice-BlankEN 1.0 + hift/campplus); size_gb feeds the disk pre-check
        weights=(WeightSpec("hf", repo="FunAudioLLM/Fun-CosyVoice3-0.5B-2512", size_gb=9.75),),
        options={"mode": "cross_lingual_instruct", "speed": 1.0, "text_frontend": True},
        caps={
            "instruct": "cosy",
            "tags": "cosy_inline",
            "knobs": "none",
            "duration": False,
            "stress_format": "plain",
            "needs_ref_text": False,
            "cross_lingual": True,
            "lora": True,
            "sr": 24000,
            "langs": ["zh", "en", "ja", "ko", "de", "es", "fr", "it", "ru"],
        },
        mock="mock_tts",
        description="CosyVoice3 0.5B: cross-lingual cloning + instruct (emotion/speed/volume)",
        homepage="https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
    ),
]
