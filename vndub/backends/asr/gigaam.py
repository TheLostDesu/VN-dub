"""GigaAM-v3 e2e CTC (ai-sage, MIT): Russian ASR used as the CER oracle."""

from vndub.backends.base import BackendSpec, WeightSpec

SPECS = [
    BackendSpec(
        name="gigaam_v3",
        role="asr",
        env="asr",
        adapter="gigaam",
        license="MIT",
        commercial_ok=True,
        vram_estimate_gb=1.2,
        cpu_ok=True,
        weights=(WeightSpec("hf", repo="ai-sage/GigaAM-v3", revision="e2e_ctc", subdir="GigaAM-v3-e2e_ctc", size_gb=0.45),),
        options={"revision": "e2e_ctc", "language": "ru", "max_sec": 25.0},
        fallback="whisper_ru",
        mock="mock_asr",
        description="Russian ASR (e2e CTC with punctuation/normalization); CER of dubbed candidates",
        homepage="https://huggingface.co/ai-sage/GigaAM-v3",
    ),
]
