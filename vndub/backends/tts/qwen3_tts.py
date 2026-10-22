"""Qwen3-TTS 12Hz 1.7B (Apache-2.0), ``hf`` env, ``qwen-tts`` package.

* Base: zero-shot cloning from a 3 s clip. Base has no ``instruct``; for
  route A we use ``x_vector_only_mode=True`` (timbre only, no reference
  transcript), which reduces language leakage from the Japanese reference.
* CustomVoice: ``instruct`` (natural-language emotion/tone) with a built-in
  speaker; used for instruct-driven candidates.
* VoiceDesign: voice from a text description; used by ``voicebank`` to design
  route-B performers.
* 0.6B Base: LoRA target for route C on 8 GB.
"""

from vndub.backends.base import BackendSpec, WeightSpec

_HOME = "https://github.com/QwenLM/Qwen3-TTS"
_COMMON = dict(role="tts", env="main", license="Apache-2.0", commercial_ok=True, cpu_ok=True, mock="mock_tts", homepage=_HOME)

SPECS = [
    BackendSpec(
        name="qwen3_tts",
        adapter="qwen3_tts",
        vram_estimate_gb=4.5,
        weights=(
            WeightSpec("hf", repo="Qwen/Qwen3-TTS-12Hz-1.7B-Base", size_gb=4.3),
            WeightSpec("hf", repo="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", size_gb=4.3),
        ),
        options={
            "base_model": "Qwen3-TTS-12Hz-1.7B-Base",
            "instruct_model": "Qwen3-TTS-12Hz-1.7B-CustomVoice",
            "x_vector_only_mode": True,
            "dtype": "bfloat16",
            "max_new_tokens_margin": 1.3,
            "custom_speaker": "auto",
        },
        caps={
            "instruct": "qwen",
            "tags": "none",
            "knobs": "none",
            "duration": False,
            "stress_format": "plain",
            "needs_ref_text": False,
            "cross_lingual": True,
            "langs": ["zh", "en", "ja", "ko", "de", "fr", "ru", "pt", "es", "it"],
            "sr": 24000,
            "lora": False,
        },
        description="Qwen3-TTS 1.7B Base (cloning) + CustomVoice (instruct)",
        **_COMMON,
    ),
    BackendSpec(
        name="qwen3_tts_voicedesign",
        adapter="qwen3_tts_vd",
        vram_estimate_gb=4.5,
        weights=(WeightSpec("hf", repo="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", size_gb=4.3),),
        options={"dtype": "bfloat16"},
        caps={"design": True, "instruct": "qwen", "stress_format": "plain", "sr": 24000, "langs": ["ru", "ja", "en", "zh"]},
        description="Qwen3-TTS VoiceDesign: native target-language performer voices from text",
        **_COMMON,
    ),
    BackendSpec(
        name="qwen3_tts_06b",
        adapter="qwen3_tts",
        vram_estimate_gb=2.5,
        weights=(WeightSpec("hf", repo="Qwen/Qwen3-TTS-12Hz-0.6B-Base", size_gb=1.8),),
        options={"base_model": "Qwen3-TTS-12Hz-0.6B-Base", "instruct_model": "", "x_vector_only_mode": True, "dtype": "bfloat16"},
        caps={"instruct": "none", "stress_format": "plain", "sr": 24000, "lora": True, "langs": ["ru", "ja", "en", "zh"]},
        profiles=("optional",),
        description="Qwen3-TTS 0.6B Base: route-C LoRA fallback target",
        **_COMMON,
    ),
]
