"""FunAudioLLM/SenseVoiceSmall: event tags (Laughter, Cry, Breath, BGM, ...).

Tags are read from the raw output *before* ``rich_transcription_postprocess``.
"""

from vndub.backends.base import BackendSpec, WeightSpec

SENSEVOICE_EVENTS = {
    "<|Laughter|>": "laughter",
    "<|Cry|>": "cry",
    "<|Breath|>": "breath",
    "<|BGM|>": "bgm",
    "<|Cough|>": "cough",
    "<|Sneeze|>": "sneeze",
    "<|Applause|>": "applause",
}

SPECS = [
    BackendSpec(
        name="sensevoice",
        role="events",
        env="funasr",
        adapter="sensevoice",
        license="FunASR model license (maintainers state commercial use OK)",
        commercial_ok=True,
        vram_estimate_gb=1.0,
        cpu_ok=True,
        weights=(WeightSpec("hf", repo="FunAudioLLM/SenseVoiceSmall", size_gb=0.9),),
        options={"language": "ja", "max_sec": 30.0},
        mock="mock_events",
        description="Rich transcription with emotion and audio-event tags",
        homepage="https://huggingface.co/FunAudioLLM/SenseVoiceSmall",
    ),
]
