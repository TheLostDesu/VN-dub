"""litagin/anime_speech_emotion_classification (MIT): galgame-domain SER, 10 classes."""

from vndub.backends.base import BackendSpec, WeightSpec

ANIME_LABELS = ["Angry", "Disgusted", "Embarrassed", "Fearful", "Happy", "Sad", "Surprised", "Neutral", "Sexual1", "Sexual2"]

# anime label -> emotion2vec label used for the weighted vote (style_prompt)
ANIME_TO_E2V = {
    "Angry": "angry",
    "Disgusted": "disgusted",
    "Embarrassed": "embarrassed",  # kept as an extra class
    "Fearful": "fearful",
    "Happy": "happy",
    "Sad": "sad",
    "Surprised": "surprised",
    "Neutral": "neutral",
    "Sexual1": "other",
    "Sexual2": "other",
}

SPECS = [
    BackendSpec(
        name="anime_ser",
        role="ser",
        env="hf",
        adapter="anime_ser",
        license="MIT",
        commercial_ok=True,
        vram_estimate_gb=0.8,
        cpu_ok=True,
        weights=(WeightSpec("hf", repo="litagin/anime_speech_emotion_classification", size_gb=0.66),),
        options={"trust_remote_code": True},
        mock="mock_anime_ser",
        description="Galgame-domain emotion classifier (incl. Embarrassed)",
        homepage="https://huggingface.co/litagin/anime_speech_emotion_classification",
    ),
]
