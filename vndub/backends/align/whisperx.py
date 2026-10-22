"""WhisperX-style CTC alignment with jonatasgrosman/wav2vec2-large-xlsr-53-japanese.

The adapter uses ``whisperx.align`` when whisperx is installed in the env and
otherwise the same algorithm via ``torchaudio.functional.forced_align``
(character CTC over the NFKC-normalized script text).
"""

from vndub.backends.base import BackendSpec, WeightSpec

SPECS = [
    BackendSpec(
        name="whisperx_ja",
        role="align",
        env="main",
        adapter="ctc_align",
        license="Apache-2.0 (model) / BSD-2 (whisperx)",
        commercial_ok=True,
        vram_estimate_gb=1.3,
        cpu_ok=True,
        weights=(WeightSpec("hf", repo="jonatasgrosman/wav2vec2-large-xlsr-53-japanese", size_gb=1.3),),
        options={"language": "ja"},  # a Japanese CTC vocabulary: the stages skip it for voices in another language
        mock="mock_aligner",
        description="Character CTC forced alignment (fallback aligner)",
        homepage="https://github.com/m-bain/whisperX",
    ),
]
