"""RVC v2 through Applio (MIT), ``rvc`` env.

Training: Applio CLI (preprocess -> extract (rmvpe) -> train -> index) on the
character's clean clips, selected for emotion coverage. Inference keeps the
performer's F0 contour and timing; ``f0_transpose`` shifts the performer's
median pitch to the character's. The Applio checkout lives in
``.venvs/rvc/src/Applio`` (created by ``scripts/make_envs``) or at
``tools.yaml: applio_dir``.
"""

from vndub.backends.base import BackendSpec, WeightSpec

SPECS = [
    BackendSpec(
        name="rvc",
        role="vc",
        env="rvc",
        adapter="applio",
        license="MIT",
        commercial_ok=True,
        vram_estimate_gb=2.0,
        cpu_ok=True,
        weights=(WeightSpec("env", subdir="applio_prerequisites", post="applio_prerequisites", size_gb=1.5),),
        options={"index_rate": 0.5, "protect": 0.33, "f0_method": "rmvpe", "train_vram_gb": 6.0, "timeout_item_s": 600},
        caps={"trainable": True, "zero_shot": False, "keeps_prosody": True},
        mock="mock_vc",
        description="Per-character timbre model; primary VC for route B",
        homepage="https://github.com/IAHispano/Applio",
    ),
    BackendSpec(
        name="rmvpe",
        role="f0",
        env="rvc",
        adapter="rmvpe",
        license="Apache-2.0 (RMVPE) / MIT (Applio)",
        commercial_ok=True,
        vram_estimate_gb=0.6,
        cpu_ok=True,
        weights=(WeightSpec("env", subdir="applio_prerequisites", post="applio_prerequisites", size_gb=1.5),),
        fallback="torchcrepe",
        mock="librosa_f0",
        description="RMVPE F0 extractor (robust to BGM), from the RVC ecosystem",
        homepage="https://github.com/Dream-High/RMVPE",
    ),
]
