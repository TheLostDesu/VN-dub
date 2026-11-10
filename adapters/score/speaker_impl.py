"""Speaker embeddings written as ``.npy`` (ECAPA 192-d, WavLM-SV 512-d), L2-normalized."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from adapters.common.rpc import ModelImpl, audio_seconds, load_audio, load_audios, run_in_batches, weight_dir
from adapters.common.yamlcompat import ruamel_old_api_compat


def _save(out: str, vec) -> dict[str, Any]:
    import numpy as np

    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    if n > 0:
        v = v / n
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.save(out, v)
    return {"path": out, "dim": int(v.size)}


class EcapaEmbedder(ModelImpl):
    selftest_op = "embed"

    probe_imports = ("torch", "speechbrain.inference.speaker")  # what load() imports (ModelImpl.probe)

    @classmethod
    def probe_setup(cls, params: dict[str, Any]) -> None:
        ruamel_old_api_compat()

    @classmethod
    def probe_checks(cls, params: dict[str, Any]) -> dict[str, Any] | None:
        from adapters.common.yamlcompat import hyperpyyaml_smoke

        return hyperpyyaml_smoke()  # hyperparams.yaml goes through HyperPyYAML (round 11: ruamel 0.19)

    def load(self) -> dict[str, Any]:
        import torch

        ruamel_old_api_compat()  # hyperparams.yaml goes through HyperPyYAML's old ruamel loader
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:  # speechbrain < 1.0
            from speechbrain.pretrained import EncoderClassifier  # type: ignore
        d = weight_dir(self.params, 0)
        # savedir == source: speechbrain uses the pulled files in place (no copies, no symlinks,
        # and no failure on the optional custom.py that this repo does not ship)
        # hyperparams.yaml points pretrained_path at the hub repo: redirect it to the local snapshot
        kwargs: dict[str, Any] = {"source": str(d), "savedir": str(d), "run_opts": {"device": self.device}, "overrides": {"pretrained_path": str(d)}}
        try:
            from speechbrain.utils.fetching import LocalStrategy

            kwargs["local_strategy"] = LocalStrategy.COPY  # no symlinks on Windows
        except ImportError:
            pass
        self.model = EncoderClassifier.from_hparams(**kwargs)
        self.model.eval()
        self._torch = torch
        return {"source": str(d)}

    def op_embed(self, item: dict[str, Any], **_: Any) -> dict[str, Any]:
        torch = self._torch
        y, _ = load_audio(item["wav"], 16000)
        with torch.inference_mode():
            emb = self.model.encode_batch(torch.from_numpy(y).unsqueeze(0).to(self.device))
        return _save(item["out"], emb.squeeze().cpu().numpy())

    def batch_embed(self, items: list[dict[str, Any]], **kw: Any) -> list[dict[str, Any]]:
        """Several clips per ``EncoderClassifier.encode_batch(wavs, wav_lens)`` call, OPT-IN (``options['pad_batch']``).

        With ``pad_batch`` the clips are sorted by duration, cut into sub-batches of ``options['batch_size']``
        (default 8) and zero-padded to the longest clip of the sub-batch; ``wav_lens`` carries each clip's
        length relative to the padded one (``len / max_len``), which speechbrain uses to mask the padding in
        the sentence mean normalisation, the attentive statistics pooling and the squeeze-excitation blocks.
        The TDNN/Res2Net convolutions still read the padded frames at the clip end (reflect padding is only
        applied at the end of the *batch*), so the embedding is NOT identical to the single-clip call:
        measured on CPU, cosine vs ``op_embed`` is 0.984-0.999 for clips of 0.5-2 s, ~0.998 for 3-4 s and
        >0.9997 from ~6 s, as soon as a clip gets one frame (10 ms) of padding. Because the requirement is
        results equal to the per-item path, the default is to run clip by clip (same numbers as ``op_embed``,
        per-item errors); enable ``pad_batch`` only when that drift is acceptable. A failing padded sub-batch
        (incl. CUDA OOM) is redone clip by clip with :meth:`op_embed`.
        """
        size = self.sub_batch_size() if self.options.get("pad_batch") else 1
        return run_in_batches(items, size, self._embed_chunk, lambda it: self.op_embed(it, **kw), label=self.name, sort_key=audio_seconds)

    def _embed_chunk(self, items: list[dict[str, Any]]) -> list[Any]:
        import numpy as np

        torch = self._torch
        ys, out = load_audios(items)
        ok = [i for i, e in enumerate(out) if e is None]
        if ok:
            lens = [len(ys[i]) for i in ok]
            top = max(lens)
            x = np.zeros((len(ok), top), dtype=np.float32)
            for r, i in enumerate(ok):
                x[r, : lens[r]] = ys[i]
            rel = torch.tensor([n / top for n in lens], dtype=torch.float32)
            with torch.inference_mode():
                emb = self.model.encode_batch(torch.from_numpy(x).to(self.device), rel.to(self.device))
            emb = emb.reshape(len(ok), -1).cpu().numpy()
            for r, i in enumerate(ok):
                try:
                    out[i] = _save(items[i]["out"], emb[r])
                except Exception as e:  # per-item failure (unwritable path ...)
                    out[i] = e
        return out


class WavLMSV(ModelImpl):
    """No ``batch_embed`` on purpose: wavlm-base-plus-sv has ``feat_extract_norm="group"``, whose first conv layer
    normalises over the whole (padded) time axis, so padding changes the embedding even with an attention mask
    (random-weight check with the default WavLM-base config: cosine 0.94 for a 0.5 s clip next to a 1.5 s one)."""

    selftest_op = "embed"

    probe_imports = ("torch", "transformers")  # what load() imports (ModelImpl.probe)

    def load(self) -> dict[str, Any]:
        import torch
        from transformers import AutoFeatureExtractor, WavLMForXVector

        d = weight_dir(self.params, 0)
        self.fe = AutoFeatureExtractor.from_pretrained(str(d))
        self.model = WavLMForXVector.from_pretrained(str(d)).to(self.device).eval()
        self._torch = torch
        return {"source": str(d)}

    def op_embed(self, item: dict[str, Any], **_: Any) -> dict[str, Any]:
        torch = self._torch
        y, _ = load_audio(item["wav"], 16000)
        inputs = self.fe(y, sampling_rate=16000, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            emb = self.model(**inputs).embeddings
        return _save(item["out"], emb.squeeze().cpu().numpy())
