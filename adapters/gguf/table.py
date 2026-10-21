"""Dispatch table of the ``gguf`` env (subprocess wrappers around native GGUF runners)."""

TABLE = {
    "fish_s2": "adapters.gguf.cli_impl:FishS2CLI",
    "zonos2": "adapters.gguf.cli_impl:Zonos2CLI",
    "moss_tts": "adapters.gguf.cli_impl:MossTTSCLI",
}
