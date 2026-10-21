"""``python -m adapters.asr.main`` -- JSON-RPC server (or ``--selftest``) for the asr env."""

from adapters.asr.table import TABLE
from adapters.common.rpc import lazy_table, main

if __name__ == "__main__":
    main(lazy_table(TABLE), "asr")
