"""``python -m adapters.hf.main`` -- JSON-RPC server (or ``--selftest``) for the hf env."""

from adapters.common.rpc import lazy_table, main
from adapters.hf.table import TABLE

if __name__ == "__main__":
    main(lazy_table(TABLE), "hf")
