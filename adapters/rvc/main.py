"""``python -m adapters.rvc.main`` -- JSON-RPC server (or ``--selftest``) for the rvc env."""

from adapters.common.rpc import lazy_table, main
from adapters.rvc.table import TABLE

if __name__ == "__main__":
    main(lazy_table(TABLE), "rvc")
