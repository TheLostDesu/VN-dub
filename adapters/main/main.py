"""``python -m adapters.main.main`` -- JSON-RPC server (or ``--selftest``) for the main env."""

from adapters.common.rpc import lazy_table, main
from adapters.main.table import TABLE

if __name__ == "__main__":
    main(lazy_table(TABLE), "main")
