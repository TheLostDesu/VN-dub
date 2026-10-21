"""``python -m adapters.f5.main`` -- JSON-RPC server (or ``--selftest``)."""

from adapters.common.rpc import lazy_table, main
from adapters.f5.table import TABLE

if __name__ == "__main__":
    main(lazy_table(TABLE), "f5")
