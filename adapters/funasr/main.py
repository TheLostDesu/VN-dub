"""``python -m adapters.funasr.main`` -- JSON-RPC server (or ``--selftest``) for the funasr env."""

from adapters.common.rpc import lazy_table, main
from adapters.funasr.table import TABLE

if __name__ == "__main__":
    main(lazy_table(TABLE), "funasr")
