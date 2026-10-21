"""``python -m adapters.chatterbox.main`` -- JSON-RPC server (or ``--selftest``)."""

from adapters.chatterbox.table import TABLE
from adapters.common.rpc import lazy_table, main

if __name__ == "__main__":
    main(lazy_table(TABLE), "chatterbox")
