"""``python -m adapters.higgs.main`` -- JSON-RPC server (or ``--selftest``)."""

from adapters.common.rpc import lazy_table, main
from adapters.higgs.table import TABLE

if __name__ == "__main__":
    main(lazy_table(TABLE), "higgs")
