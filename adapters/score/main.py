"""``python -m adapters.score.main`` -- JSON-RPC server (or ``--selftest``) for the score env."""

from adapters.common.rpc import lazy_table, main
from adapters.score.table import TABLE

if __name__ == "__main__":
    main(lazy_table(TABLE), "score")
