"""Phase 16.8 — DuckDB-backed columnar input-table reader (Tulipa-style).

Tulipa keeps each case's input tables in a per-case DuckDB database and does
columnar joins on them; nexus has historically read inputs with pandas, which
is plenty fast at test sizes but slower than DuckDB on EU-scale tables with
wide joins. This module adds an optional DuckDB fast path with a transparent
pandas fallback, so callers get DuckDB's columnar speed when the package is
installed and identical results from pandas when it is not.

    from nexus_energy.io_tables import read_table, join_tables, read_csv_dir
    df = read_table("assets.csv")                      # auto: duckdb if present
    df = read_table("assets.csv", engine="pandas")     # force pandas
    merged = join_tables(left, right, on="asset", engine="auto")

All readers return a ``pandas.DataFrame`` regardless of engine, so downstream
code is engine-agnostic.
"""

from __future__ import annotations

import glob
import importlib.util
import os


def duckdb_available() -> bool:
    """True iff the optional ``duckdb`` package is importable."""
    return importlib.util.find_spec("duckdb") is not None


def _resolve_engine(engine: str) -> str:
    e = engine.lower()
    if e == "auto":
        return "duckdb" if duckdb_available() else "pandas"
    if e == "duckdb" and not duckdb_available():
        raise ImportError(
            "engine='duckdb' requested but the 'duckdb' package is not "
            "installed. `pip install duckdb` or use engine='pandas'/'auto'.")
    if e not in ("duckdb", "pandas"):
        raise ValueError(f"Unknown engine {engine!r}; use 'auto'|'duckdb'|'pandas'.")
    return e


def read_table(path: str, *, engine: str = "auto", **read_kwargs):
    """Read a single CSV/Parquet table into a DataFrame.

    DuckDB path uses ``read_csv_auto`` / ``read_parquet`` (columnar, typed);
    the pandas path uses ``read_csv`` / ``read_parquet``. Both return a
    ``pandas.DataFrame``.
    """
    eng = _resolve_engine(engine)
    is_parquet = path.lower().endswith((".parquet", ".pq"))
    if eng == "duckdb":
        import duckdb
        con = duckdb.connect()
        try:
            fn = "read_parquet" if is_parquet else "read_csv_auto"
            return con.execute(f"SELECT * FROM {fn}('{path}')").df()
        finally:
            con.close()
    import pandas as pd
    if is_parquet:
        return pd.read_parquet(path, **read_kwargs)
    return pd.read_csv(path, **read_kwargs)


def join_tables(left, right, *, on, how: str = "inner", engine: str = "auto"):
    """Join two DataFrames on ``on`` (a column or list of columns).

    DuckDB path runs the join in its columnar engine (fast on wide EU-scale
    inputs); pandas path uses ``DataFrame.merge``. Returns a ``DataFrame``.
    """
    eng = _resolve_engine(engine)
    keys = [on] if isinstance(on, str) else list(on)
    if eng == "duckdb":
        import duckdb
        con = duckdb.connect()
        try:
            con.register("left_tbl", left)
            con.register("right_tbl", right)
            cond = " AND ".join(f"left_tbl.{k} = right_tbl.{k}" for k in keys)
            jt = {"inner": "INNER", "left": "LEFT", "right": "RIGHT",
                  "outer": "FULL OUTER"}.get(how, "INNER")
            return con.execute(
                f"SELECT * FROM left_tbl {jt} JOIN right_tbl ON {cond}").df()
        finally:
            con.close()
    return left.merge(right, on=keys, how=how)


def read_csv_dir(directory: str, *, engine: str = "auto", pattern: str = "*.csv",
                 **read_kwargs) -> dict:
    """Read every CSV in ``directory`` into ``{table_name: DataFrame}``.

    ``table_name`` is the filename without extension — the per-case table set a
    Tulipa-style workflow expects. Uses :func:`read_table` per file, so the
    engine choice (DuckDB / pandas) applies uniformly.
    """
    out = {}
    for fp in sorted(glob.glob(os.path.join(directory, pattern))):
        name = os.path.splitext(os.path.basename(fp))[0]
        out[name] = read_table(fp, engine=engine, **read_kwargs)
    return out
