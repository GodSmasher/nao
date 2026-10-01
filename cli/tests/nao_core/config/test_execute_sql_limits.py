"""Tests for the SQL result caps enforced by DatabaseConfig.execute_sql."""

from __future__ import annotations

import os
from unittest.mock import patch

import pandas as pd
import pytest

from nao_core.config.databases.base import DatabaseConfig
from nao_core.config.exceptions import ResultTooLargeError


class _FetchallCursor:
    """Cursor shape nao's execute_sql takes the fetchall branch for."""

    def __init__(self, rows: list[tuple], columns: list[str], *, support_fetchmany: bool = True) -> None:
        self.description = [(name,) for name in columns]
        self._rows = rows
        self._offset = 0
        if not support_fetchmany:
            # Simulate a driver that only exposes fetchall (no streaming).
            self.fetchmany = None  # type: ignore[assignment]

    def fetchmany(self, size: int) -> list[tuple]:  # noqa: D401
        chunk = self._rows[self._offset : self._offset + size]
        self._offset += size
        return chunk

    def fetchall(self) -> list[tuple]:  # noqa: D401
        chunk = self._rows[self._offset :]
        self._offset = len(self._rows)
        return chunk


class _DataFrameCursor:
    """Cursor shape nao's execute_sql takes the fetchdf branch for."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def fetchdf(self) -> pd.DataFrame:  # noqa: D401
        return self._df


class _StubBackend:
    def __init__(self, cursor: object) -> None:
        self._cursor = cursor
        self.disconnected = False

    def raw_sql(self, _sql: str) -> object:
        return self._cursor

    def disconnect(self) -> None:
        self.disconnected = True


def _run(cursor: object) -> pd.DataFrame:
    """Call execute_sql with a stub backend, bypassing config construction."""
    backend = _StubBackend(cursor)
    return DatabaseConfig.execute_sql.__func__(DatabaseConfig, "SELECT 1", backend)  # type: ignore[arg-type]


def test_fetchall_path_returns_rows_under_the_cap():
    cursor = _FetchallCursor(rows=[(i, f"v{i}") for i in range(5)], columns=["id", "name"])
    df = _run(cursor)
    assert list(df.columns) == ["id", "name"]
    assert len(df) == 5


def test_fetchall_path_aborts_before_materializing_everything_when_row_cap_is_hit():
    rows = [(i,) for i in range(50)]
    cursor = _FetchallCursor(rows=rows, columns=["id"])
    with patch.dict(os.environ, {"NAO_SQL_MAX_RESULT_ROWS": "10"}):
        with pytest.raises(ResultTooLargeError) as excinfo:
            _run(cursor)
    assert "more than 10 rows" in str(excinfo.value)
    assert "NAO_SQL_MAX_RESULT_ROWS" in str(excinfo.value)


def test_fetchall_path_aborts_when_byte_cap_is_hit():
    rows = [("x" * 1024,) for _ in range(100)]
    cursor = _FetchallCursor(rows=rows, columns=["payload"])
    with patch.dict(os.environ, {"NAO_SQL_MAX_RESULT_BYTES": "2048"}):
        with pytest.raises(ResultTooLargeError) as excinfo:
            _run(cursor)
    assert "bytes" in str(excinfo.value)


def test_dataframe_path_rejects_oversized_result_after_the_driver_materialized_it():
    df = pd.DataFrame({"id": range(100)})
    cursor = _DataFrameCursor(df)
    with patch.dict(os.environ, {"NAO_SQL_MAX_RESULT_ROWS": "10"}):
        with pytest.raises(ResultTooLargeError) as excinfo:
            _run(cursor)
    assert "100 rows exceed the cap of 10" in str(excinfo.value)


def test_dataframe_path_passes_results_within_the_cap_through_unchanged():
    df = pd.DataFrame({"id": [1, 2, 3]})
    cursor = _DataFrameCursor(df)
    result = _run(cursor)
    pd.testing.assert_frame_equal(result, df)


def test_invalid_env_values_fall_back_to_defaults_rather_than_disabling_the_cap():
    cursor = _FetchallCursor(rows=[(i,) for i in range(3)], columns=["id"])
    with patch.dict(os.environ, {"NAO_SQL_MAX_RESULT_ROWS": "not-a-number"}):
        df = _run(cursor)
    assert len(df) == 3
