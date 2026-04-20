from __future__ import annotations

from collections.abc import MutableMapping, Sequence
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from pioneerml.data_loader.loaders.stage.stages.base_stage import BaseStage


class PurityRowGuardStage(BaseStage):
    """
    Row-level guard/filter stage for PURITY parquet inputs.

    Training can drop invalid rows aggressively (e.g. no ATAR hits), while
    inference can keep all rows for strict output alignment.
    """

    name = "guard_rows"
    requires = ("table",)
    provides = ("table",)

    def __init__(
        self,
        *,
        enabled: bool = True,
        training_only: bool = True,
        require_nonempty_atar: bool = True,
        require_nonempty_total_hits: bool = True,
        max_total_hits: int | None = None,
        require_finite_scalar_columns: Sequence[str] | None = None,
        atar_column: str = "atar_x",
        lyso_column: str = "lyso_x",
    ) -> None:
        self.enabled = bool(enabled)
        self.training_only = bool(training_only)
        self.require_nonempty_atar = bool(require_nonempty_atar)
        self.require_nonempty_total_hits = bool(require_nonempty_total_hits)
        self.max_total_hits = None if max_total_hits is None else int(max_total_hits)
        self.require_finite_scalar_columns = tuple(
            str(col).strip()
            for col in (require_finite_scalar_columns or [])
            if str(col).strip() != ""
        )
        self.atar_column = str(atar_column)
        self.lyso_column = str(lyso_column)

    @staticmethod
    def _list_lengths_or_zeros(*, table: pa.Table, column: str) -> np.ndarray:
        if column not in table.column_names:
            return np.zeros((int(table.num_rows),), dtype=np.int64)
        arr = table.column(column).combine_chunks()
        lens = pc.list_value_length(arr)
        lens = pc.fill_null(lens, 0)
        return lens.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)

    @staticmethod
    def _finite_scalar_mask(*, table: pa.Table, columns: Sequence[str]) -> tuple[np.ndarray, dict[str, int]]:
        keep = np.ones((int(table.num_rows),), dtype=bool)
        dropped_by_col: dict[str, int] = {}
        for col in columns:
            if col not in table.column_names:
                continue
            arr = table.column(col).combine_chunks()
            try:
                finite = pc.is_finite(arr)
            except Exception:
                finite = pc.is_valid(arr)
            finite = pc.fill_null(finite, False)
            col_keep = finite.to_numpy(zero_copy_only=False).astype(bool, copy=False)
            dropped_by_col[str(col)] = int((~col_keep).sum())
            keep &= col_keep
        return keep, dropped_by_col

    def run_loader(self, *, state: MutableMapping[str, Any], owner) -> None:
        if not self.enabled:
            return
        if self.training_only and not bool(getattr(owner, "include_targets", False)):
            return

        table = state.get("table")
        if table is None or int(table.num_rows) == 0:
            state["table"] = None
            state["chunk_out"] = None
            state["stop_pipeline"] = True
            return

        atar_len = self._list_lengths_or_zeros(table=table, column=self.atar_column)
        lyso_len = self._list_lengths_or_zeros(table=table, column=self.lyso_column)
        total_len = atar_len + lyso_len

        keep = np.ones((int(table.num_rows),), dtype=bool)
        if self.require_nonempty_atar:
            keep &= atar_len > 0
        if self.require_nonempty_total_hits:
            keep &= total_len > 0
        if self.max_total_hits is not None:
            keep &= total_len <= int(self.max_total_hits)
        nonfinite_by_col: dict[str, int] = {}
        if len(self.require_finite_scalar_columns) > 0:
            finite_keep, nonfinite_by_col = self._finite_scalar_mask(
                table=table,
                columns=self.require_finite_scalar_columns,
            )
            keep &= finite_keep

        if keep.size == 0 or not bool(np.any(keep)):
            state["table"] = None
            state["chunk_out"] = None
            state["stop_pipeline"] = True
            state["purity_row_guard_stats"] = {
                "kept_rows": 0,
                "dropped_rows": int(table.num_rows),
                "drop_no_atar": int(np.sum(atar_len <= 0)),
                "drop_no_hits": int(np.sum(total_len <= 0)),
                "drop_nonfinite_scalar": nonfinite_by_col,
            }
            return

        if not bool(np.all(keep)):
            table = table.filter(pa.array(keep))

        state["table"] = table
        state["purity_row_guard_stats"] = {
            "kept_rows": int(np.sum(keep)),
            "dropped_rows": int(keep.size - np.sum(keep)),
            "drop_no_atar": int(np.sum(atar_len <= 0)),
            "drop_no_hits": int(np.sum(total_len <= 0)),
            "drop_nonfinite_scalar": nonfinite_by_col,
        }
