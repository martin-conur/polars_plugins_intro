import polars as pl
from polars.utils.udfs import _get_shared_lib_location

lib = _get_shared_lib_location(__file__)


@pl.api.register_expr_namespace("mp")
class MinimalExamples:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def noop(self) -> pl.Expr:
        return self._expr.register_plugin(
            lib=lib,
            symbol="noop",
            is_elementwise=True,
        )

    def abs_i64(self) -> pl.Expr:
        return self._expr.register_plugin(
            lib=lib,
            symbol="abs_i64",
            is_elementwise=True
        )