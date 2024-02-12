import polars as pl
from polars.utils.udfs import _get_shared_lib_location
from polars.type_aliases import IntoExpr
from minimal_plugin.utils import parse_into_expr

lib = _get_shared_lib_location(__file__)


def noop(expr: IntoExpr) -> pl.Expr:
    expr = parse_into_expr(expr)
    return expr.register_plugin(
        lib=lib,
        symbol="noop",
        is_elementwise=True,
    )

def abs_i64(expr) -> pl.Expr:
    expr = parse_into_expr(expr)
    return expr.register_plugin(
        lib=lib,
        symbol="abs_i64",
        is_elementwise=True
    )

def abs_numeric(expr) -> pl.Expr:
    expr = parse_into_expr(expr)
    return expr.register_plugin(
        lib=lib,
        symbol="abs_numeric",
        is_elementwise=True
    )

def sum_i64(expr, other:IntoExpr) -> pl.Expr:
    expr = parse_into_expr(expr)
    return expr.register_plugin(
        lib=lib,
        symbol="sum_i64",
        is_elementwise=True,
        args=[other]
    )

def cum_sum(expr) -> pl.Expr:
    expr = parse_into_expr(expr)
    return expr.register_plugin(
        lib=lib,
        symbol="cum_sum",
        is_elementwise=False
    )

def pig_latinnify(expr: IntoExpr) -> pl.Expr:
    expr = parse_into_expr(expr)
    return expr.register_plugin(
        lib=lib,
        symbol="pig_latinnify",
        is_elementwise=True
    )

def snowball_stem(expr: IntoExpr) -> pl.Expr:
    expr = parse_into_expr(expr)
    return expr.register_plugin(
        lib=lib,
        symbol="snowball_stem",
        is_elementwise=True
    )

def add_suffix(expr: IntoExpr, *, suffix: str) -> pl.Expr:
    expr = parse_into_expr(expr)
    return expr.register_plugin(
        lib=lib,
        symbol="add_suffix",
        is_elementwise=True,
        kwargs={
            "suffix": suffix
        }
    )

def weighted_mean(expr: IntoExpr, weights: IntoExpr) -> pl.Expr:
    expr = parse_into_expr(expr)
    return expr.register_plugin(
        lib=lib,
        symbol="weighted_mean",
        is_elementwise=True,
        args=[weights]
    ) 

def weighted_standard_deviation(expr: IntoExpr, weights: IntoExpr) -> pl.Expr:
    expr = parse_into_expr(expr)
    return expr.register_plugin(
        lib=lib,
        symbol="weighted_standard_deviation",
        is_elementwise=True,
        args=[weights]
    )

def shift_struct(expr: IntoExpr) -> pl.Expr:
    expr = parse_into_expr(expr)
    return expr.register_plugin(
        lib=lib,
        symbol="shift_struct",
        is_elementwise=True
    )

