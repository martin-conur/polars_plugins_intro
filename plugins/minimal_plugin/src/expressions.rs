#![allow(clippy::unused_unit)]
use polars::prelude::*;
use polars::prelude::arity::{binary_elementwise, binary_elementwise_values};
use pyo3_polars::derive::polars_expr;
use pyo3_polars::export::polars_core::utils::CustomIterTools;
use pyo3_polars::export::polars_core::export::num::Signed;
// use std::borrow::Cow;
use std::fmt::Write;
use rust_stemmers::{Algorithm, Stemmer};
use serde::Deserialize;

mod utils_fn;
use utils_fn::binary_amortized_elementwise;

fn same_output_type(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    Ok(field.clone())
}

#[polars_expr(output_type_func=same_output_type)]
fn noop(inputs: &[Series]) -> PolarsResult<Series> {
    let s = &inputs[0];
    Ok(s.clone())
} 

#[polars_expr(output_type=Int64)]
fn abs_i64(inputs: &[Series]) -> PolarsResult<Series> {
    let s = &inputs[0];
    let ca = s.i64()?;

    let out = ca.apply_values(|x| x.abs());
    Ok(out.into_series())
}

fn impl_abs_numeric<T>(ca: &ChunkedArray<T>) ->ChunkedArray<T> 
where
    T: PolarsNumericType,
    T::Native: Signed,
{
    ca.apply(|opt_v: Option<T::Native>| opt_v.map(|v: T::Native| v.abs()))
}

#[polars_expr(output_type_func=same_output_type)]
fn abs_numeric(inputs: &[Series]) -> PolarsResult<Series> {
    let s = &inputs[0];
    match s.dtype() {
        DataType::Int32 => Ok(impl_abs_numeric(s.i32().unwrap()).into_series()),
        DataType::Int64 => Ok(impl_abs_numeric(s.i64().unwrap()).into_series()),
        DataType::Float32 => Ok(impl_abs_numeric(s.f32().unwrap()).into_series()),
        DataType::Float64 => Ok(impl_abs_numeric(s.f64().unwrap()).into_series()),
        dtype => {
            polars_bail!(InvalidOperation:format!("dtype {dtype} not \
            supported for abs_numeric, expected Int32, Int64, Float32, Float64."))
        }
    }
}

#[polars_expr(output_type=Int64)]
fn sum_i64(inputs: &[Series]) -> PolarsResult<Series> {
    let left: &Int64Chunked = inputs[0].i64()?;
    let right: &Int64Chunked = inputs[1].i64()?;

    let out: Int64Chunked = binary_elementwise(
        left,
        right,
        |left: Option<i64>, right: Option<i64>| match (left, right) {
            (Some(left), Some(right)) => Some(left + right),
            _ => None
        }
    );
    Ok(out.into_series())
}

#[polars_expr(output_type_func=same_output_type)]
fn cum_sum(inputs: &[Series]) ->PolarsResult<Series> {
    let s = &inputs[0];
    let ca: &Int64Chunked = s.i64()?;
    let out: Int64Chunked = ca
        .into_iter()
        .scan(
            None, |state: &mut Option<i64>, x: Option<i64> | {
                let sum = match (*state, x) {
                    (Some(inner_state), Some(x)) => {
                        *state = Some(inner_state + x);
                        *state
                    },
                    (None, Some(x)) => {
                        *state = Some(x);
                        *state
                    },
                    (_, None) => None
                };
                Some(sum)
            }
        )
        .collect_trusted();
    let out: Int64Chunked = out.with_name(ca.name());
    Ok(out.into_series())
}


// // not optimized way
// #[polars_expr(output_type=String)]
// fn pig_latinnify(inputs: &[Series]) -> PolarsResult<Series> {
//     let s = &inputs[0];
//     let ca: &StringChunked = s.str()?;
//     let out: StringChunked = ca
//         .apply(|v_opt: Option<&str>| {
//             v_opt.map(|value: &str| {
//                 if let Some(first_char) = value.chars().next() {
//                     return Cow::Owned(format!("{}{}ay", &value[1..], first_char))
//                 } else {
//                     Cow::Owned(value.to_string())
//                 }
//             })
//         });
//     Ok(out.into_series())
// }

#[polars_expr(output_type=String)]
fn pig_latinnify(inputs: &[Series]) -> PolarsResult<Series> {
    let ca: &StringChunked = inputs[0].str()?;
    let out: StringChunked = ca
        .apply_to_buffer(|value: &str, output: &mut String| {
            if let Some(first_char) = value.chars().next() {
                write!(output, "{}{}ay", &value[1..], first_char).unwrap()
            }
        });
    Ok(out.into_series())
}

#[polars_expr(output_type=String)]
fn snowball_stem(inputs:&[Series]) -> PolarsResult<Series> {
    let ca: &StringChunked = inputs[0].str()?;
    let es_stemmer = Stemmer::create(Algorithm::Spanish);
    let out: StringChunked = ca.apply_to_buffer(
        |value: &str, output: &mut String| {
            write!(output, "{}", es_stemmer.stem(value)).unwrap()
        }
    );
    Ok(out.into_series())
}

#[derive(Deserialize)]
struct AddSuffixKwargs {
    suffix: String
}

#[polars_expr(output_type=String)]
fn add_suffix(inputs: &[Series], kwargs:AddSuffixKwargs) -> PolarsResult<Series> {
    let ca: &StringChunked = inputs[0].str()?;
    let out: StringChunked = ca.apply_to_buffer(|value, output| {
        write!(output, "{}{}", value, kwargs.suffix).unwrap();
    });
    Ok(out.into_series())
}

#[polars_expr(output_type=Float64)]
fn weighted_mean(inputs: &[Series]) -> PolarsResult<Series> {
    let values = inputs[0].list()?;
    let weights = inputs[1].list()?;

    let out: Float64Chunked = binary_amortized_elementwise(
        values,
        weights,
        |values_inner: &Series, weights_inner: &Series| ->Option<f64> {
            let values_inner = values_inner.i64().unwrap();
            let weights_inner = weights_inner.f64().unwrap();

            let out_inner: Float64Chunked = binary_elementwise_values(
                values_inner,
                weights_inner,
                |value: i64, weight: f64| value as f64 * weight
            );
            match (out_inner.sum(), weights_inner.sum()) {
                (Some(weighted_sum), Some(weights_sum)) => Some(weighted_sum / weights_sum),
                _ => None
            }
        },
    );
    Ok(out.into_series())
}

#[polars_expr(output_type=Float64)]
fn weighted_standard_deviation(inputs: &[Series]) -> PolarsResult<Series> {
    let values = inputs[0].list()?;
    let weights = inputs[1].list()?;

    let out: Float64Chunked = binary_amortized_elementwise(
        values,
        weights,
        |values_inner: &Series, weights_inner: &Series| ->Option<f64> {
            let values_inner = values_inner.i64().expect("Should be Int64"); // how do I print this error in polars?
            let weights_inner = weights_inner.f64().expect("Should be Float64"); // how do I print this error in polars?

            let mult_inner: Float64Chunked = binary_elementwise_values(
                values_inner,
                weights_inner,
                |value: i64, weight: f64| value as f64 * weight
            );
            let weighted_sum = match (mult_inner.sum(), weights_inner.sum()) {
                (Some(weighted_sum), Some(weights_sum)) => Some(weighted_sum / weights_sum),
                _ => None
            };

            let weighted_std_dev_inner: Float64Chunked = binary_elementwise_values(
                values_inner,
                weights_inner,
                |value: i64, weight: f64| {
                    let n = value as f64 -  weighted_sum.unwrap();
                    n * n * weight
                }
            );

            match (weighted_std_dev_inner.sum(), weights_inner.sum()) {
                (Some(weighted_sum_), Some(weights_sum)) => {
                        let m = (weights_inner.len() as f64 - 1.0) / weights_inner.len() as f64;
                        Some((weighted_sum_ / (weights_sum * m)).sqrt())
                    }
                _ => None
            }

        },
    );
    Ok(out.into_series())
}

fn shifted_struct(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    match field.data_type() {
        DataType::Struct(fields) => {
            let mut field_0 = fields[0].clone();
            let name = field_0.name().clone();
            field_0.set_name(fields[fields.len() - 1].name().clone());

            let mut fields = fields[1..]
                .iter()
                .zip(fields[0..fields.len() - 1].iter())
                .map(|(fld, name)| Field::new(name.name(), fld.data_type().clone()))
                .collect::<Vec<_>>();
            fields.push(field_0);
            Ok(Field::new(&name, DataType::Struct(fields)))
        },
        _ => unreachable!()
    }
}

#[polars_expr(output_type_func=shifted_struct)]
fn shift_struct(inputs: &[Series]) -> PolarsResult<Series> {
    let struct_ = inputs[0].struct_()?;
    let fields = struct_.fields();
    if fields.is_empty() {
        return Ok(inputs[0].clone());
    }
    let mut field_0 = fields[0].clone();
    field_0.rename(fields[fields.len() - 1].name());
    let mut fields = fields[1..]
        .iter()
        .zip(fields[..fields.len() - 1].iter())
        .map(|(s, name)| {
            let mut s = s.clone();
            s.rename(name.name());
            s
        })
        .collect::<Vec<_>>();
    fields.push(field_0);
    StructChunked::new(struct_.name(), &fields).map(|ca| ca.into_series())
}