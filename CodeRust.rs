// rustimport:pyo3

use pyo3::prelude::*;

#[pyfunction]
fn square(x: i32) -> i32 {
    x * x
}
