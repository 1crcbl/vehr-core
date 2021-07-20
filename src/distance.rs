//! Supports for calculating distance among nodes.

pub trait DistFn {
    /// Calculates the distance between two points based on their position vectors.
    fn compute_slice(&self, a: &[f64], b: &[f64]) -> f64;

    /// Calculates the distance between two points based on their indices.
    fn compute_index(&self, a: usize, b: usize) -> f64;
}

/// Implementation of Euclidean distance function.
///
/// # Examples
/// ```
/// # use vehr_core::distance::EucDist;
/// # use vehr_core::distance::DistFn;
/// let a = vec![0., 1., 2., 3., 4.];
/// let b = vec![5., 6., 7., 8., 4.];
/// let edf = EucDist::new();
/// assert_eq!(10., edf.compute_slice(&a, &b));
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct EucDist {}

impl EucDist {
    /// Creates a new instance of `Self`.
    pub fn new() -> Self {
        Self {}
    }
}

impl DistFn for EucDist {
    #[inline]
    fn compute_slice(&self, a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .fold(0_f64, |acc, (x1, x2)| acc + (x1 - x2).powi(2))
            .sqrt()
    }

    #[inline]
    fn compute_index(&self, _: usize, _: usize) -> f64 {
        unimplemented!("Distance computation from indices is not implemented for EucDist.")
    }
}

macro_rules! gen_constructor {
    ($struct_name:ident) => {
        impl $struct_name {
            /// Creates a new instance of `Self`.
            pub fn new(dim: usize, data: Vec<f64>) -> Self {
                Self { dim, data }
            }
        }
    };
}

/// Implementation of retrieving distance among nodes from a given full matrix.
///
/// # Examples
/// ```
/// # use vehr_core::distance::FullMtxDist;
/// # use vehr_core::distance::DistFn;
/// let data = vec![
///     [0., 1., 2., 3., 4.],
///     [1., 0., 5., 6., 7.],
///     [2., 5., 0., 8., 9.],
///     [3., 6., 8., 0., 10.],
///     [4., 7., 9., 10., 0.],
/// ]
/// .iter()
/// .cloned()
/// .flatten()
/// .collect();
///
/// let md = FullMtxDist::new(5, data);
/// assert_eq!(3., md.compute_index(3, 0));
/// assert_eq!(8., md.compute_index(2, 3));
/// ```
#[derive(Clone, Debug)]
pub struct FullMtxDist {
    pub(crate) dim: usize,
    pub(crate) data: Vec<f64>,
}

gen_constructor!(FullMtxDist);

impl DistFn for FullMtxDist {
    fn compute_index(&self, index_a: usize, index_b: usize) -> f64 {
        if index_a >= self.dim || index_b >= self.dim {
            0.
        } else {
            self.data[index_a * self.dim + index_b]
        }
    }

    fn compute_slice(&self, _: &[f64], _: &[f64]) -> f64 {
        unimplemented!("Distance computation from slices is not implemented for FullMtxDist.")
    }
}

/// Implementation of retrieving distance among nodes from a matrix given in the lower column format,
/// not including diagonal elements.
///
/// # Examples
/// ```
/// # use vehr_core::distance::LowerColDist;
/// # use vehr_core::distance::DistFn;
/// // Lower column format for the matrix:
/// //  0. | 1. | 2. | 3.  | 4.
/// //  1. | 0. | 5. | 6.  | 7.
/// //  2. | 5. | 0. | 8.  | 9.
/// //  3. | 6. | 8. | 0.  | 10.
/// //  4. | 7. | 9. | 10. | 0.
/// let data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10.];
///
/// let lc = LowerColDist::new(5, data);
/// assert_eq!(3., lc.compute_index(3, 0));
/// assert_eq!(8., lc.compute_index(2, 3));
/// ```
#[derive(Clone, Debug)]
pub struct LowerColDist {
    pub(crate) dim: usize,
    pub(crate) data: Vec<f64>,
}

gen_constructor!(LowerColDist);

impl DistFn for LowerColDist {
    fn compute_index(&self, index_a: usize, index_b: usize) -> f64 {
        if index_a >= self.dim || index_b >= self.dim || index_a == index_b {
            0.
        } else {
            let ii = find_index_upper_row(index_a, index_b, self.dim);
            self.data[ii]
        }
    }

    fn compute_slice(&self, _: &[f64], _: &[f64]) -> f64 {
        unimplemented!("Distance computation from slices is not implemented for LowerColDist.")
    }
}

/// Implementation of retrieving distance among nodes from a matrix given in the upper row format,
/// not including diagonal elements.
///
/// # Examples
/// ```
/// # use vehr_core::distance::UpperRowDist;
/// # use vehr_core::distance::DistFn;
/// // Upper row format for the matrix:
/// //  0. | 1. | 2. | 3.  | 4.
/// //  1. | 0. | 5. | 6.  | 7.
/// //  2. | 5. | 0. | 8.  | 9.
/// //  3. | 6. | 8. | 0.  | 10.
/// //  4. | 7. | 9. | 10. | 0.
/// let data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10.];
///
/// let ur = UpperRowDist::new(5, data);
/// assert_eq!(3., ur.compute_index(3, 0));
/// assert_eq!(8., ur.compute_index(2, 3));
#[derive(Clone, Debug)]
pub struct UpperRowDist {
    pub(crate) dim: usize,
    pub(crate) data: Vec<f64>,
}

gen_constructor!(UpperRowDist);

impl DistFn for UpperRowDist {
    #[inline]
    fn compute_index(&self, index_a: usize, index_b: usize) -> f64 {
        if index_a >= self.dim || index_b >= self.dim || index_a == index_b {
            0.
        } else {
            let ii = find_index_upper_row(index_a, index_b, self.dim);
            self.data[ii]
        }
    }

    fn compute_slice(&self, _: &[f64], _: &[f64]) -> f64 {
        unimplemented!("Distance computation from slices is not implemented for UpperRowDist.")
    }
}

#[inline(always)]
fn find_index_upper_row(index_a: usize, index_b: usize, dim: usize) -> usize {
    let (a, b) = if index_a < index_b {
        (index_a, index_b)
    } else {
        (index_b, index_a)
    };

    a * dim + b - (((a + 2) * (a + 1)) >> 1)
}

/// Implementation of retrieving distance among nodes from a matrix given in the lower column format,
/// including the diagonal.
///
/// # Examples
/// ```
/// # use vehr_core::distance::LowerColDiagDist;
/// # use vehr_core::distance::DistFn;
/// // Lower column format with diagonal elements for the matrix:
/// //  0. | 1. | 2. | 3.  | 4.
/// //  1. | 0. | 5. | 6.  | 7.
/// //  2. | 5. | 0. | 8.  | 9.
/// //  3. | 6. | 8. | 0.  | 10.
/// //  4. | 7. | 9. | 10. | 0.
/// let data = vec![0., 1., 2., 3., 4., 0., 5., 6., 7., 0., 8., 9., 0., 10., 0.];
///
/// let lcd = LowerColDiagDist::new(5, data);
/// assert_eq!(3., lcd.compute_index(3, 0));
/// assert_eq!(8., lcd.compute_index(2, 3));
#[derive(Clone, Debug)]
pub struct LowerColDiagDist {
    pub(crate) dim: usize,
    pub(crate) data: Vec<f64>,
}

gen_constructor!(LowerColDiagDist);

impl DistFn for LowerColDiagDist {
    #[inline]
    fn compute_index(&self, index_a: usize, index_b: usize) -> f64 {
        if index_a >= self.dim || index_b >= self.dim || index_a == index_b {
            0.
        } else {
            let ii = find_index_upper_row_diag(index_a, index_b, self.dim);
            self.data[ii]
        }
    }

    #[inline]
    fn compute_slice(&self, _: &[f64], _: &[f64]) -> f64 {
        unimplemented!("Distance computation from slices is not implemented for LowerColDiagDist.")
    }
}

/// Implementation of retrieving distance among nodes from a matrix given in the upper row format,
/// including the diagonal.
///
/// # Examples
/// ```
/// # use vehr_core::distance::UpperRowDiagDist;
/// # use vehr_core::distance::DistFn;
/// // Upper row format with diagonal elements for the matrix:
/// //  0. | 1. | 2. | 3.  | 4.
/// //  1. | 0. | 5. | 6.  | 7.
/// //  2. | 5. | 0. | 8.  | 9.
/// //  3. | 6. | 8. | 0.  | 10.
/// //  4. | 7. | 9. | 10. | 0.
/// let data = vec![0., 1., 2., 3., 4., 0., 5., 6., 7., 0., 8., 9., 0., 10., 0.];
///
/// let urd = UpperRowDiagDist::new(5, data);
/// assert_eq!(3., urd.compute_index(3, 0));
/// assert_eq!(8., urd.compute_index(2, 3));
#[derive(Clone, Debug)]
pub struct UpperRowDiagDist {
    pub(crate) dim: usize,
    pub(crate) data: Vec<f64>,
}

gen_constructor!(UpperRowDiagDist);

impl DistFn for UpperRowDiagDist {
    #[inline]
    fn compute_index(&self, index_a: usize, index_b: usize) -> f64 {
        if index_a >= self.dim || index_b >= self.dim || index_a == index_b {
            0.
        } else {
            let ii = find_index_upper_row_diag(index_a, index_b, self.dim);
            self.data[ii]
        }
    }

    #[inline]
    fn compute_slice(&self, _: &[f64], _: &[f64]) -> f64 {
        unimplemented!("Distance computation from slices is not implemented for UpperRowDiagDist.")
    }
}

#[inline(always)]
fn find_index_upper_row_diag(index_a: usize, index_b: usize, dim: usize) -> usize {
    let (a, b) = if index_a < index_b {
        (index_a, index_b)
    } else {
        (index_b, index_a)
    };
    a * dim + b - ((a * (a + 1)) >> 1)
}

/// Implementation of retrieving distance among nodes from a matrix given in the upper column format,
/// not including diagonal elements.
///
/// # Examples
/// ```
/// # use vehr_core::distance::UpperColDist;
/// # use vehr_core::distance::DistFn;
/// // Upper column format with diagonals for the matrix:
/// //  0. | 1. | 2. | 3.  | 4.
/// //  1. | 0. | 5. | 6.  | 7.
/// //  2. | 5. | 0. | 8.  | 9.
/// //  3. | 6. | 8. | 0.  | 10.
/// //  4. | 7. | 9. | 10. | 0.
/// let data = vec![1., 2., 5., 3., 6., 8., 4., 7., 9., 10.];
///
/// let uc = UpperColDist::new(5, data);
/// assert_eq!(3., uc.compute_index(3, 0));
/// assert_eq!(8., uc.compute_index(2, 3));
#[derive(Clone, Debug)]
pub struct UpperColDist {
    pub(crate) dim: usize,
    pub(crate) data: Vec<f64>,
}

gen_constructor!(UpperColDist);

impl DistFn for UpperColDist {
    fn compute_index(&self, index_a: usize, index_b: usize) -> f64 {
        if index_a >= self.dim || index_b >= self.dim || index_a == index_b {
            0.
        } else {
            let ii = find_index_lower_row(index_a, index_b);
            self.data[ii]
        }
    }

    fn compute_slice(&self, _: &[f64], _: &[f64]) -> f64 {
        unimplemented!("Distance computation from slices is not implemented for UpperColDist.")
    }
}

/// Implementation of retrieving distance among nodes from a matrix given in the lower row format,
/// not including diagonal elements.
///
/// # Examples
/// ```
/// # use vehr_core::distance::LowerRowDist;
/// # use vehr_core::distance::DistFn;
/// // Lower row format for the matrix:
/// //  0. | 1. | 2. | 3.  | 4.
/// //  1. | 0. | 5. | 6.  | 7.
/// //  2. | 5. | 0. | 8.  | 9.
/// //  3. | 6. | 8. | 0.  | 10.
/// //  4. | 7. | 9. | 10. | 0.
/// let data = vec![1., 2., 5., 3., 6., 8., 4., 7., 9., 10.];
///
/// let lr = LowerRowDist::new(5, data);
/// assert_eq!(3., lr.compute_index(3, 0));
/// assert_eq!(8., lr.compute_index(2, 3));
#[derive(Clone, Debug)]
pub struct LowerRowDist {
    pub(crate) dim: usize,
    pub(crate) data: Vec<f64>,
}

gen_constructor!(LowerRowDist);

impl DistFn for LowerRowDist {
    #[inline]
    fn compute_index(&self, index_a: usize, index_b: usize) -> f64 {
        if index_a >= self.dim || index_b >= self.dim || index_a == index_b {
            0.
        } else {
            let ii = find_index_lower_row(index_a, index_b);
            self.data[ii]
        }
    }

    fn compute_slice(&self, _: &[f64], _: &[f64]) -> f64 {
        unimplemented!("Distance computation from slices is not implemented for LowerRowDist.")
    }
}

#[inline(always)]
fn find_index_lower_row(index_a: usize, index_b: usize) -> usize {
    let (a, b) = if index_a > index_b {
        (index_b, index_a)
    } else {
        (index_a, index_b)
    };
    a + (((b - 1) * b) >> 1)
}

/// Implementation of retrieving distance among nodes from a matrix given in the upper column format,
/// including diagonal elements.
///
/// # Examples
/// ```
/// # use vehr_core::distance::UpperColDiagDist;
/// # use vehr_core::distance::DistFn;
/// // Upper column format with diagonals for the matrix:
/// //  0. | 1. | 2. | 3.  | 4.
/// //  1. | 0. | 5. | 6.  | 7.
/// //  2. | 5. | 0. | 8.  | 9.
/// //  3. | 6. | 8. | 0.  | 10.
/// //  4. | 7. | 9. | 10. | 0.
/// let data = vec![0., 1., 0., 2., 5., 0., 3., 6., 8., 0., 4., 7., 9., 10., 0.];
///
/// let ucd = UpperColDiagDist::new(5, data);
/// assert_eq!(3., ucd.compute_index(3, 0));
/// assert_eq!(8., ucd.compute_index(2, 3));
#[derive(Clone, Debug)]
pub struct UpperColDiagDist {
    pub(crate) dim: usize,
    pub(crate) data: Vec<f64>,
}

gen_constructor!(UpperColDiagDist);

impl DistFn for UpperColDiagDist {
    fn compute_index(&self, index_a: usize, index_b: usize) -> f64 {
        if index_a >= self.dim || index_b >= self.dim || index_a == index_b {
            0.
        } else {
            let ii = find_index_lower_row_diag(index_a, index_b);
            self.data[ii]
        }
    }

    fn compute_slice(&self, _: &[f64], _: &[f64]) -> f64 {
        unimplemented!("Distance computation from slices is not implemented for UpperColDiagDist.")
    }
}

/// Implementation of retrieving distance among nodes from a matrix given in the lower row format,
/// including diagonal elements.
///
/// # Examples
/// ```
/// # use vehr_core::distance::LowerRowDiagDist;
/// # use vehr_core::distance::DistFn;
/// // Lower row format with diagonals for the matrix:
/// //  0. | 1. | 2. | 3.  | 4.
/// //  1. | 0. | 5. | 6.  | 7.
/// //  2. | 5. | 0. | 8.  | 9.
/// //  3. | 6. | 8. | 0.  | 10.
/// //  4. | 7. | 9. | 10. | 0.
/// let data = vec![0., 1., 0., 2., 5., 0., 3., 6., 8., 0., 4., 7., 9., 10., 0.];
///
/// let lrd = LowerRowDiagDist::new(5, data);
/// assert_eq!(3., lrd.compute_index(3, 0));
/// assert_eq!(8., lrd.compute_index(2, 3));
#[derive(Clone, Debug)]
pub struct LowerRowDiagDist {
    pub(crate) dim: usize,
    pub(crate) data: Vec<f64>,
}

gen_constructor!(LowerRowDiagDist);

impl DistFn for LowerRowDiagDist {
    #[inline]
    fn compute_index(&self, index_a: usize, index_b: usize) -> f64 {
        if index_a >= self.dim || index_b >= self.dim || index_a == index_b {
            0.
        } else {
            let ii = find_index_lower_row_diag(index_a, index_b);
            self.data[ii]
        }
    }

    fn compute_slice(&self, _: &[f64], _: &[f64]) -> f64 {
        unimplemented!("Distance computation from slices is not implemented for LowerRowDiagDist.")
    }
}

#[inline(always)]
fn find_index_lower_row_diag(index_a: usize, index_b: usize) -> usize {
    let (a, b) = if index_a > index_b {
        (index_a, index_b)
    } else {
        (index_b, index_a)
    };
    b + (((a + 1) * a) >> 1)
}

#[cfg(test)]
mod tests {
    use crate::distance::{
        DistFn, EucDist, FullMtxDist, LowerColDiagDist, LowerColDist, LowerRowDiagDist,
        LowerRowDist, UpperColDiagDist, UpperColDist, UpperRowDiagDist, UpperRowDist,
    };

    #[test]
    fn test_euclidean() {
        let a = vec![0., 1., 2., 3., 4.];
        let b = vec![5., 6., 7., 8., 4.];
        let edf = EucDist::new();
        assert_eq!(10., edf.compute_slice(&a, &b));
    }

    fn validate_mtx<D: DistFn>(mtx: &D) {
        let data: Vec<_> = vec![
            [0., 1., 2., 3., 4.],
            [1., 0., 5., 6., 7.],
            [2., 5., 0., 8., 9.],
            [3., 6., 8., 0., 10.],
            [4., 7., 9., 10., 0.],
        ];

        for ii in 0..5 {
            for jj in 0..5 {
                let exp = data[ii][jj];
                let res = mtx.compute_index(ii, jj);
                assert_eq!(exp, res);
            }
        }
    }

    #[test]
    fn test_full_matrix() {
        let data = vec![
            [0., 1., 2., 3., 4.],
            [1., 0., 5., 6., 7.],
            [2., 5., 0., 8., 9.],
            [3., 6., 8., 0., 10.],
            [4., 7., 9., 10., 0.],
        ]
        .iter()
        .cloned()
        .flatten()
        .collect();

        validate_mtx(&FullMtxDist::new(5, data));
    }

    #[test]
    fn test_lower_col() {
        let data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10.];
        validate_mtx(&LowerColDist::new(5, data));
    }

    #[test]
    fn test_upper_row() {
        let data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10.];
        validate_mtx(&UpperRowDist::new(5, data));
    }

    #[test]
    fn test_lower_col_diag() {
        let data = vec![0., 1., 2., 3., 4., 0., 5., 6., 7., 0., 8., 9., 0., 10., 0.];
        validate_mtx(&LowerColDiagDist::new(5, data));
    }

    #[test]
    fn test_upper_row_diag() {
        let data = vec![0., 1., 2., 3., 4., 0., 5., 6., 7., 0., 8., 9., 0., 10., 0.];
        validate_mtx(&UpperRowDiagDist::new(5, data));
    }

    #[test]
    fn test_upper_col() {
        let data = vec![1., 2., 5., 3., 6., 8., 4., 7., 9., 10.];
        validate_mtx(&UpperColDist::new(5, data));
    }

    #[test]
    fn test_lower_row() {
        let data = vec![1., 2., 5., 3., 6., 8., 4., 7., 9., 10.];
        validate_mtx(&LowerRowDist::new(5, data));
    }

    #[test]
    fn test_upper_col_diag() {
        let data = vec![0., 1., 0., 2., 5., 0., 3., 6., 8., 0., 4., 7., 9., 10., 0.];
        validate_mtx(&UpperColDiagDist::new(5, data));
    }

    #[test]
    fn test_lower_row_diag() {
        let data = vec![0., 1., 0., 2., 5., 0., 3., 6., 8., 0., 4., 7., 9., 10., 0.];
        validate_mtx(&LowerRowDiagDist::new(5, data));
    }
}
