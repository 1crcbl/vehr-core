#[derive(Clone, Copy, Debug, Default, PartialEq, Hash)]
pub struct EucDistance {}

impl EucDistance {
    pub fn new() -> Self {
        Self {}
    }
}

macro_rules! gen_new_mtx {
    ($struct_name:ident) => {
        impl $struct_name {
            pub fn new(dim: usize, data: Vec<f64>) -> Self {
                Self { dim, data }
            }
        }
    };
}
pub struct FullMtxDist {
    pub(crate) dim: usize,
    pub(crate) data: Vec<f64>,
}

gen_new_mtx!(FullMtxDist);

pub struct LowerColDist {
    pub(crate) dim: usize,
    pub(crate) data: Vec<f64>,
}

gen_new_mtx!(LowerColDist);
