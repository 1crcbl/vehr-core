use crate::{
    distance::{EucDistance, FullMtxDist, LowerColDist},
    tour::{MetaNode, Node},
};

pub trait NodeIndex {
    fn index(&self) -> usize;
}

impl NodeIndex for usize {
    fn index(&self) -> usize {
        *self
    }
}

impl NodeIndex for Node {
    fn index(&self) -> usize {
        self.get_index()
    }
}

impl<M> NodeIndex for MetaNode<M> {
    fn index(&self) -> usize {
        self.get_index()
    }
}

pub trait DistanceFunc {
    fn compute(&self, index_a: usize, a: &[f64], index_b: usize, b: &[f64]) -> f64;
}

impl DistanceFunc for EucDistance {
    fn compute(&self, _: usize, a: &[f64], _: usize, b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .fold(0_f64, |acc, (x1, x2)| acc + (x1 - x2).powi(2))
            .sqrt()
    }
}

impl DistanceFunc for FullMtxDist {
    fn compute(&self, index_a: usize, _: &[f64], index_b: usize, _: &[f64]) -> f64 {
        if index_a >= self.dim || index_b >= self.dim {
            0.
        } else {
            self.data[index_a * self.dim + index_b]
        }
    }
}

impl DistanceFunc for LowerColDist {
    fn compute(&self, index_a: usize, _: &[f64], index_b: usize, _: &[f64]) -> f64 {
        if index_a >= self.dim || index_b >= self.dim || index_a == index_b {
            0.
        } else {
            let ii = if index_a < index_b {
                index_a * self.dim + index_b - (((index_a + 2) * (index_a + 1)) >> 1)
            } else {
                index_b * self.dim + index_a - (((index_b + 2) * (index_b + 1)) >> 1)
            };
            self.data[ii]
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::traits::{DistanceFunc, FullMtxDist, LowerColDist};

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

        let fmd = FullMtxDist::new(5, data);
        assert_eq!(6., fmd.compute(1, &[], 3, &[]));
        assert_eq!(6., fmd.compute(3, &[], 1, &[]));
        assert_eq!(0., fmd.compute(2, &[], 2, &[]));
        assert_eq!(10., fmd.compute(3, &[], 4, &[]));
        assert_eq!(10., fmd.compute(4, &[], 3, &[]));
    }

    #[test]
    fn test_lower_row() {
        let data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10.];
        let lrd = LowerColDist::new(5, data);
        assert_eq!(6., lrd.compute(1, &[], 3, &[]));
        assert_eq!(6., lrd.compute(3, &[], 1, &[]));
        assert_eq!(0., lrd.compute(2, &[], 2, &[]));
        assert_eq!(10., lrd.compute(3, &[], 4, &[]));
        assert_eq!(10., lrd.compute(4, &[], 3, &[]));
    }
}
