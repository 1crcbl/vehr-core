use std::ptr::NonNull;

use crate::{
    tour::{MetaNode, Node, NodeKind},
    traits::{DistanceFunc, NodeIndex},
};

#[derive(Clone, Debug)]
pub struct NodeRegistry<M> {
    dim: usize,
    cache: DistanceCache,
    locations: Vec<f64>,
    nodes: Vec<MetaNode<M>>,
}

impl<M> NodeRegistry<M> {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            cache: DistanceCache::default(),
            locations: Vec::new(),
            nodes: Vec::new(),
        }
    }

    pub fn with_capacity(dim: usize, capacity: usize) -> Self {
        Self {
            dim,
            cache: DistanceCache::default(),
            locations: Vec::with_capacity(dim * capacity),
            nodes: Vec::with_capacity(capacity),
        }
    }

    #[inline]
    pub fn cache(&self) -> &DistanceCache {
        &self.cache
    }

    #[inline]
    pub fn loc<I>(&self, index: I) -> Option<&[f64]>
    where
        I: NodeIndex,
    {
        let left = index.index() * self.dim;
        let right = left + self.dim;
        self.locations.get(left..right)
    }

    #[inline]
    pub fn node<I>(&self, index: I) -> Option<&MetaNode<M>>
    where
        I: NodeIndex,
    {
        self.nodes.get(index.index())
    }

    pub fn add(&mut self, mut location: Vec<f64>, kind: NodeKind, demand: f64, metadata: M) -> Node
    where
        M: Clone,
    {
        let index = self.nodes.len();
        let mn = MetaNode::new(index, kind, demand, metadata);
        let node = mn.node();
        self.nodes.push(mn);
        self.locations.append(&mut location);
        node
    }

    /// Computes the distance matrix between nodes by using the given function ```f``` and stores
    /// the result in a cache.
    pub fn compute<F>(&mut self, f: &F)
    where
        F: DistanceFunc,
    {
        let len = self.nodes.len();
        let len2 = len * len;
        let mut dist = vec![0.; len2];
        let mut nearest = Vec::with_capacity(len2);

        (0..len).for_each(|_| {
            nearest.append(&mut (0..len).collect());
        });

        if self.locations.is_empty() {
            (0..len).for_each(|id1| {
                let offset = id1 * len;
                (0..len).for_each(|id2| {
                    dist[offset + id2] = f.compute(id1, &[], id2, &[]);
                })
            })
        } else {
            self.locations
                .chunks(self.dim)
                .enumerate()
                .for_each(|(id1, loc1)| {
                    let offset = id1 * len;
                    self.locations
                        .chunks(self.dim)
                        .enumerate()
                        .for_each(|(id2, loc2)| {
                            dist[offset + id2] = f.compute(0, loc1, 0, loc2);
                        })
                });
        }

        (0..len).for_each(|id1| {
            let left = id1 * len;
            let right = left + len;

            nearest[left..right].sort_by(|ii, jj| {
                if *ii == *jj {
                    std::cmp::Ordering::Equal
                } else {
                    let pos_x1 = id1 * len;
                    let d_ii = dist[pos_x1 + *ii];
                    let d_jj = dist[pos_x1 + *jj];
                    d_ii.partial_cmp(&d_jj).unwrap()
                }
            });
        });

        self.cache = DistanceCache::new(len, dist, nearest);
    }
}

#[derive(Clone, Debug)]
pub struct DistanceCache {
    inner: Option<NonNull<InnerCache>>,
}

impl DistanceCache {
    pub fn new(len: usize, distances: Vec<f64>, nearest_nb: Vec<usize>) -> Self {
        let inner = Box::new(InnerCache {
            len,
            distances,
            nearest_nb,
        });

        Self {
            inner: NonNull::new(Box::leak(inner)),
        }
    }

    pub fn distance<I>(&self, a: I, b: I) -> f64
    where
        I: NodeIndex,
    {
        match self.inner {
            Some(inner) => unsafe {
                let len = inner.as_ref().len;
                if a.index() < len && b.index() < len {
                    let index = a.index() * len + b.index();
                    inner.as_ref().distances[index]
                } else {
                    0.
                }
            },
            None => panic!("Distance matrix is either uninitialised or already dropped."),
        }
    }

    pub fn nearest<I>(&self, a: I) -> &[usize]
    where
        I: NodeIndex,
    {
        match self.inner {
            Some(inner) => unsafe {
                let len = inner.as_ref().len;
                if a.index() < inner.as_ref().len {
                    let left = a.index() * len;
                    let right = left + len;
                    &inner.as_ref().nearest_nb[left..right]
                } else {
                    &[]
                }
            },
            None => panic!("Distance matrix is either uninitialised or already dropped."),
        }
    }
}

impl Default for DistanceCache {
    fn default() -> Self {
        Self { inner: None }
    }
}

impl Drop for DistanceCache {
    fn drop(&mut self) {
        if let Some(ptr) = std::mem::take(&mut self.inner) {
            unsafe {
                let _ = Box::from_raw(ptr.as_ptr());
            }
        }
    }
}

#[derive(Clone, Debug)]
struct InnerCache {
    len: usize,
    distances: Vec<f64>,
    nearest_nb: Vec<usize>,
}

#[cfg(test)]
mod tests {
    use crate::distance::LowerColDist;

    use super::NodeRegistry;

    #[test]
    fn test_distance() {
        let mut node_reg = NodeRegistry::<()>::new(5);
        node_reg.add(vec![0.; 0], crate::tour::NodeKind::Request, 10., ());
        node_reg.add(vec![0.; 0], crate::tour::NodeKind::Request, 10., ());
        node_reg.add(vec![0.; 0], crate::tour::NodeKind::Request, 10., ());
        node_reg.add(vec![0.; 0], crate::tour::NodeKind::Request, 10., ());
        node_reg.add(vec![0.; 0], crate::tour::NodeKind::Request, 10., ());

        let data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10.];
        let lrd = LowerColDist::new(5, data);

        node_reg.compute(&lrd);
        let cache = node_reg.cache();

        assert_eq!(6., cache.distance(1, 3));
        assert_eq!(6., cache.distance(3, 1));
        assert_eq!(0., cache.distance(2, 2));
        assert_eq!(10., cache.distance(3, 4));
        assert_eq!(10., cache.distance(4, 3));
    }

    #[test]
    fn test_nearest() {
        let mut node_reg = NodeRegistry::<()>::new(5);
        node_reg.add(vec![0.; 0], crate::tour::NodeKind::Request, 10., ());
        node_reg.add(vec![0.; 0], crate::tour::NodeKind::Request, 10., ());
        node_reg.add(vec![0.; 0], crate::tour::NodeKind::Request, 10., ());
        node_reg.add(vec![0.; 0], crate::tour::NodeKind::Request, 10., ());
        node_reg.add(vec![0.; 0], crate::tour::NodeKind::Request, 10., ());

        let data = vec![9., 6., 4., 7., 2., 3., 1., 8., 4., 7.];
        let lrd = LowerColDist::new(5, data);

        node_reg.compute(&lrd);
        let cache = node_reg.cache();

        assert_eq!(&vec![0, 3, 2, 4, 1], cache.nearest(0));
        assert_eq!(&vec![1, 4, 2, 3, 0], cache.nearest(1));
        assert_eq!(&vec![2, 1, 4, 0, 3], cache.nearest(2));
        assert_eq!(&vec![3, 1, 0, 4, 2], cache.nearest(3));
        assert_eq!(&vec![4, 1, 2, 0, 3], cache.nearest(4));
    }
}
