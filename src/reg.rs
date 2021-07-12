use std::ptr::NonNull;

use crate::{
    data::{Node, NodeKind},
    traits::{DistanceFunc, NodeIndex},
};

pub struct NodeRegistry<M> {
    dim: usize,
    cache: DistanceCache,
    locations: Vec<f64>,
    nodes: Vec<Node<M>>,
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
    pub fn cache(&self) -> DistanceCache {
        self.cache.clone()
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
    pub fn node<I>(&self, index: I) -> Option<&Node<M>>
    where
        I: NodeIndex,
    {
        self.nodes.get(index.index())
    }

    pub fn add(
        &mut self,
        mut location: Vec<f64>,
        kind: NodeKind,
        demand: f64,
        metadata: M,
    ) -> Node<M>
    where
        M: Clone,
    {
        let index = self.nodes.len();
        let node = Node::new(index, kind, demand, metadata);
        self.nodes.push(node.clone());
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
        let mut dist = vec![0.; len * len];

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

        self.cache = DistanceCache::new(len, dist);
    }
}

#[derive(Clone, Debug)]
pub struct DistanceCache {
    inner: NonNull<InnerCache>,
}

impl DistanceCache {
    pub fn new(dim: usize, distances: Vec<f64>) -> Self {
        let inner = Box::new(InnerCache { dim, distances });

        Self {
            inner: NonNull::new(Box::leak(inner)).unwrap(),
        }
    }

    pub fn distance<I>(&self, a: I, b: I) -> f64
    where
        I: NodeIndex,
    {
        unsafe {
            let dim = self.inner.as_ref().dim;
            if a.index() < dim && b.index() < dim {
                let index = a.index() * dim + b.index();
                self.inner.as_ref().distances[index]
            } else {
                0.
            }
        }
    }
}

impl Default for DistanceCache {
    fn default() -> Self {
        Self {
            inner: NonNull::dangling(),
        }
    }
}

impl Drop for DistanceCache {
    fn drop(&mut self) {
        todo!()
    }
}

#[derive(Clone, Debug)]
struct InnerCache {
    dim: usize,
    distances: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use crate::traits::LowerColDist;

    use super::NodeRegistry;

    #[test]
    fn test_distance() {
        let mut node_reg = NodeRegistry::<()>::new(5);
        node_reg.add(vec![0.; 0], crate::data::NodeKind::Request, 10., ());
        node_reg.add(vec![0.; 0], crate::data::NodeKind::Request, 10., ());
        node_reg.add(vec![0.; 0], crate::data::NodeKind::Request, 10., ());
        node_reg.add(vec![0.; 0], crate::data::NodeKind::Request, 10., ());
        node_reg.add(vec![0.; 0], crate::data::NodeKind::Request, 10., ());

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
}
