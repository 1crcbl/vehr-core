use std::{collections::HashSet, ptr::NonNull};

use crate::{
    tour::{Node, NodeKind},
    traits::{DistanceFunc, NodeIndex},
};

#[derive(Clone, Debug)]
pub struct NodeRegistry {
    dim: usize,
    cache: DistanceCache,
    locations: Vec<f64>,
    nodes: Vec<Node>,
    depots: HashSet<usize>,
}

impl NodeRegistry {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            cache: DistanceCache::default(),
            locations: Vec::new(),
            nodes: Vec::new(),
            depots: HashSet::new(),
        }
    }

    pub fn with_capacity(dim: usize, capacity: usize, n_depots: usize) -> Self {
        Self {
            dim,
            cache: DistanceCache::default(),
            locations: Vec::with_capacity(dim * capacity),
            nodes: Vec::with_capacity(capacity),
            depots: HashSet::with_capacity(n_depots),
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
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
    pub fn node<I>(&self, index: I) -> Option<&Node>
    where
        I: NodeIndex,
    {
        self.nodes.get(index.index())
    }

    #[inline]
    pub fn depot(&self) -> Option<&Node> {
        match self.depots.iter().take(1).next() {
            Some(id) => self.nodes.get(*id),
            None => None,
        }
    }

    pub fn add(&mut self, mut location: Vec<f64>, kind: NodeKind, demand: f64) -> Node {
        let index = self.nodes.len();
        let node = Node::new(index, kind, demand);
        self.nodes.push(node.clone());
        self.locations.append(&mut location);

        if kind == NodeKind::Depot {
            self.depots.insert(node.index());
        }

        node
    }

    pub fn node_iter(&self) -> std::slice::Iter<Node> {
        self.nodes.iter()
    }

    pub fn node_iter_mut(&mut self) -> std::slice::IterMut<Node> {
        self.nodes.iter_mut()
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

impl Drop for NodeRegistry {
    fn drop(&mut self) {
        unsafe {
            for mut node in self.nodes.drain(..) {
                if let Some(inner) = std::mem::take(&mut node.inner) {
                    (*inner.as_ptr()).route = None;
                    (*inner.as_ptr()).predecessor = None;
                    (*inner.as_ptr()).successor = None;
                    Box::from_raw(inner.as_ptr());
                }
            }

            if let Some(ptr) = std::mem::take(&mut self.cache.inner) {
                let _ = Box::from_raw(ptr.as_ptr());
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct DistanceCache {
    inner: Option<NonNull<InnerCache>>,
}

impl DistanceCache {
    pub(super) fn new(len: usize, distances: Vec<f64>, nearest_nb: Vec<usize>) -> Self {
        let inner = Box::new(InnerCache {
            len,
            distances,
            nearest_nb,
        });

        Self {
            inner: NonNull::new(Box::leak(inner)),
        }
    }

    pub fn distance<I>(&self, a: &I, b: &I) -> f64
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

#[derive(Clone, Debug)]
struct InnerCache {
    len: usize,
    distances: Vec<f64>,
    nearest_nb: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct TourSet {
    hs: HashSet<Vec<usize>>,
}

impl TourSet {
    pub fn new() -> Self {
        Self { hs: HashSet::new() }
    }

    pub fn insert(&mut self, tour: Vec<usize>) {
        self.hs.insert(tour);
    }

    pub fn contains(&self, tour: &[usize]) -> bool {
        self.hs.contains(tour)
    }
}

impl Default for TourSet {
    fn default() -> Self {
        Self { hs: HashSet::new() }
    }
}

#[cfg(test)]
mod tests {
    use crate::distance::LowerColDist;

    use super::{NodeRegistry, TourSet};

    #[test]
    fn test_distance() {
        let mut node_reg = NodeRegistry::new(5);
        node_reg.add(vec![0.; 0], crate::tour::NodeKind::Request, 10.);
        node_reg.add(vec![0.; 0], crate::tour::NodeKind::Request, 10.);
        node_reg.add(vec![0.; 0], crate::tour::NodeKind::Request, 10.);
        node_reg.add(vec![0.; 0], crate::tour::NodeKind::Request, 10.);
        node_reg.add(vec![0.; 0], crate::tour::NodeKind::Request, 10.);

        let data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10.];
        let lrd = LowerColDist::new(5, data);

        node_reg.compute(&lrd);
        let cache = node_reg.cache();

        assert_eq!(6., cache.distance(&1, &3));
        assert_eq!(6., cache.distance(&3, &1));
        assert_eq!(0., cache.distance(&2, &2));
        assert_eq!(10., cache.distance(&3, &4));
        assert_eq!(10., cache.distance(&4, &3));
    }

    #[test]
    fn test_nearest() {
        let mut node_reg = NodeRegistry::new(5);
        node_reg.add(vec![0.; 0], crate::tour::NodeKind::Request, 10.);
        node_reg.add(vec![0.; 0], crate::tour::NodeKind::Request, 10.);
        node_reg.add(vec![0.; 0], crate::tour::NodeKind::Request, 10.);
        node_reg.add(vec![0.; 0], crate::tour::NodeKind::Request, 10.);
        node_reg.add(vec![0.; 0], crate::tour::NodeKind::Request, 10.);

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

    #[test]
    fn test_tourset() {
        let mut tourset = TourSet::new();
        tourset.insert(vec![0, 1, 2, 3, 0, 4, 5, 6]);
        assert!(tourset.contains(&vec![0, 1, 2, 3, 0, 4, 5, 6]));
        assert!(!tourset.contains(&vec![0, 1, 2, 4, 6, 0, 3, 5]));
    }
}
