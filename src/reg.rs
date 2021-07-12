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
            cache: DistanceCache::new(),
            locations: Vec::new(),
            nodes: Vec::new(),
        }
    }

    pub fn with_capacity(dim: usize, capacity: usize) -> Self {
        Self {
            dim,
            cache: DistanceCache::new(),
            locations: Vec::with_capacity(dim * capacity),
            nodes: Vec::with_capacity(capacity),
        }
    }

    #[inline]
    pub fn cache(&self) -> DistanceCache {
        self.cache
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
    pub fn compute<F>(&mut self, _f: F)
    where
        F: DistanceFunc,
    {
        todo!()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DistanceCache {
    inner: NonNull<InnerCache>,
}

impl DistanceCache {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Default for DistanceCache {
    fn default() -> Self {
        Self {
            inner: NonNull::dangling(),
        }
    }
}

#[derive(Clone, Debug)]
struct InnerCache {
    dim: usize,
    distances: Vec<f64>,
}

#[cfg(test)]
mod tests {}
