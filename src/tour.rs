use std::ptr::NonNull;

use crate::reg::NodeRegistry;

// TODO: impl Drop for all Inner structs.

macro_rules! panic_ptr {
    ($name:expr) => {
        panic!("{} is either uninitialised or already dropped.", $name);
    };
}

#[derive(Clone, Debug)]
pub struct MetaNode<M> {
    data: Node,
    meta: M,
}

impl<M> MetaNode<M> {
    pub(crate) fn new(index: usize, kind: NodeKind, demand: f64, meta: M) -> Self {
        let data = Node::new(index, kind, demand);
        Self { data, meta }
    }

    #[inline]
    pub(crate) fn get_index(&self) -> usize {
        self.data.get_index()
    }

    #[inline]
    pub fn node(&self) -> Node {
        self.data.clone()
    }

    #[inline]
    pub fn metadata(&self) -> &M {
        &self.meta
    }
}

#[derive(Clone, Debug)]
pub struct Node {
    inner: Option<NonNull<InnerNode>>,
}

impl Node {
    pub(super) fn new(index: usize, kind: NodeKind, demand: f64) -> Self {
        let inner = Box::new(InnerNode {
            index,
            demand,
            kind,
            route: None,
            predecessor: None,
            successor: None,
        });

        Self {
            inner: NonNull::new(Box::leak(inner)),
        }
    }

    #[inline]
    pub(crate) fn get_index(&self) -> usize {
        match self.inner {
            Some(inner) => unsafe { inner.as_ref().index },
            None => panic_ptr!("Node"),
        }
    }

    #[inline]
    pub fn demand(&self) -> f64 {
        match self.inner {
            Some(inner) => unsafe { inner.as_ref().demand },
            None => panic_ptr!("Node"),
        }
    }

    #[inline]
    pub fn kind(&self) -> NodeKind {
        match self.inner {
            Some(inner) => unsafe { inner.as_ref().kind },
            None => panic_ptr!("Node"),
        }
    }
}

impl Default for Node {
    fn default() -> Self {
        Self { inner: None }
    }
}

impl Drop for Node {
    fn drop(&mut self) {
        if let Some(inner) = self.inner {
            unsafe {
                (*inner.as_ptr()).route = None;
                (*inner.as_ptr()).predecessor = None;
                (*inner.as_ptr()).successor = None;
            }
            Box::new(inner.as_ptr());
        }
        self.inner = None;
    }
}

#[derive(Clone, Debug)]
struct InnerNode {
    index: usize,
    demand: f64,
    kind: NodeKind,
    route: Option<NonNull<InnerRoute>>,
    predecessor: Option<NonNull<InnerNode>>,
    successor: Option<NonNull<InnerNode>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Hash)]
pub enum NodeKind {
    Depot,
    Request,
}

#[derive(Clone, Debug)]
pub struct Route {
    inner: Option<NonNull<InnerRoute>>,
}

impl Route {
    pub fn new(depot: &Node, vehicle_capacity: f64) -> Self {
        let inner = Box::new(InnerRoute {
            n_nodes: 0,
            capacity: vehicle_capacity,
            load: 0.,
            depot: depot.inner,
            first: None,
            last: None,
            rev: false,
        });

        Self {
            inner: NonNull::new(Box::leak(inner)),
        }
    }

    #[inline]
    pub fn capacity(&self) -> f64 {
        match self.inner {
            Some(inner) => unsafe { inner.as_ref().capacity },
            None => panic_ptr!("Route"),
        }
    }

    #[inline]
    pub fn load(&self) -> f64 {
        match self.inner {
            Some(inner) => unsafe { inner.as_ref().load },
            None => panic_ptr!("Route"),
        }
    }

    #[inline]
    pub fn n_nodes(&self) -> usize {
        match self.inner {
            Some(inner) => unsafe { inner.as_ref().n_nodes + 1 },
            None => panic_ptr!("Route"),
        }
    }

    pub fn check_capacity(&self, demand: f64) -> bool {
        match self.inner {
            Some(inner) => unsafe {
                let tmp_load = inner.as_ref().load + demand;
                tmp_load <= inner.as_ref().capacity
            },
            None => panic_ptr!("Route"),
        }
    }

    pub fn rev(&mut self, rev: bool) {
        match self.inner {
            Some(inner) => unsafe { (*inner.as_ptr()).rev = rev },
            None => panic_ptr!("Route"),
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        match self.inner {
            Some(inner) => unsafe {
                let inner = inner.as_ref();
                inner.first.is_none() && inner.last.is_none()
            },
            None => panic_ptr!("Route"),
        }
    }

    pub fn push_back(&mut self, node: &Node) {
        match (self.inner, node.inner) {
            (Some(inner), Some(node_inner)) => unsafe {
                (*node_inner.as_ptr()).route = self.inner;

                if inner.as_ref().first.is_none() {
                    (*inner.as_ptr()).first = node.inner;
                    (*inner.as_ptr()).last = node.inner;
                } else if inner.as_ref().rev {
                    (*node_inner.as_ptr()).successor = inner.as_ref().first;
                    (*inner.as_ref().first.unwrap().as_ptr()).predecessor = node.inner;
                    (*inner.as_ptr()).first = node.inner;
                } else {
                    (*node_inner.as_ptr()).predecessor = inner.as_ref().last;
                    (*inner.as_ref().last.unwrap().as_ptr()).successor = node.inner;
                    (*inner.as_ptr()).last = node.inner;
                }

                (*inner.as_ptr()).load += node_inner.as_ref().demand;
                (*inner.as_ptr()).n_nodes += 1;
            },
            _ => panic_ptr!("Node or route"),
        };
    }

    pub fn index_vec(&self) -> Vec<usize> {
        match self.inner {
            Some(inner) => unsafe {
                let mut result = Vec::with_capacity(self.n_nodes());
                result.push(inner.as_ref().depot.unwrap().as_ref().index);

                let rev = inner.as_ref().rev;

                let mut node = if rev {
                    inner.as_ref().last
                } else {
                    inner.as_ref().first
                };

                while let Some(tmp) = node {
                    result.push(tmp.as_ref().index);
                    if rev {
                        node = tmp.as_ref().predecessor;
                    } else {
                        node = tmp.as_ref().successor;
                    }
                }

                result
            },
            None => panic_ptr!("Route"),
        }
    }
}

impl Default for Route {
    fn default() -> Self {
        Self { inner: None }
    }
}

impl Drop for Route {
    fn drop(&mut self) {
        if let Some(inner) = self.inner {
            unsafe {
                (*inner.as_ptr()).depot = None;
                (*inner.as_ptr()).first = None;
                (*inner.as_ptr()).last = None;
                Box::new(inner.as_ptr());
            }
        }
        self.inner = None;
    }
}

#[derive(Clone, Debug)]
pub struct Tour<M> {
    reg: NodeRegistry<M>,
    route: Vec<Route>,
}

impl<M> Tour<M> {
    pub fn new(reg: NodeRegistry<M>) -> Self {
        Self {
            reg,
            route: Vec::with_capacity(0),
        }
    }

    pub fn init_cw(&mut self) {}
}

#[derive(Clone, Debug)]
struct InnerRoute {
    n_nodes: usize,
    /// Vehicle's capacity in the current route.
    capacity: f64,
    load: f64,
    depot: Option<NonNull<InnerNode>>,
    first: Option<NonNull<InnerNode>>,
    last: Option<NonNull<InnerNode>>,
    rev: bool,
}

#[cfg(test)]
mod tests {
    use super::{Node, NodeKind, Route};

    #[test]
    fn test_push_back() {
        let depot = Node::new(0, NodeKind::Depot, 0.);
        let mut route = Route::new(&depot, 1000.);

        let nodes: Vec<_> = (1..=10)
            .map(|ii| Node::new(ii, NodeKind::Request, 10.))
            .collect();
        nodes.iter().take(5).for_each(|node| route.push_back(node));
        route.rev(true);
        nodes
            .iter()
            .rev()
            .take(5)
            .for_each(|node| route.push_back(node));

        assert_eq!(100., route.load());
        assert_eq!(11, route.n_nodes());

        assert_eq!(&vec![0, 5, 4, 3, 2, 1, 10, 9, 8, 7, 6], &route.index_vec());

        route.rev(false);
        assert_eq!(&vec![0, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5], &route.index_vec());
    }
}
