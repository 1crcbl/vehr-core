use std::{collections::BinaryHeap, ptr::NonNull};

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
    pub fn node(&self) -> &Node {
        &self.data
    }

    #[inline]
    pub fn metadata(&self) -> &M {
        &self.meta
    }

    #[inline]
    pub(crate) fn into_value(self) -> (Node, M) {
        (self.data, self.meta)
    }
}

impl<M> AsRef<Node> for MetaNode<M> {
    fn as_ref(&self) -> &Node {
        &self.data
    }
}

impl<M> AsMut<Node> for MetaNode<M> {
    fn as_mut(&mut self) -> &mut Node {
        &mut self.data
    }
}

#[derive(Clone, Debug)]
pub struct Node {
    pub(crate) inner: Option<NonNull<InnerNode>>,
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

    fn pos(&self) -> NodePos {
        match self.inner {
            Some(inner) => unsafe {
                if let Some(r) = inner.as_ref().route {
                    let rinner = inner.as_ref();

                    match (
                        rinner.predecessor.is_some(),
                        rinner.successor.is_some(),
                        r.as_ref().rev,
                    ) {
                        (true, true, _) => NodePos::Interior,
                        (true, false, true) => NodePos::First,
                        (true, false, false) => NodePos::Last,
                        (false, true, true) => NodePos::Last,
                        (false, true, false) => NodePos::First,
                        (false, false, _) => NodePos::First,
                    }
                } else {
                    NodePos::None
                }
            },
            None => panic_ptr!("Node"),
        }
    }
}

impl Default for Node {
    fn default() -> Self {
        Self { inner: None }
    }
}

impl AsRef<Node> for Node {
    fn as_ref(&self) -> &Node {
        self
    }
}

impl AsMut<Node> for Node {
    fn as_mut(&mut self) -> &mut Node {
        self
    }
}

impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl Eq for Node {}

#[derive(Clone, Debug)]
pub(crate) struct InnerNode {
    index: usize,
    demand: f64,
    kind: NodeKind,
    pub(crate) route: Option<NonNull<InnerRoute>>,
    pub(crate) predecessor: Option<NonNull<InnerNode>>,
    pub(crate) successor: Option<NonNull<InnerNode>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Hash)]
pub enum NodeKind {
    Depot,
    Request,
}

#[derive(Clone, Copy, Debug, PartialEq, Hash)]
enum NodePos {
    First,
    Interior,
    Last,
    None,
}

#[derive(Clone, Debug)]
pub struct Route {
    inner: Option<NonNull<InnerRoute>>,
}

impl Route {
    pub fn new<N>(depot: N, vehicle_capacity: f64) -> Self
    where
        N: AsRef<Node>,
    {
        let inner = Box::new(InnerRoute {
            n_nodes: 0,
            capacity: vehicle_capacity,
            load: 0.,
            depot: depot.as_ref().inner,
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
            Some(inner) => unsafe {
                let tmp = if inner.as_ref().depot.is_some() { 1 } else { 0 };
                inner.as_ref().n_nodes + tmp
            },
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

    /// Moves all nodes from `other` into the front of `Self`, taking into account the traversal direction.
    pub fn append_front(&mut self, _other: &mut Self) {
        todo!()
    }

    /// Inserts a node at the end of the route.
    #[inline]
    pub fn push_back(&mut self, node: &Node) {
        Self::push_back_(&self.inner, node);
    }

    fn push_back_(route: &Option<NonNull<InnerRoute>>, node: &Node) {
        match (route, node.inner) {
            (Some(inner), Some(node_inner)) => unsafe {
                (*node_inner.as_ptr()).route = *route;

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

    /// Inserts a node at the beginning of the route.
    #[inline]
    pub fn push_front(&mut self, node: &Node) {
        Self::push_front_(&self.inner, node);
    }

    fn push_front_(route: &Option<NonNull<InnerRoute>>, node: &Node) {
        match (route, node.inner) {
            (Some(inner), Some(node_inner)) => unsafe {
                (*node_inner.as_ptr()).route = *route;

                if inner.as_ref().first.is_none() {
                    (*inner.as_ptr()).first = node.inner;
                    (*inner.as_ptr()).last = node.inner;
                } else if inner.as_ref().rev {
                    (*node_inner.as_ptr()).predecessor = inner.as_ref().last;
                    (*inner.as_ref().last.unwrap().as_ptr()).successor = node.inner;
                    (*inner.as_ptr()).last = node.inner;
                } else {
                    (*node_inner.as_ptr()).successor = inner.as_ref().first;
                    (*inner.as_ref().first.unwrap().as_ptr()).predecessor = node.inner;
                    (*inner.as_ptr()).first = node.inner;
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

                if let Some(x) = inner.as_ref().depot {
                    result.push(x.as_ref().index)
                }

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
                Box::from_raw(inner.as_ptr());
            }
        }
        self.inner = None;
    }
}

#[derive(Clone, Debug)]
pub(crate) struct InnerRoute {
    n_nodes: usize,
    /// Vehicle's capacity in the current route.
    capacity: f64,
    load: f64,
    depot: Option<NonNull<InnerNode>>,
    first: Option<NonNull<InnerNode>>,
    last: Option<NonNull<InnerNode>>,
    rev: bool,
}

#[derive(Clone, Debug)]
pub struct Tour<M> {
    vehicle_capacity: f64,
    reg: NodeRegistry<M>,
    routes: Vec<Route>,
}

impl<M> Tour<M> {
    pub fn new(reg: NodeRegistry<M>, vehicle_capacity: f64) -> Self {
        Self {
            vehicle_capacity,
            reg,
            routes: Vec::with_capacity(0),
        }
    }

    pub fn init_cw(&mut self) {
        let depot = self.reg.depot().expect("There must be at least one depot.");
        let len = self.reg.len();
        let mut routes = Vec::with_capacity(len - 1);

        let mut bh = BinaryHeap::with_capacity(len * len);
        let cache = self.reg.cache();

        for mn1 in self.reg.node_iter() {
            let node = mn1.node();
            if node.kind() != NodeKind::Depot {
                let mut r = Route::new(depot, self.vehicle_capacity);
                r.push_back(&node);
                routes.push(r);
            }

            let d_depot1 = cache.distance(mn1, depot);
            for mn2 in self.reg.node_iter() {
                let d_depot2 = cache.distance(mn2, depot);
                let d_12 = cache.distance(mn1, mn2);
                let saving = d_depot1 + d_depot2 - d_12;
                bh.push(SavingPair::new(mn1.node(), mn2.node(), saving));
            }
        }

        while let Some(pair) = bh.pop() {
            let (node1, node2, _) = pair.into_value();

            match (node1.pos(), node2.pos()) {
                (NodePos::First, NodePos::First) => todo!(),
                (NodePos::First, NodePos::Last) => todo!(),
                (NodePos::Last, NodePos::First) => todo!(),
                (NodePos::Last, NodePos::Last) => todo!(),
                _ => {}
            }
        }

        self.routes = routes;
    }
}

#[derive(Debug)]
struct SavingPair<'s> {
    node1: &'s Node,
    node2: &'s Node,
    saving: f64,
}

impl<'s> SavingPair<'s> {
    pub fn new(node1: &'s Node, node2: &'s Node, saving: f64) -> Self {
        Self {
            node1,
            node2,
            saving,
        }
    }

    pub fn into_value(self) -> (&'s Node, &'s Node, f64) {
        (self.node1, self.node2, self.saving)
    }
}

impl<'s> PartialEq for SavingPair<'s> {
    fn eq(&self, other: &Self) -> bool {
        self.node1 == other.node1 && self.node2 == other.node2
    }
}

impl<'s> Eq for SavingPair<'s> {}

impl<'s> PartialOrd for SavingPair<'s> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if self == other {
            Some(std::cmp::Ordering::Equal)
        } else if self.saving < other.saving {
            Some(std::cmp::Ordering::Less)
        } else {
            Some(std::cmp::Ordering::Greater)
        }
    }
}

impl<'s> Ord for SavingPair<'s> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if self == other {
            std::cmp::Ordering::Equal
        } else if self.saving < other.saving {
            std::cmp::Ordering::Less
        } else {
            std::cmp::Ordering::Greater
        }
    }
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

    #[test]
    fn test_push_front() {
        let depot = Node::new(0, NodeKind::Depot, 0.);
        let mut route = Route::new(&depot, 1000.);

        let nodes: Vec<_> = (1..=10)
            .map(|ii| Node::new(ii, NodeKind::Request, 10.))
            .collect();
        nodes.iter().take(5).for_each(|node| route.push_front(node));
        route.rev(true);
        nodes
            .iter()
            .rev()
            .take(5)
            .for_each(|node| route.push_front(node));

        assert_eq!(100., route.load());
        assert_eq!(11, route.n_nodes());

        assert_eq!(&vec![0, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5], &route.index_vec());

        route.rev(false);
        assert_eq!(&vec![0, 5, 4, 3, 2, 1, 10, 9, 8, 7, 6], &route.index_vec());
    }
}
