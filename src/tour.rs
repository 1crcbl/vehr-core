use std::{collections::BinaryHeap, marker::PhantomData, ptr::NonNull};

use crate::{
    reg::{DistanceCache, NodeRegistry},
    traits::NodeIndex,
};

macro_rules! panic_ptr {
    ($name:expr) => {
        panic!("{} is either uninitialised or already dropped.", $name);
    };
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

    pub(crate) fn from_nonnull(ptr: Option<NonNull<InnerNode>>) -> Self {
        Self { inner: ptr }
    }

    #[inline]
    pub(crate) fn get_index(&self) -> usize {
        unsafe { self.inner.unwrap().as_ref().index }
    }

    #[inline]
    pub fn demand(&self) -> f64 {
        unsafe { self.inner.unwrap().as_ref().demand }
    }

    #[inline]
    pub fn kind(&self) -> NodeKind {
        unsafe { self.inner.unwrap().as_ref().kind }
    }

    #[inline]
    unsafe fn pos(&self) -> NodePos {
        match self.inner {
            Some(inner) => match inner.as_ref().route {
                Some(route) => {
                    if let Some(pred) = inner.as_ref().predecessor {
                        if pred.as_ref().kind == NodeKind::Depot {
                            if route.as_ref().rev {
                                return NodePos::Last;
                            } else {
                                return NodePos::First;
                            }
                        }
                    }

                    if let Some(succ) = inner.as_ref().successor {
                        if succ.as_ref().kind == NodeKind::Depot {
                            if route.as_ref().rev {
                                return NodePos::First;
                            } else {
                                return NodePos::Last;
                            }
                        }
                    }

                    NodePos::Interior
                }
                None => NodePos::None,
            },
            None => NodePos::None,
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

#[derive(Debug)]
pub struct Route {
    inner: Option<NonNull<InnerRoute>>,
}

impl Route {
    pub fn new<N>(depot: N, vehicle_capacity: f64, cache: &DistanceCache) -> Self
    where
        N: AsRef<Node>,
    {
        let inner = Box::new(InnerRoute {
            n_nodes: 0,
            capacity: vehicle_capacity,
            load: 0.,
            first: None,
            last: None,
            cache: cache.clone(),
            rev: false,
            has_depot: depot.as_ref().inner.is_some(),
        });

        let route = Self {
            inner: NonNull::new(Box::leak(inner)),
        };

        unsafe {
            // Create new depot
            let depot_ref = depot.as_ref().inner.unwrap().as_ref();
            let new_depot = Box::new(InnerNode {
                index: depot_ref.index,
                demand: depot_ref.demand,
                kind: depot_ref.kind,
                route: route.inner,
                predecessor: None,
                successor: None,
            });
            let new_depot = NonNull::new(Box::leak(new_depot));
            Self::push_(&route.inner, &new_depot, false);
        }

        route
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
            Some(inner) => unsafe { inner.as_ref().n_nodes },
            None => panic_ptr!("Route"),
        }
    }

    #[inline]
    pub fn check_capacity(&self, demand: f64) -> bool {
        match self.inner {
            Some(inner) => unsafe {
                let tmp_load = inner.as_ref().load + demand;
                tmp_load <= inner.as_ref().capacity
            },
            None => panic_ptr!("Route"),
        }
    }

    #[inline]
    unsafe fn check_capacity_(
        route1: &Option<NonNull<InnerRoute>>,
        route2: &Option<NonNull<InnerRoute>>,
    ) -> bool {
        match (route1, route2) {
            (Some(inner1), Some(inner2)) => {
                let tmp_load = inner1.as_ref().load + inner2.as_ref().load;
                tmp_load <= inner1.as_ref().capacity
            }
            _ => panic_ptr!("Route"),
        }
    }

    #[inline]
    pub fn rev(&mut self, rev: bool) {
        unsafe {
            Self::rev_(&self.inner, rev);
        }
    }

    #[inline]
    unsafe fn rev_(route: &Option<NonNull<InnerRoute>>, rev: bool) {
        match route {
            Some(inner) => (*inner.as_ptr()).rev = rev,
            None => panic_ptr!("Route"),
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        match self.inner {
            Some(inner) => unsafe {
                inner.as_ref().n_nodes == 0
                    || (inner.as_ref().n_nodes == 1 && inner.as_ref().has_depot)
            },
            None => panic_ptr!("Route"),
        }
    }

    /// Moves all nodes from `other` into the front of `Self`, taking into account the traversal direction.
    #[inline]
    pub fn append_front(&mut self, other: &mut Self) {
        unsafe {
            Self::append_front_(&mut self.inner, &mut other.inner);
        }
    }

    #[inline]
    unsafe fn append_front_(
        route1: &mut Option<NonNull<InnerRoute>>,
        route2: &mut Option<NonNull<InnerRoute>>,
    ) {
        if route1 != route2 {
            if let Some(inner) = route2 {
                if inner.as_ref().rev {
                    while let Some(inner_node) = Route::pop_first_(route2) {
                        Self::push_(route1, &Some(inner_node), true);
                    }
                } else {
                    while let Some(inner_node) = Route::pop_last_(route2) {
                        Self::push_(route1, &Some(inner_node), true);
                    }
                }
            }
        }
    }

    #[inline]
    /// Moves all nodes from `other` into the back of `Self`, taking into account the traversal direction.
    pub fn append_back(&mut self, other: &mut Self) {
        unsafe {
            Self::append_back_(&mut self.inner, &mut other.inner);
        }
    }

    #[inline]
    unsafe fn append_back_(
        route1: &mut Option<NonNull<InnerRoute>>,
        route2: &mut Option<NonNull<InnerRoute>>,
    ) {
        if route1 != route2 {
            if let Some(inner) = route2 {
                if inner.as_ref().rev {
                    while let Some(inner_node) = Route::pop_last_(route2) {
                        Self::push_(route1, &Some(inner_node), false);
                    }
                } else {
                    while let Some(inner_node) = Route::pop_first_(route2) {
                        Self::push_(route1, &Some(inner_node), false);
                    }
                }
            }
        }
    }

    /// Inserts a node at the end of the route.
    #[inline]
    pub fn push_back(&mut self, node: &Node) {
        unsafe {
            Self::push_(&self.inner, &node.inner, false);
        }
    }

    /// Inserts a node at the beginning of the route.
    #[inline]
    pub fn push_front(&mut self, node: &Node) {
        unsafe {
            Self::push_(&self.inner, &node.inner, true);
        }
    }

    // flag: false: back | true: front
    #[inline]
    unsafe fn push_(
        route: &Option<NonNull<InnerRoute>>,
        node: &Option<NonNull<InnerNode>>,
        flag: bool,
    ) {
        match (route, node) {
            (Some(inner), Some(node_inner)) => {
                (*node_inner.as_ptr()).route = *route;

                match inner.as_ref().first {
                    Some(first) => {
                        let flag = flag ^ inner.as_ref().rev;

                        if flag {
                            // push back  & rev = true
                            // push front & rev = false
                            if let Some(second) = first.as_ref().successor {
                                (*node_inner.as_ptr()).successor = first.as_ref().successor;
                                (*second.as_ptr()).predecessor = *node;
                            } else {
                                (*node_inner.as_ptr()).successor = inner.as_ref().last;
                                (*inner.as_ref().last.unwrap().as_ptr()).predecessor = *node;
                                (*inner.as_ptr()).last = *node;
                            }

                            (*first.as_ptr()).successor = *node;
                            (*node_inner.as_ptr()).predecessor = inner.as_ref().first;
                        } else {
                            // push back  & rev = false
                            // push front & rev = true
                            (*first.as_ptr()).predecessor = *node;
                            (*node_inner.as_ptr()).successor = inner.as_ref().first;
                            (*node_inner.as_ptr()).predecessor = inner.as_ref().last;
                            (*inner.as_ref().last.unwrap().as_ptr()).successor = *node;
                            (*inner.as_ptr()).last = *node;
                        }
                    }
                    None => {
                        (*inner.as_ptr()).first = *node;
                        (*inner.as_ptr()).last = *node;
                    }
                }

                (*inner.as_ptr()).load += node_inner.as_ref().demand;
                (*inner.as_ptr()).n_nodes += 1;
            }
            _ => panic_ptr!("Node or route"),
        };
    }

    #[inline]
    pub fn pop_back(&mut self) -> Option<Node> {
        unsafe {
            let result = Route::pop_back_(&mut self.inner);
            if result.is_some() {
                Some(Node { inner: result })
            } else {
                None
            }
        }
    }

    #[inline]
    unsafe fn pop_back_(route: &mut Option<NonNull<InnerRoute>>) -> Option<NonNull<InnerNode>> {
        match route {
            Some(inner) => {
                if inner.as_ref().rev {
                    Route::pop_first_(route)
                } else {
                    Route::pop_last_(route)
                }
            }
            None => None,
        }
    }

    #[inline]
    pub fn pop_front(&mut self) -> Option<Node> {
        unsafe {
            let result = Route::pop_front_(&mut self.inner);
            if result.is_some() {
                Some(Node { inner: result })
            } else {
                None
            }
        }
    }

    #[inline]
    unsafe fn pop_front_(route: &mut Option<NonNull<InnerRoute>>) -> Option<NonNull<InnerNode>> {
        match route {
            Some(inner) => {
                if inner.as_ref().rev {
                    Route::pop_last_(route)
                } else {
                    Route::pop_first_(route)
                }
            }
            None => None,
        }
    }

    #[inline]
    unsafe fn pop_first_(route: &mut Option<NonNull<InnerRoute>>) -> Option<NonNull<InnerNode>> {
        match *route {
            Some(inner) => {
                if inner.as_ref().n_nodes == 0 {
                    None
                } else {
                    let first = inner.as_ref().first.unwrap();
                    if inner.as_ref().n_nodes == 1 && first.as_ref().kind == NodeKind::Depot {
                        None
                    } else {
                        let result = first.as_ref().successor;
                        if let Some(inner_node) = result {
                            (*inner.as_ptr()).load -= inner_node.as_ref().demand;
                            (*inner.as_ptr()).n_nodes -= 1;

                            match inner_node.as_ref().predecessor {
                                Some(pred) => {
                                    (*pred.as_ptr()).successor = (*inner_node.as_ptr()).successor
                                }
                                None => (*inner.as_ptr()).first = None,
                            }

                            match inner_node.as_ref().successor {
                                Some(pred) => {
                                    (*pred.as_ptr()).predecessor =
                                        (*inner_node.as_ptr()).predecessor
                                }
                                None => (*inner.as_ptr()).first = None,
                            }

                            (*inner_node.as_ptr()).route = None;
                            (*inner_node.as_ptr()).predecessor = None;
                            (*inner_node.as_ptr()).successor = None;
                        }
                        result
                    }
                }
            }
            None => None,
        }
    }

    #[inline]
    unsafe fn pop_last_(route: &Option<NonNull<InnerRoute>>) -> Option<NonNull<InnerNode>> {
        match route {
            Some(inner) => {
                if inner.as_ref().n_nodes == 0 {
                    None
                } else {
                    let first = inner.as_ref().first.unwrap();
                    if inner.as_ref().n_nodes == 1 && first.as_ref().kind == NodeKind::Depot {
                        None
                    } else {
                        let result = first.as_ref().predecessor;
                        if let Some(inner_node) = result {
                            (*inner.as_ptr()).load -= inner_node.as_ref().demand;
                            (*inner.as_ptr()).n_nodes -= 1;

                            match inner_node.as_ref().predecessor {
                                Some(pred) => {
                                    (*pred.as_ptr()).successor = (*inner_node.as_ptr()).successor
                                }
                                None => (*inner.as_ptr()).first = None,
                            }

                            match inner_node.as_ref().successor {
                                Some(pred) => {
                                    (*pred.as_ptr()).predecessor =
                                        (*inner_node.as_ptr()).predecessor
                                }
                                None => (*inner.as_ptr()).first = None,
                            }

                            (*inner_node.as_ptr()).route = None;
                            (*inner_node.as_ptr()).predecessor = None;
                            (*inner_node.as_ptr()).successor = None;
                        }
                        result
                    }
                }
            }
            None => None,
        }
    }

    #[inline]
    pub fn insert_back(pivot: &mut Node, node: &mut Node) {
        if let (Some(innerp), Some(innern)) = (pivot.inner, node.inner) {
            unsafe {
                if let Some(route) = innerp.as_ref().route {
                    (*route.as_ptr()).n_nodes += 1;
                    (*route.as_ptr()).load += innern.as_ref().demand;
                    (*innern.as_ptr()).route = innerp.as_ref().route;

                    if route.as_ref().rev {
                        let old_pred = innerp.as_ref().predecessor;
                        (*innerp.as_ptr()).predecessor = node.inner;
                        (*innern.as_ptr()).successor = pivot.inner;

                        if let Some(pred) = old_pred {
                            (*innern.as_ptr()).predecessor = old_pred;
                            (*pred.as_ptr()).successor = node.inner;
                        } else {
                            (*route.as_ptr()).first = node.inner;
                        }
                    } else {
                        let old_succ = innerp.as_ref().successor;
                        (*innerp.as_ptr()).successor = node.inner;
                        (*innern.as_ptr()).predecessor = pivot.inner;

                        if let Some(succ) = old_succ {
                            (*innern.as_ptr()).successor = old_succ;
                            (*succ.as_ptr()).predecessor = node.inner;
                        } else {
                            (*route.as_ptr()).last = node.inner;
                        }
                    }
                }
            }
        }
    }

    pub fn eject(node: &mut Node) {
        if let Some(inner) = node.inner {
            unsafe {
                if let Some(route) = inner.as_ref().route {
                    (*route.as_ptr()).n_nodes -= 1;
                    (*route.as_ptr()).load -= inner.as_ref().demand;

                    if let Some(pred) = inner.as_ref().predecessor {
                        (*pred.as_ptr()).successor = inner.as_ref().successor;
                    } else {
                        (*route.as_ptr()).first = inner.as_ref().successor;
                    }

                    if let Some(succ) = inner.as_ref().successor {
                        (*succ.as_ptr()).predecessor = inner.as_ref().predecessor;
                    } else {
                        (*route.as_ptr()).last = inner.as_ref().predecessor;
                    }

                    (*inner.as_ptr()).route = None;
                    (*inner.as_ptr()).predecessor = None;
                    (*inner.as_ptr()).successor = None;
                }
            }
        }
    }

    #[inline]
    pub fn total_distance(&self) -> f64 {
        let mut result = 0.;

        if let Some(inner) = self.inner {
            unsafe {
                let cache = &inner.as_ref().cache;
                for arc in self.arc_iter() {
                    result += cache.distance(arc.tail(), arc.head());
                }
            }
        }

        result
    }

    pub fn index_vec(&self) -> Vec<usize> {
        match self.inner {
            Some(_) => {
                let mut result = Vec::with_capacity(self.n_nodes());
                self.node_iter().for_each(|node| result.push(node.index()));
                result
            }
            None => panic_ptr!("Route"),
        }
    }

    pub fn node_iter(&self) -> NodeIter {
        match self.inner {
            Some(inner) => unsafe {
                let rev = inner.as_ref().rev;
                let node = inner.as_ref().first;
                NodeIter::new(inner.as_ref().n_nodes, rev, node)
            },
            None => NodeIter::default(),
        }
    }

    pub fn arc_iter(&self) -> ArcIter {
        match self.inner {
            Some(inner) => unsafe {
                let n_nodes = inner.as_ref().n_nodes;
                let len = if n_nodes <= 1 {
                    0
                } else if n_nodes == 2 {
                    1
                } else {
                    n_nodes
                };

                ArcIter::new(inner.as_ref().rev, len, inner.as_ref().first)
            },
            None => ArcIter::default(),
        }
    }
}

impl Default for Route {
    fn default() -> Self {
        Self { inner: None }
    }
}

pub struct NodeIter<'s> {
    len: usize,
    rev: bool,
    node: Option<NonNull<InnerNode>>,
    phantom: PhantomData<&'s ()>,
}

impl<'s> NodeIter<'s> {
    fn new(len: usize, rev: bool, node: Option<NonNull<InnerNode>>) -> Self {
        Self {
            len,
            rev,
            node,
            phantom: PhantomData,
        }
    }
}

impl<'s> Iterator for NodeIter<'s> {
    // TODO: should be &'s Node
    type Item = Node;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.len == 0 {
            None
        } else {
            self.node.map(|node| unsafe {
                let t = &*node.as_ptr();
                let result = Node::from_nonnull(self.node);
                self.len -= 1;

                if self.rev {
                    self.node = t.predecessor;
                } else {
                    self.node = t.successor;
                }

                result
            })
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }

    #[inline]
    fn last(self) -> Option<Self::Item>
    where
        Self: Sized,
    {
        todo!()
    }
}

impl<'s> Default for NodeIter<'s> {
    fn default() -> Self {
        Self {
            len: 0,
            rev: false,
            node: None,
            phantom: PhantomData,
        }
    }
}

pub struct ArcIter<'s> {
    rev: bool,
    len: usize,
    node: Option<NonNull<InnerNode>>,
    phantom: PhantomData<&'s ()>,
}

impl<'s> ArcIter<'s> {
    fn new(rev: bool, len: usize, node: Option<NonNull<InnerNode>>) -> Self {
        Self {
            rev,
            len,
            node,
            phantom: PhantomData,
        }
    }
}

impl<'s> Iterator for ArcIter<'s> {
    type Item = Arc;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.len == 0 {
            None
        } else {
            self.node.map(|node| unsafe {
                let next = if self.rev {
                    node.as_ref().predecessor
                } else {
                    node.as_ref().successor
                };

                let arc = Arc {
                    head: Node::from_nonnull(next),
                    tail: Node::from_nonnull(self.node),
                };

                self.node = next;
                self.len -= 1;

                arc
            })
        }
    }
}

impl<'s> Default for ArcIter<'s> {
    fn default() -> Self {
        Self {
            rev: false,
            len: 0,
            node: None,
            phantom: PhantomData,
        }
    }
}

#[derive(Debug)]
pub struct Arc {
    head: Node,
    tail: Node,
}

impl Arc {
    #[inline]
    pub fn head(&self) -> &Node {
        &self.head
    }

    #[inline]
    pub fn tail(&self) -> &Node {
        &self.tail
    }
}

#[derive(Clone, Debug)]
pub(crate) struct InnerRoute {
    n_nodes: usize,
    /// Vehicle's capacity in the current route.
    capacity: f64,
    load: f64,
    first: Option<NonNull<InnerNode>>,
    last: Option<NonNull<InnerNode>>,
    cache: DistanceCache,
    rev: bool,
    has_depot: bool,
}

#[derive(Debug)]
pub struct Tour {
    vehicle_capacity: f64,
    reg: NodeRegistry,
    routes: Vec<Route>,
}

impl Tour {
    /// Create a new instance of `Tour`.
    pub fn new(reg: NodeRegistry, vehicle_capacity: f64) -> Self {
        Self {
            vehicle_capacity,
            reg,
            routes: Vec::with_capacity(0),
        }
    }

    /// Returns the number of nodes.
    #[inline]
    pub fn n_nodes(&self) -> usize {
        self.reg.len()
    }

    #[inline]
    pub fn node(&self, index: usize) -> Option<&Node> {
        self.reg.node(index)
    }

    #[inline]
    pub fn node_mut(&mut self, index: usize) -> Option<&mut Node> {
        self.reg.node_mut(index)
    }

    #[inline]
    pub fn n_routes(&self) -> usize {
        self.routes.len()
    }

    #[inline]
    pub fn route(&self, index: usize) -> Option<&Route> {
        self.routes.get(index)
    }

    #[inline]
    pub fn route_mut(&mut self, index: usize) -> Option<&mut Route> {
        self.routes.get_mut(index)
    }

    #[inline]
    pub fn route_vec(&self) -> Vec<Vec<usize>> {
        self.routes.iter().map(|r| r.index_vec()).collect()
    }

    #[inline]
    pub fn route_vec_sorted(&self) -> Vec<usize> {
        let mut tmp: Vec<_> = self.routes.iter().map(|r| r.index_vec()).collect();
        tmp.sort_by(|a, b| a[1].cmp(&b[1]));
        tmp.into_iter().flatten().collect()
    }

    #[inline]
    pub fn new_route(&mut self) -> &mut Route {
        let route = Route::new(
            self.reg.depot().unwrap(),
            self.vehicle_capacity,
            self.reg.cache(),
        );
        self.routes.push(route);
        self.routes.last_mut().unwrap()
    }

    #[inline]
    pub fn total_distance(&self) -> f64 {
        self.routes
            .iter()
            .fold(0., |acc, x| acc + x.total_distance())
    }

    #[inline]
    pub fn distance<I>(&self, a: &I, b: &I) -> f64
    where
        I: NodeIndex,
    {
        self.reg.cache().distance(a, b)
    }

    /// Initialises the tour by using the Clarke-Wright savings algorithm.
    pub fn init_cw(&mut self) {
        let depot = self.reg.depot().expect("There must be at least one depot.");
        let len = self.reg.len();
        let mut routes = Vec::with_capacity(len - 1);

        let mut bh = BinaryHeap::with_capacity(len * len);
        let cache = self.reg.cache();

        for node1 in self.reg.node_iter() {
            if node1.kind() != NodeKind::Depot {
                let mut r = Route::new(depot, self.vehicle_capacity, self.reg.cache());
                r.push_back(&node1);
                routes.push(r);
            }

            let d_depot1 = cache.distance(node1, depot);
            for node2 in self.reg.node_iter() {
                if node2.kind() != NodeKind::Depot && node1 != node2 {
                    let d_depot2 = cache.distance(node2, depot);
                    let d_12 = cache.distance(node1, node2);
                    let saving = d_depot1 + d_depot2 - d_12;
                    bh.push(SavingPair::new(node1, node2, saving));
                }
            }
        }

        while let Some(pair) = bh.pop() {
            let (node1, node2, _) = pair.into_value();

            if node1.kind() != NodeKind::Depot && node2.kind() != NodeKind::Depot {
                unsafe {
                    let mut route1 = node1.inner.as_ref().unwrap().as_ref().route;
                    let mut route2 = node2.inner.as_ref().unwrap().as_ref().route;

                    if Route::check_capacity_(&route1, &route2) {
                        match (node1.pos(), node2.pos()) {
                            (NodePos::First, NodePos::First) => {
                                Route::rev_(&route2, true);
                                Route::append_front_(&mut route1, &mut route2);
                            }
                            (NodePos::First, NodePos::Last) => {
                                Route::append_front_(&mut route1, &mut route2)
                            }
                            (NodePos::Last, NodePos::First) => {
                                Route::append_back_(&mut route1, &mut route2)
                            }
                            (NodePos::Last, NodePos::Last) => {
                                Route::rev_(&route2, true);
                                Route::append_back_(&mut route1, &mut route2);
                            }
                            _ => {}
                        }
                    }
                }
            }
        }

        routes.retain(|r| !r.is_empty());
        routes.shrink_to_fit();
        self.routes = routes;
    }

    #[inline]
    pub fn drop_empty(&mut self) {
        self.routes.retain(|r| !r.is_empty());
    }

    #[inline]
    pub fn route_iter(&self) -> std::slice::Iter<Route> {
        self.routes.iter()
    }

    #[inline]
    pub fn route_iter_mut(&mut self) -> std::slice::IterMut<Route> {
        self.routes.iter_mut()
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
    use approx::assert_relative_eq;

    use crate::{
        distance::LowerColDist,
        reg::{DistanceCache, NodeRegistry},
        tour::Tour,
    };

    use super::{Node, NodeKind, Route};

    #[test]
    fn test_push_back() {
        let depot = Node::new(0, NodeKind::Depot, 0.);
        let mut route = Route::new(&depot, 1000., &DistanceCache::default());

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
        let mut route = Route::new(&depot, 1000., &DistanceCache::default());

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

    #[test]
    fn test_append_back() {
        let depot = Node::new(0, NodeKind::Depot, 0.);
        let mut route1 = Route::new(&depot, 1000., &DistanceCache::default());
        let nodes: Vec<_> = (1..=10)
            .map(|ii| Node::new(ii, NodeKind::Request, 10.))
            .collect();
        nodes
            .iter()
            .take(5)
            .for_each(|node| route1.push_front(node));

        let mut route2 = Route::new(&depot, 1000., &DistanceCache::default());
        nodes
            .iter()
            .rev()
            .take(5)
            .for_each(|node| route2.push_front(node));

        route1.append_back(&mut route2);
        assert_eq!(&vec![0, 5, 4, 3, 2, 1, 6, 7, 8, 9, 10], &route1.index_vec());
    }

    #[test]
    fn test_append_front() {
        let depot = Node::new(0, NodeKind::Depot, 0.);
        let mut route1 = Route::new(&depot, 1000., &DistanceCache::default());
        let nodes: Vec<_> = (1..=10)
            .map(|ii| Node::new(ii, NodeKind::Request, 10.))
            .collect();
        nodes
            .iter()
            .take(5)
            .for_each(|node| route1.push_front(node));

        let mut route2 = Route::new(&depot, 1000., &DistanceCache::default());
        nodes
            .iter()
            .rev()
            .take(5)
            .for_each(|node| route2.push_front(node));

        route1.append_front(&mut route2);
        assert_eq!(&vec![0, 6, 7, 8, 9, 10, 5, 4, 3, 2, 1], &route1.index_vec());
    }

    #[test]
    fn test_pop_back() {
        let depot = Node::new(0, NodeKind::Depot, 0.);
        let mut route = Route::new(&depot, 1000., &DistanceCache::default());

        let nodes: Vec<_> = (1..=2)
            .map(|ii| Node::new(ii, NodeKind::Request, 10.))
            .collect();
        nodes.iter().for_each(|node| route.push_back(node));

        route.pop_back();
        route.pop_back();

        assert_eq!(0., route.load());
        assert_eq!(1, route.n_nodes());
        assert_eq!(&vec![0], &route.index_vec());

        route.rev(true);
        assert_eq!(&vec![0], &route.index_vec());
        assert!(&route.is_empty());
    }

    #[test]
    fn test_pop_front() {
        let depot = Node::new(0, NodeKind::Depot, 0.);
        let mut route = Route::new(&depot, 1000., &DistanceCache::default());

        let nodes: Vec<_> = (1..=2)
            .map(|ii| Node::new(ii, NodeKind::Request, 10.))
            .collect();
        nodes.iter().for_each(|node| route.push_back(node));

        route.pop_front();
        route.pop_front();

        assert_eq!(0., route.load());
        assert_eq!(1, route.n_nodes());
        assert_eq!(&vec![0], &route.index_vec());

        route.rev(true);
        assert_eq!(&vec![0], &route.index_vec());
        assert!(&route.is_empty());
    }

    #[test]
    fn test_cw() {
        let dist_mtx = vec![
            vec![4., 4., 2.83, 4., 5., 2., 4.24],
            vec![5.66, 6.32, 8., 8.54, 4.47, 3.16],
            vec![2.83, 5.66, 8.06, 6., 7.62],
            vec![2.83, 5.39, 4.47, 7.07],
            vec![3., 4.47, 7.62],
            vec![4.12, 7.],
            vec![3.16],
        ]
        .iter()
        .cloned()
        .flatten()
        .collect::<Vec<f64>>();

        let lcd = LowerColDist::new(8, dist_mtx);

        let mut reg = NodeRegistry::new(8);
        reg.add(vec![0.; 0], NodeKind::Depot, 0.);
        reg.add(vec![0.; 0], NodeKind::Request, 12.);
        reg.add(vec![0.; 0], NodeKind::Request, 12.);
        reg.add(vec![0.; 0], NodeKind::Request, 6.);
        reg.add(vec![0.; 0], NodeKind::Request, 16.);
        reg.add(vec![0.; 0], NodeKind::Request, 15.);
        reg.add(vec![0.; 0], NodeKind::Request, 10.);
        reg.add(vec![0.; 0], NodeKind::Request, 8.);

        reg.compute(&lcd);

        let mut tour = Tour::new(reg, 30.);
        tour.init_cw();

        let routes = tour.route_vec();
        assert_eq!(3, routes.len());

        // let mut tourset = TourSet::new();
        // tourset.insert(vec![0, 1, 7, 6, 0, 2, 5, 0, 3, 4]);
        // assert!(tourset.contains(&tour.route_vec_sorted()));

        assert_relative_eq!(39.04, tour.total_distance());
    }

    #[test]
    fn test_eject() {
        let depot = Node::new(0, NodeKind::Depot, 0.);
        let mut route = Route::new(&depot, 1000., &DistanceCache::default());

        let mut nodes: Vec<_> = (1..=10)
            .map(|ii| Node::new(ii, NodeKind::Request, 10.))
            .collect();
        nodes.iter().take(5).for_each(|node| route.push_back(node));

        Route::eject(nodes.get_mut(2).unwrap());
        Route::eject(nodes.get_mut(4).unwrap());

        route.rev(true);
        nodes
            .iter()
            .rev()
            .take(5)
            .for_each(|node| route.push_back(node));

        Route::eject(nodes.get_mut(5).unwrap());
        Route::eject(nodes.get_mut(9).unwrap());

        assert_eq!(&vec![0, 4, 2, 1, 9, 8, 7], &route.index_vec());
    }
}
