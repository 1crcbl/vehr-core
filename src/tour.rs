//! Tour module
use std::{
    collections::{BinaryHeap, HashSet},
    marker::PhantomData,
    ptr::NonNull,
};

use crate::{
    reg::{DistanceCache, NodeRegistry},
    traits::NodeIndex,
};

macro_rules! panic_ptr {
    ($name:expr) => {
        panic!("{} is either uninitialised or already dropped.", $name);
    };
}

/// A struct representing a location in a routing problem instance.
#[derive(Clone, Debug)]
pub struct Node {
    pub(crate) inner: Option<NonNull<InnerNode>>,
}

impl Node {
    pub fn new(index: usize, kind: NodeKind, demand: f64) -> Self {
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

/// An enum representing the type for each node.
///
/// In a routing problem instance, every node must be either a [`NodeKind::Depot`] or a [`NodeKind::Request`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NodeKind {
    /// A location from which vehicles originate.
    ///
    /// A depot node can be assigned to multiple routes.
    Depot,
    /// A location where vehicles must arrive and provide requested services.
    ///
    /// A request node can only be assigned to one route.
    Request,
}

///
#[derive(Clone, Copy, Debug, PartialEq, Hash)]
enum NodePos {
    First,
    Interior,
    Last,
    None,
}

/// A struct representing a route (sub-tour) in a routing problem.
///
/// Each route contains a subset of locations in a routing problem and has maximally one node of type
/// [`NodeKind::Depot`] while the other nodes in that route must be of type [`NodeKind::Request`].
/// The intersection set of two routes in a tour is either empty or contains exactly one element: a
/// depot.
///
/// It is assumed that for every route, a vehicle must start from the route's depot, visit all request
/// nodes assigned to the route and return back to the depot.
#[derive(Debug)]
pub struct Route {
    inner: Option<NonNull<InnerRoute>>,
}

impl Route {
    // Similar to new, but without depot.
    // This function should only be used from Tour::with_setup().
    fn new_(vehicle_capacity: f64, cache: &DistanceCache) -> Self {
        let inner = Box::new(InnerRoute {
            n_nodes: 0,
            capacity: vehicle_capacity,
            load: 0.,
            first: None,
            last: None,
            cache: cache.clone(),
            rev: false,
            has_depot: false,
        });

        Self {
            inner: NonNull::new(Box::leak(inner)),
        }
    }

    pub fn new<N>(depot: N, vehicle_capacity: f64, cache: &DistanceCache) -> Self
    where
        N: AsRef<Node>,
    {
        let route = Self::new_(vehicle_capacity, cache);

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
            (*route.inner.unwrap().as_ptr()).has_depot = true;
            Self::push_(&route.inner, &new_depot, false);
        }

        route
    }

    /// Returns the capacity of a vehicle assigned to this route.
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

    /// Reverses the `Route`'s traversal direction.
    ///
    /// # Examples
    /// ```
    /// # use vehr_core::reg::DistanceCache;
    /// # use vehr_core::tour::Node;
    /// # use vehr_core::tour::NodeKind;
    /// # use vehr_core::tour::Route;
    /// let mut route = Route::new(&Node::new(0, NodeKind::Depot, 0.), 10., &DistanceCache::default());
    /// (1..=3).for_each(|ii| route.push_back(&Node::new(ii, NodeKind::Request, 1.)));
    ///
    /// assert_eq!(&vec![0, 1, 2, 3], &route.index_vec());
    /// route.rev(true);
    /// assert_eq!(&vec![0, 3, 2, 1], &route.index_vec());
    /// ```
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

    /// Checks whether the route is empty.
    ///
    /// A route is empty when it contains no request nodes. This means that a route containing only
    /// a depot node is also considered empty.
    ///
    /// # Examples
    /// ```
    /// # use vehr_core::reg::DistanceCache;
    /// # use vehr_core::tour::Node;
    /// # use vehr_core::tour::NodeKind;
    /// # use vehr_core::tour::Route;
    /// #
    /// let mut route = Route::new(&Node::new(0, NodeKind::Depot, 0.), 10., &DistanceCache::default());
    /// assert!(route.is_empty());
    ///
    /// route.push_back(&Node::new(1, NodeKind::Request, 10.));
    /// assert!(!route.is_empty());
    /// ```
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

    /// Moves all nodes from `other` into the front of `Self`, taking into account the traversal direction,
    /// leaving `other` empty.
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
    /// Moves all nodes from `other` into the back of `Self`, taking into account the traversal direction,
    /// leaving `other` empty.
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

    /// Inserts a node at the end of the `Route`, according to its traversal direction.
    ///
    /// # Examples
    /// ```
    /// # use vehr_core::reg::DistanceCache;
    /// # use vehr_core::tour::Node;
    /// # use vehr_core::tour::NodeKind;
    /// # use vehr_core::tour::Route;
    /// #
    /// let depot = Node::new(0, NodeKind::Depot, 0.);
    /// let mut nodes: Vec<_> = (1..=6).map(|ii| Node::new(ii, NodeKind::Request, 10.)).collect();
    /// nodes.insert(0, depot);
    /// let mut route = Route::new(&nodes[0], 1000., &DistanceCache::default());
    ///
    /// // Adds 1, 2, 3
    /// route.push_back(nodes.get(1).unwrap());
    /// assert_eq!(&vec![0, 1], &route.index_vec());
    /// route.push_back(nodes.get(2).unwrap());
    /// assert_eq!(&vec![0, 1, 2], &route.index_vec());
    /// route.push_back(nodes.get(3).unwrap());
    /// assert_eq!(&vec![0, 1, 2, 3], &route.index_vec());
    ///
    /// route.rev(true);
    ///
    /// // Adds 4, 5, 6
    /// route.push_back(nodes.get(4).unwrap());
    /// assert_eq!(&vec![0, 3, 2, 1, 4], &route.index_vec());
    /// route.push_back(nodes.get(5).unwrap());
    /// assert_eq!(&vec![0, 3, 2, 1, 4, 5], &route.index_vec());
    /// route.push_back(nodes.get(6).unwrap());
    /// assert_eq!(&vec![0, 3, 2, 1, 4, 5, 6], &route.index_vec());
    /// ```
    #[inline]
    pub fn push_back(&mut self, node: &Node) {
        unsafe {
            Self::push_(&self.inner, &node.inner, false);
        }
    }

    /// Inserts a node at the beginning of the `Route`, according to its traversal direction.
    ///
    /// # Examples
    /// ```
    /// # use vehr_core::reg::DistanceCache;
    /// # use vehr_core::tour::Node;
    /// # use vehr_core::tour::NodeKind;
    /// # use vehr_core::tour::Route;
    /// let depot = Node::new(0, NodeKind::Depot, 0.);
    /// let mut nodes: Vec<_> = (1..=6).map(|ii| Node::new(ii, NodeKind::Request, 10.)).collect();
    /// nodes.insert(0, depot);
    /// let mut route = Route::new(&nodes[0], 1000., &DistanceCache::default());
    ///
    /// // Adds 1, 2, 3
    /// route.push_front(nodes.get(1).unwrap());
    /// assert_eq!(&vec![0, 1], &route.index_vec());
    /// route.push_front(nodes.get(2).unwrap());
    /// assert_eq!(&vec![0, 2, 1], &route.index_vec());
    /// route.push_front(nodes.get(3).unwrap());
    /// assert_eq!(&vec![0, 3, 2, 1], &route.index_vec());
    ///
    /// route.rev(true);
    ///
    /// // Adds 4, 5, 6
    /// route.push_front(nodes.get(4).unwrap());
    /// assert_eq!(&vec![0, 4, 1, 2, 3], &route.index_vec());
    /// route.push_front(nodes.get(5).unwrap());
    /// assert_eq!(&vec![0, 5, 4, 1, 2, 3], &route.index_vec());
    /// route.push_front(nodes.get(6).unwrap());
    /// assert_eq!(&vec![0, 6, 5, 4, 1, 2, 3], &route.index_vec());
    /// ```
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

    /// Removes the last [`NodeKind::Request`] node in the `Route` and returns it, or [`None`] if it is empty.
    ///
    /// The function takes into account the `Route`'s traversal direction.
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

    /// Removes the first [`NodeKind::Request`] node in the `Route` and returns it, or [`None`] if it is empty.
    ///
    /// The function takes into account the `Route`'s traversal direction.
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

    /// Removes the first [`NodeKind::Request`] node in the `Route` and returns it, or [`None`] if it is empty.
    ///
    /// The function ignores the `Route`'s traversal direction.
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

    /// Removes the last [`NodeKind::Request`] node in the `Route` and returns it, or [`None`] if it is empty.
    ///
    /// The function ignores the `Route`'s traversal direction.
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

    /// Inserts a node at the back of the `pivot` node.
    ///
    /// Nothing will change if the `pivot` node is not assigned to any route.
    #[inline]
    pub fn insert_back(pivot: &mut Node, node: &mut Node) {
        unsafe {
            Self::insert_(pivot, node, false);
        }
    }

    /// Inserts a node at the front of the `pivot` node.
    ///
    /// Nothing will change if the `pivot` node is not assigned to any route.
    #[inline]
    pub fn insert_front(pivot: &mut Node, node: &mut Node) {
        unsafe {
            Self::insert_(pivot, node, true);
        }
    }

    unsafe fn insert_(pivot: &mut Node, node: &mut Node, front: bool) {
        // insert_front -> front = true
        // insert_back  -> front = false
        if let (Some(innerp), Some(innern)) = (pivot.inner, node.inner) {
            if let Some(route) = innerp.as_ref().route {
                (*route.as_ptr()).n_nodes += 1;
                (*route.as_ptr()).load += innern.as_ref().demand;
                (*innern.as_ptr()).route = innerp.as_ref().route;

                if route.as_ref().rev ^ front {
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
                    (*innern.as_ptr()).predecessor = pivot.inner;
                    (*innerp.as_ptr()).successor = node.inner;

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

    /// Removes a node from its current `Route`. The function will do nothing if it is not assigned
    /// to any `Route`.
    ///
    /// # Examples
    /// ```
    /// # use vehr_core::reg::DistanceCache;
    /// # use vehr_core::tour::Node;
    /// # use vehr_core::tour::NodeKind;
    /// # use vehr_core::tour::Route;
    /// let mut route = Route::new(&Node::new(0, NodeKind::Depot, 0.), 10., &DistanceCache::default());
    /// let mut nodes: Vec<_> = (1..=4).map(|ii| {
    ///         let node = Node::new(ii, NodeKind::Request, 10.);
    ///         route.push_back(&node);
    ///         node
    /// }).collect();
    ///
    /// Route::eject(nodes.get_mut(2).unwrap());
    /// assert_eq!(&vec![0, 1, 2 , 4], &route.index_vec());
    /// ```
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

    /// Returns the total distance of a vehicle travelling in the `Route`.
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

    /// Returns a front-to-back node iterator.
    ///
    /// The iterator will traverse all nodes according the `Route`'s traversal direction. The node iterator
    /// will always start with the `Route`'s depot and end at the final [`NodeKind::Request`] node.
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

    /// Returns a front-to-back arc iterator.
    ///
    /// The iterator will traverse all nodes according the `Route`'s traversal direction.
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

    pub fn with_setup(&mut self, setup: &TourSetup) {
        if !self.reg.depots.is_superset(&setup.depots) {
            panic!("Unknown depots in TourSetup")
        }

        let mut routes = Vec::with_capacity(setup.routes.len());
        for vr in &setup.routes {
            let depot = self.reg.node(*vr.first().unwrap()).unwrap();
            let mut route = Route::new(depot, self.vehicle_capacity, self.reg.cache());
            vr.iter().skip(1).for_each(|idx| {
                let node = self.reg.node(*idx).unwrap();
                route.push_back(node);
            });
            routes.push(route);
        }

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

pub struct TourSetup {
    depots: HashSet<usize>,
    routes: Vec<Vec<usize>>,
    dist: Option<f64>,
}

impl TourSetup {
    pub fn new() -> Self {
        Self {
            ..Default::default()
        }
    }

    pub fn with_depots(depots: HashSet<usize>) -> Self {
        Self {
            depots,
            ..Default::default()
        }
    }

    pub fn with_routes(depots: HashSet<usize>, mut routes: Vec<Vec<usize>>) -> Self {
        for rt in routes.iter_mut().filter(|v| v.len() > 1) {
            let idx = rt.iter().enumerate().find_map(|(node, ii)| {
                if depots.contains(&node) {
                    Some(*ii)
                } else {
                    None
                }
            });

            if let Some(idx) = idx {
                rt.rotate_left(idx);
            }
        }

        Self {
            depots,
            routes,
            dist: None,
        }
    }

    pub fn add_depot(&mut self, depot: usize) {
        self.depots.insert(depot);
    }

    pub fn add_route(&mut self, mut route: Vec<usize>) {
        self.adjust_route(&mut route);
        self.routes.push(route);
    }

    fn adjust_route(&self, route: &mut Vec<usize>) {
        let idx = route.iter().enumerate().find_map(|(ii, node)| {
            if self.depots.contains(node) {
                Some(ii)
            } else {
                None
            }
        });

        if let Some(idx) = idx {
            route.rotate_left(idx);
        }
    }

    pub fn routes(&self) -> &[Vec<usize>] {
        &self.routes
    }

    pub fn dist(&self) -> Option<f64> {
        self.dist
    }
}

impl Default for TourSetup {
    fn default() -> Self {
        Self {
            depots: HashSet::new(),
            routes: Vec::new(),
            dist: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use core::f64;

    use approx::assert_relative_eq;

    use crate::{
        distance::LowerColDist,
        reg::{DistanceCache, NodeRegistry},
        tour::Tour,
    };

    use super::{Node, NodeKind, Route, TourSetup};

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
        let mut tour = Tour::new(make_reg(), 30.);
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

    #[test]
    fn test_insert_back() {
        let depot = Node::new(0, NodeKind::Depot, 0.);
        let mut route = Route::new(&depot, 100., &DistanceCache::default());

        let mut nodes: Vec<_> = (1..=3)
            .map(|ii| Node::new(ii, NodeKind::Request, 1.))
            .collect();
        nodes.iter().for_each(|node| route.push_back(node));

        assert_eq!(&vec![0, 1, 2, 3], &route.index_vec());
        Route::insert_back(
            nodes.get_mut(1).unwrap(),
            &mut Node::new(4, NodeKind::Request, 1.),
        );
        assert_eq!(&vec![0, 1, 2, 4, 3], &route.index_vec());
        route.rev(true);
        Route::insert_back(
            nodes.get_mut(1).unwrap(),
            &mut Node::new(5, NodeKind::Request, 1.),
        );
        assert_eq!(&vec![0, 3, 4, 2, 5, 1], &route.index_vec());
    }

    #[test]
    fn test_insert_front() {
        let depot = Node::new(0, NodeKind::Depot, 0.);
        let mut route = Route::new(&depot, 100., &DistanceCache::default());

        let mut nodes: Vec<_> = (1..=3)
            .map(|ii| Node::new(ii, NodeKind::Request, 1.))
            .collect();
        nodes.iter().for_each(|node| route.push_back(node));

        assert_eq!(&vec![0, 1, 2, 3], &route.index_vec());
        Route::insert_front(
            nodes.get_mut(1).unwrap(),
            &mut Node::new(4, NodeKind::Request, 1.),
        );
        assert_eq!(&vec![0, 1, 4, 2, 3], &route.index_vec());
        route.rev(true);
        Route::insert_front(
            nodes.get_mut(1).unwrap(),
            &mut Node::new(5, NodeKind::Request, 1.),
        );
        assert_eq!(&vec![0, 3, 5, 2, 4, 1], &route.index_vec());
    }

    #[test]
    fn test_setup() {
        let mut tour = Tour::new(make_reg(), 100.);

        let mut setup = TourSetup::new();
        setup.add_depot(0);
        setup.add_route((0..8).collect());
        tour.with_setup(&setup);

        assert_eq!(1, tour.n_routes());
        let routes = tour.route_vec();
        assert_eq!(1, routes.len());
        assert_eq!(&(0..8).collect::<Vec<_>>(), routes.first().unwrap());

        let mut setup = TourSetup::new();
        setup.add_depot(0);
        setup.add_route(vec![3, 2, 0, 4, 1]);
        setup.add_route(vec![6, 0]);
        setup.add_route(vec![0, 7, 5]);
        tour.with_setup(&setup);

        assert_eq!(3, tour.n_routes());
        let routes = tour.route_vec();
        assert_eq!(3, routes.len());
        let mut count = 0;
        for rt in &routes {
            for rts in setup.routes() {
                if rt == rts {
                    count += 1;
                    break;
                }
            }
        }
        assert_eq!(3, count);
    }

    fn make_reg() -> NodeRegistry {
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
        reg
    }
}
