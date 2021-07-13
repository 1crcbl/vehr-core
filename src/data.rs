use std::ptr::NonNull;

// TODO: impl Drop for all Inner structs.

macro_rules! panic_ptr {
    ($name:expr) => {
        panic!("{} is either uninitialised or already dropped.", $name);
    };
}

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
        self.data
    }

    #[inline]
    pub fn metadata(&self) -> &M {
        &self.meta
    }
}

#[derive(Clone, Copy, Debug)]
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
        Self {
            inner: None
        }
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

    pub fn check_capacity(&self, demand: f64) -> bool {
        match self.inner {
            Some(inner) => unsafe {
                let tmp_load = inner.as_ref().load + demand;
                tmp_load <= inner.as_ref().capacity
            },
            None => panic_ptr!("Route")
        }
    }

    pub fn rev(&mut self, rev: bool) {
        match self.inner {
            Some(inner) => unsafe { (*inner.as_ptr()).rev = rev },
            None => panic_ptr!("Route"),
        }
    }

    pub fn is_empty(&self) -> bool {
        match self.inner {
            Some(inner) => unsafe {
                let inner = inner.as_ref();
                inner.first.is_none() && inner.last.is_none()
            },
            None => panic_ptr!("Route"),
        }
    }
}

impl Default for Route {
    fn default() -> Self {
        Self {
            inner: None,
        }
    }
}

#[derive(Clone, Debug)]
struct InnerRoute {
    /// Vehicle's capacity.
    capacity: f64,
    load: f64,
    depot: Option<NonNull<InnerNode>>,
    first: Option<NonNull<InnerNode>>,
    last: Option<NonNull<InnerNode>>,
    rev: bool,
}
