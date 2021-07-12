use std::ptr::NonNull;

#[derive(Clone, Debug)]
pub struct Node<M> {
    inner: NonNull<InnerNode<M>>,
}

impl<M> Node<M> {
    pub(crate) fn new(index: usize, kind: NodeKind, demand: f64, meta: M) -> Self {
        let inner = Box::new(InnerNode::<M> {
            index,
            demand,
            kind,
            meta,
        });

        Self {
            inner: NonNull::new(Box::leak(inner)).unwrap(),
        }
    }

    #[inline]
    pub(crate) fn get_index(&self) -> usize {
        unsafe { self.inner.as_ref().index }
    }

    #[inline]
    pub fn demand(&self) -> f64 {
        unsafe { self.inner.as_ref().demand }
    }

    #[inline]
    pub fn kind(&self) -> NodeKind {
        unsafe { self.inner.as_ref().kind }
    }

    #[inline]
    pub fn metadata(&self) -> &M {
        unsafe { &self.inner.as_ref().meta }
    }
}

struct InnerNode<M> {
    index: usize,
    demand: f64,
    kind: NodeKind,
    meta: M,
}

#[derive(Clone, Copy, Debug, PartialEq, Hash)]
pub enum NodeKind {
    Depot,
    Request,
}
