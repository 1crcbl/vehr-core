use crate::tour::Node;

pub trait NodeIndex {
    fn index(&self) -> usize;
}

impl NodeIndex for usize {
    fn index(&self) -> usize {
        *self
    }
}

impl NodeIndex for Node {
    fn index(&self) -> usize {
        self.get_index()
    }
}
