use crate::metrics::calculate_cosine_similarity;

#[derive(Clone)]
pub struct Node {
    // id is the unique identifier for the node
    id: u32,
    // value is the value of the node, aka the vector embedding
    value: Vec<f32>,
    // connections represents the ids of the nodes that this node is connected to.
    // index = level, value = ids of nodes in that level that this node is connected to
    connections: Vec<Vec<u32>>,
    max_level: usize,
}

impl Node {
    pub fn new(id: u32, value: Vec<f32>, max_level: usize) -> Self {
        Self {
            id,
            value,
            connections: vec![Vec::new(); max_level + 1],
            max_level,
        }
    }

    // get the connections for a given level
    pub fn connections(&self, level: usize) -> Vec<u32> {
        if level >= self.connections.len() {
            return vec![];
        }
        self.connections[level].clone()
    }

    pub fn distance(&self, other: &Node) -> f32 {
        calculate_cosine_similarity(&self.value, &other.value)
    }

    pub fn id(&self) -> u32 {
        self.id
    }

    pub fn connect(&mut self, other: &mut Node, level: usize) {
        self.connections[level].push(other.id);
        other.connections[level].push(self.id);
    }

    pub fn disconnect(&mut self, other: &mut Node, level: usize) {
        self.connections[level].retain(|&id| id != other.id);
        other.connections[level].retain(|&id| id != self.id);
    }

    pub fn remove_connections(&mut self, ids: &[u32], level: usize) {
        self.connections[level].retain(|id| !ids.contains(id));
    }
    pub fn value(&self) -> &Vec<f32> {
        &self.value
    }
}
