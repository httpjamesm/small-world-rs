use serde::{Deserialize, Serialize};

use crate::distance_metric::DistanceMetric;
use crate::primitives::vector::Vector;
use std::collections::HashSet;

#[derive(Clone, Serialize, Deserialize)]
pub struct Node {
    // id is the unique identifier for the node
    id: u32,
    // value is the value of the node, aka the vector embedding
    value: Vector,
    // connections represents the ids of the nodes that this node is connected to.
    // index = level, value = ids of nodes in that level that this node is connected to
    connections: Vec<HashSet<u32>>,
}

impl Node {
    pub fn new(id: u32, value: Vector, max_level: usize) -> Self {
        Self {
            id,
            value,
            connections: vec![HashSet::new(); max_level + 1],
        }
    }

    // get the connections for a given level
    pub fn connections(&self, level: usize) -> Vec<u32> {
        if level >= self.connections.len() {
            return vec![];
        }
        self.connections[level].clone().into_iter().collect()
    }

    pub fn distance(&self, other: &Vector, distance_metric: &DistanceMetric) -> f32 {
        distance_metric.distance(&self.value, other)
    }

    pub fn id(&self) -> u32 {
        self.id
    }

    pub fn connect(&mut self, other: &mut Node, level: usize) {
        self.connections[level].insert(other.id);
        other.connections[level].insert(self.id);
    }

    pub fn disconnect(&mut self, other: &mut Node, level: usize) {
        self.connections[level].remove(&other.id);
        other.connections[level].remove(&self.id);
    }

    pub fn remove_connections(&mut self, ids: &[u32], level: usize) {
        self.connections[level].retain(|id| !ids.contains(id));
    }

    pub fn value(&self) -> &Vector {
        &self.value
    }
}
