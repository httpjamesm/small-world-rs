use std::collections::{BinaryHeap, HashMap, HashSet};

use super::node::Node;
use crate::{distance_metric::DistanceMetric, primitives::vector::Vector};
use anyhow::{bail, Result};
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};

/// World is the main struct that represents the full HNSW graph world
#[derive(Clone, Serialize, Deserialize)]
pub struct World {
    /// nodes is a list of all the nodes in the world by id
    nodes: HashMap<u32, Node>,
    /// level_entrypoints is a list of the ids of the entrypoint nodes for each level
    /// index = level, value = id of the entrypoint node
    level_entrypoints: Vec<u32>,
    /// m is the maximum number of connections for a node
    m: usize,
    /// ef_construction is the maximum number of connections to explore for a node during construction
    ef_construction: usize,
    /// ef_search is the maximum number of connections to explore for a node during search
    ef_search: usize,
    // max_level is the maximum level of the HNSW graph
    max_level: usize,
    /// distance_metric is the distance metric used to calculate distances between vectors
    distance_metric: DistanceMetric,
}

impl World {
    pub fn new(
        m: usize,
        ef_construction: usize,
        ef_search: usize,
        distance_metric: DistanceMetric,
    ) -> Result<Self> {
        // ef_construction must be >= M
        if ef_construction < m {
            bail!("ef_construction must be >= M");
        }

        Ok(Self {
            nodes: HashMap::new(),
            level_entrypoints: vec![],
            m,
            ef_construction,
            ef_search,
            max_level: 1,
            distance_metric,
        })
    }

    /// new_from_dump creates a new world from a serialized dump
    pub fn new_from_dump(data: &[u8]) -> Result<Self> {
        bincode::deserialize(data)
            .map_err(|e| anyhow::anyhow!("Failed to deserialize world: {}", e))
    }

    /// pick_node_level picks the level at which a new node should be inserted based on the probabalistic insertion strategy.
    pub(crate) fn pick_node_level(&self) -> usize {
        let p = 1.0 / (self.m as f32);
        let mut level = 0;
        while fastrand::f32() < (1.0 - p) && level < self.max_level {
            level += 1;
        }
        level
    }

    /// get_entrypoint_node gets the entrypoint node for the HNSW graph.
    fn get_entrypoint_node(&self) -> Node {
        self.get_entrypoint_node_per_level(self.max_level)
    }

    /// get_entrypoint_node_per_level gets the entrypoint node for a given level
    fn get_entrypoint_node_per_level(&self, level: usize) -> Node {
        if self.level_entrypoints.is_empty() {
            return self.nodes.values().next().unwrap().clone();
        }
        let id = self.level_entrypoints[level];
        self.nodes.get(&id).unwrap().clone()
    }

    /// greedy_search performs a greedy search for the k nearest neighbours to the query vector
    fn greedy_search(&self, query: &Vector, entry_node: &Node, level: usize) -> Vec<u32> {
        let mut visited = HashSet::new();
        let mut candidates: BinaryHeap<(OrderedFloat<f32>, u32)> = BinaryHeap::new();
        let mut best_candidates: BinaryHeap<(OrderedFloat<f32>, u32)> = BinaryHeap::new();

        // get the distance between the new node and the entry node
        let distance = entry_node.distance(query, &self.distance_metric);
        // add the entry node to the candidates
        // we're using negatives here because BinaryHeap is a max heap by default and we want min heap behaviour to find the nearest neighbours, not the furthest
        candidates.push((-OrderedFloat(distance), entry_node.id()));
        // add to the visited set
        visited.insert(entry_node.id());
        best_candidates.push((OrderedFloat(distance), entry_node.id()));

        // let's go through the graph to find the best candidates
        while !candidates.is_empty() {
            let (current_dist, current_id) = candidates.pop().unwrap();

            // if the current distance is greater than the best candidate, skip it because it's a bad candidate
            if !best_candidates.is_empty() && -current_dist > best_candidates.peek().unwrap().0 {
                continue;
            }

            // at this level, we need to check the neighbours
            for neighbour_id in self.nodes.get(&current_id).unwrap().connections(level) {
                // if we've already visited this node, skip it
                if visited.contains(&neighbour_id) {
                    continue;
                }

                // visit the node
                visited.insert(neighbour_id);

                // get the distance between the new node and the neighbour
                let distance = self
                    .nodes
                    .get(&neighbour_id)
                    .unwrap()
                    .distance(query, &self.distance_metric);

                // if this candidate is better than the best candidate
                let ef_size = if level == 0 {
                    self.ef_search
                } else {
                    self.ef_construction
                };
                if best_candidates.len() < ef_size
                    || OrderedFloat(distance) < best_candidates.peek().unwrap().0
                {
                    // The new candidate is strictly better (smaller distance) than the worst one we have so far.
                    candidates.push((OrderedFloat(distance), neighbour_id));
                    best_candidates.push((OrderedFloat(distance), neighbour_id));

                    // Enforce ef_construction by popping the largest distance from best_candidates if needed (max heap, so the root node is the furthest and therefore the worst)
                    if best_candidates.len() > ef_size {
                        best_candidates.pop();
                    }
                }
            }
        }

        // return our best candidates
        best_candidates.into_iter().map(|(_, id)| id).collect()
    }

    /// insert_node inserts a new node into the world.
    /// id must be fully unique in the World
    // 1. pick the level at which to insert the node
    // 2. find the M nearest neighbors for the node at the chosen level
    // 3. connect the new node to the neighbors and on all lower levels
    // 4. recursively connect the new node to the neighbors' neighbors
    // 5. if the new node has no connections, add it to the graph at level 0
    pub fn insert_vector(&mut self, id: u32, vector: Vector) -> Result<()> {
        // If this is the first node, initialize it as the entrypoint for all levels
        if self.nodes.is_empty() {
            let initial_level = self.pick_node_level();
            let node = Node::new(id, vector, initial_level);
            self.nodes.insert(node.id(), node.clone());
            self.level_entrypoints = vec![id; initial_level + 1];
            self.max_level = initial_level;
            return Ok(());
        }

        // ensure the id is completely unique
        if self.nodes.contains_key(&id) {
            bail!("Node id must be unique");
        }

        let new_max_level = calculate_max_level(self.nodes.len() + 1, self.m);
        if new_max_level > self.max_level {
            self.max_level = new_max_level;
            self.level_entrypoints.resize(new_max_level + 1, id);
        }

        let level = self.pick_node_level();
        let mut node = Node::new(id, vector, level);

        // add the new node to the world
        self.nodes.insert(node.id(), node.clone());

        // Start from the top-level entry point
        let mut current_node_id = self.level_entrypoints[self.max_level];

        for lvl in (level + 1..=self.max_level).rev() {
            let current_node = self.nodes.get(&current_node_id).unwrap();
            let candidates = self.greedy_search(node.value(), current_node, lvl);

            // Pick the closest candidate as the new entry point for the next level down
            if let Some(closest_id) = candidates.iter().min_by(|&id_a, &id_b| {
                let dist_a = self
                    .nodes
                    .get(id_a)
                    .unwrap()
                    .distance(&node.value(), &self.distance_metric);
                let dist_b = self
                    .nodes
                    .get(id_b)
                    .unwrap()
                    .distance(&node.value(), &self.distance_metric);
                dist_a.partial_cmp(&dist_b).unwrap()
            }) {
                current_node_id = *closest_id;
            }
        }

        // Now we are at the correct insertion level (node_level), perform a local search here
        let insertion_node = self.nodes.get(&current_node_id).unwrap();
        let nearest_neighbours = self.greedy_search(node.value(), insertion_node, level);

        // Connect new node with found neighbors and prune
        for &nbr_id in &nearest_neighbours {
            // Connect `node` and neighbor
            node.connect(self.nodes.get_mut(&nbr_id).unwrap(), level);
        }

        // prune the node if it has more than M connections
        if node.connections(level).len() > self.m {
            self.prune_node_connections(node.id(), level);
        }

        println!("Inserted node {} at level {}", id, level);
        println!("My neighbours are {:?}", node.connections(level));

        Ok(())
    }

    /// prune_node_connections prunes the connections of a node at a given level by getting rid of the furthest connections
    fn prune_node_connections(&mut self, node_id: u32, level: usize) {
        let mut distances: Vec<(u32, f32)> = {
            let node = self.nodes.get(&node_id).unwrap();
            node.connections(level)
                .iter()
                .map(|&neighbour_id| {
                    let distance = node.distance(
                        self.nodes.get(&neighbour_id).unwrap().value(),
                        &self.distance_metric,
                    );
                    (neighbour_id, distance)
                })
                .collect()
        };

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let connections_to_remove = distances
            .iter()
            .skip(self.m)
            .map(|(id, _)| *id)
            .collect::<Vec<_>>();

        // Remove connections from the current node
        let node = self.nodes.get_mut(&node_id).unwrap();
        node.remove_connections(&connections_to_remove, level);

        // Remove the corresponding back-connections from neighbor nodes
        for &neighbor_id in &connections_to_remove {
            if let Some(neighbor) = self.nodes.get_mut(&neighbor_id) {
                neighbor.remove_connections(&vec![node_id], level);
            }
        }
    }

    /// search gets the k nearest neighbours to the query vector using beam search
    pub fn search(&self, query: &Vector, k: usize, beam_width: usize) -> Result<Vec<u32>> {
        if k > self.ef_search {
            bail!(
                "k is greater than the maximum number of connections to explore for a node during search"
            );
        }

        let candidates = self.beam_search(query, beam_width);

        let mut results: Vec<(u32, f32)> = candidates
            .into_iter()
            .map(|id| {
                let node = self.nodes.get(&id).unwrap();
                let distance = node.distance(query, &self.distance_metric);
                (id, distance)
            })
            .collect();

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Return top k results
        Ok(results.into_iter().take(k).map(|(id, _)| id).collect())
    }

    /// beam_search performs a beam search for the k nearest neighbours to the query vector
    fn beam_search(&self, query: &Vector, beam_width: usize) -> Vec<u32> {
        let mut candidates: BinaryHeap<(OrderedFloat<f32>, u32)> = BinaryHeap::new();
        let entrypoint_node = self.get_entrypoint_node();
        let initial_distance = entrypoint_node.distance(query, &self.distance_metric);
        candidates.push((OrderedFloat(initial_distance), entrypoint_node.id()));

        let mut visited = HashSet::new();
        let mut final_candidates = Vec::new();

        for level in (0..=self.max_level).rev() {
            let mut next_candidates = Vec::new();

            // Store current candidates before processing
            let current_candidates: Vec<_> = candidates.drain().collect();

            for (dist, candidate_id) in current_candidates {
                if visited.contains(&candidate_id) {
                    continue;
                }
                visited.insert(candidate_id);

                let candidate = self.nodes.get(&candidate_id).unwrap();
                let local_best = self.greedy_search(&query, candidate, level);

                // Add the current candidate to final candidates
                final_candidates.push((dist, candidate_id));

                for &id in &local_best {
                    if !visited.contains(&id) {
                        let dist = self
                            .nodes
                            .get(&id)
                            .unwrap()
                            .distance(query, &self.distance_metric);
                        next_candidates.push((OrderedFloat(dist), id));
                    }
                }
            }

            // Combine current and next candidates
            candidates = next_candidates.into_iter().collect();
        }

        // Return the best candidates we've found
        final_candidates.sort_by_key(|(dist, _)| *dist);
        final_candidates
            .into_iter()
            .take(beam_width)
            .map(|(_, id)| id)
            .collect()
    }

    /// dump serializes the world to binary data so the user can save it for later use without abstraction
    pub fn dump(&self) -> Result<Vec<u8>> {
        bincode::serialize(&self).map_err(|e| anyhow::anyhow!("Failed to serialize world: {}", e))
    }
}

/// calculate_max_level calculates the maximum level of the HNSW graph based on the number of nodes and the maximum number of connections per node
fn calculate_max_level(n: usize, m: usize) -> usize {
    // p = 1/m
    // max_level ≈ log(n)/log(m)
    (((n as f64).ln() / (m as f64).ln()).ceil() as usize).max(1)
}

#[cfg(test)]
mod tests {
    use crate::distance_metric::CosineDistance;

    use super::*;

    #[test]
    fn test_world_insert_and_search() -> Result<()> {
        let mut world = World::new(5, 10, 10, DistanceMetric::Cosine(CosineDistance))?;

        let test_vectors = vec![
            (1, Vector::new_f32(&[1.0, 0.0, 0.0])),
            (2, Vector::new_f32(&[0.0, 1.0, 0.0])),
            (3, Vector::new_f32(&[0.0, 0.0, 1.0])),
            (4, Vector::new_f32(&[0.7, 0.7, 0.0])),
        ];

        for (id, vector) in test_vectors {
            world.insert_vector(id, vector)?;
        }

        let query = Vector::new_f32(&[0.8, 0.8, 0.0]);
        let results = world.search(&query, 2, 5)?;

        assert!(results.len() >= 1, "Should find at least 1 result");
        Ok(())
    }

    #[test]
    fn test_world_dump_and_load() -> Result<()> {
        // make world, dump it, hash it, load it, hash it, assert equal
        let world = World::new(5, 10, 10, DistanceMetric::Cosine(CosineDistance))?;
        let dump = world.dump()?;
        let original_hash = blake3::hash(&dump);
        let loaded_world = World::new_from_dump(&dump)?;
        let loaded_hash = blake3::hash(&loaded_world.dump()?);
        assert_eq!(original_hash, loaded_hash);
        Ok(())
    }
}
