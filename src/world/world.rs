use std::collections::{BinaryHeap, HashMap, HashSet};

use crate::metrics::calculate_cosine_similarity;

use super::node::Node;
use anyhow::{bail, Result};
use ordered_float::OrderedFloat;

// World is the main struct that represents the full HNSW graph world
pub struct World {
    // nodes is a list of all the nodes in the world by id
    nodes: HashMap<u32, Node>,
    // level_entrypoints is a list of the ids of the entrypoint nodes for each level
    // index = level, value = id of the entrypoint node
    level_entrypoints: Vec<u32>,
    // m is the maximum number of connections for a node
    m: usize,
    // ef_construction is the maximum number of connections to explore for a node during construction
    ef_construction: usize,
    // ef_search is the maximum number of connections to explore for a node during search
    ef_search: usize,
    // max_level is the maximum level of the HNSW graph
    max_level: usize,
}

impl World {
    pub fn new(
        m: usize,
        ef_construction: usize,
        ef_search: usize,
        max_level: usize,
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
            max_level,
        })
    }

    // pick_node_level picks the level at which a new node should be inserted based on the probabalistic insertion strategy.
    pub(crate) fn pick_node_level(&self) -> usize {
        let p = 1.0 / (self.m as f32);
        let mut level = 0;
        while fastrand::f32() < (1.0 - p) && level < self.max_level - 1 {
            level += 1;
        }
        level
    }

    // get_entrypoint_node gets the entrypoint node for the HNSW graph.
    fn get_entrypoint_node(&self) -> Node {
        self.get_entrypoint_node_per_level(self.max_level)
    }

    fn get_entrypoint_node_per_level(&self, level: usize) -> Node {
        if self.level_entrypoints.is_empty() {
            return self.nodes.values().next().unwrap().clone();
        }
        let id = self.level_entrypoints[level];
        self.nodes.get(&id).unwrap().clone()
    }

    fn greedy_search(&self, query: &Vec<f32>, entry_node: &Node, level: usize) -> Vec<u32> {
        let mut visited = HashSet::new();
        let mut candidates: BinaryHeap<(OrderedFloat<f32>, u32)> = BinaryHeap::new();
        let mut best_candidates: BinaryHeap<(OrderedFloat<f32>, u32)> = BinaryHeap::new();

        // get the distance between the new node and the entry node
        let distance = calculate_cosine_similarity(&query, entry_node.value());
        // add the entry node to the candidates
        // we're using negatives here because BinaryHeap is a max heap by default and we want min heap behaviour to find the nearest neighbours, not the furthest
        candidates.push((-OrderedFloat(distance), entry_node.id()));
        // add to the visited set
        visited.insert(entry_node.id());

        // let's go through the graph to find the best candidates
        while !candidates.is_empty() {
            let (current_dist, current_id) = candidates.pop().unwrap();

            // if the current distance is better than the best distance, we bound a better candidate
            if !best_candidates.is_empty() && -current_dist > best_candidates.peek().unwrap().0 {
                break;
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
                let distance = calculate_cosine_similarity(
                    &query,
                    self.nodes.get(&neighbour_id).unwrap().value(),
                );

                // if this candidate is better than the best candidate
                if best_candidates.len() < self.ef_construction
                    || -OrderedFloat(distance) > best_candidates.peek().unwrap().0
                {
                    // add the neighbour to the candidates
                    candidates.push((-OrderedFloat(distance), neighbour_id));
                    // this is our best candidate for this level
                    best_candidates.push((-OrderedFloat(distance), neighbour_id));

                    // ensure we are abiding the ef construction parameter
                    if best_candidates.len() > self.ef_construction {
                        best_candidates.pop();
                    }
                }
            }
        }

        // return our best candidates
        best_candidates.into_iter().map(|(_, id)| id).collect()
    }

    // insert_node inserts a new node into the world.
    // 1. pick the level at which to insert the node
    // 2. find the M nearest neighbors for the node at the chosen level
    // 3. connect the new node to the neighbors and on all lower levels
    // 4. recursively connect the new node to the neighbors' neighbors
    // 5. if the new node has no connections, add it to the graph at level 0
    pub(crate) fn insert_node(&mut self, node: &mut Node) -> Result<()> {
        let level = self.pick_node_level();

        // If this is the first node, initialize it as the entrypoint for all levels
        if self.nodes.is_empty() {
            self.level_entrypoints = vec![node.id(); self.max_level + 1];
            self.nodes.insert(node.id(), node.clone());
            return Ok(());
        }

        // for level and every one below, we need to connect the new node to the nearest neighbours on that level
        for level in (0..=level).rev() {
            let entrypoint_node = self.get_entrypoint_node_per_level(level);
            let nearest_neighbours = self.greedy_search(node.value(), &entrypoint_node, level);

            let mut nodes_to_prune = Vec::new();

            // connect the new node to the nearest neighbours
            for &neighbour_id in &nearest_neighbours {
                if let Some(neighbour) = self.nodes.get_mut(&neighbour_id) {
                    node.connect(neighbour, level);

                    // if the new node has more than M connections, we need to prune it
                    if node.connections(level).len() > self.m {
                        nodes_to_prune.push(node.id());
                    }

                    // if the neighbour has more than M connections, we need to prune it
                    if neighbour.connections(level).len() > self.m {
                        nodes_to_prune.push(neighbour_id);
                    }
                }
            }

            for node_id in nodes_to_prune {
                self.prune_node_connections(node_id, level);
            }
        }

        // add the new node to the world
        self.nodes.insert(node.id(), node.clone());

        Ok(())
    }

    fn prune_node_connections(&mut self, node_id: u32, level: usize) {
        let mut distances: Vec<(u32, f32)> = {
            let node = self.nodes.get(&node_id).unwrap();
            node.connections(level)
                .iter()
                .map(|&neighbour_id| {
                    let distance = node.distance(self.nodes.get(&neighbour_id).unwrap());
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

        let node = self.nodes.get_mut(&node_id).unwrap();
        node.remove_connections(&connections_to_remove, level);
    }

    // search gets the k nearest neighbours to the query vector using beam search
    pub fn search(&self, query: &Vec<f32>, k: usize, beam_width: usize) -> Vec<u32> {
        let candidates = self.beam_search(query, beam_width);

        let mut results: Vec<(u32, f32)> = candidates
            .into_iter()
            .map(|id| {
                let node = self.nodes.get(&id).unwrap();
                let distance = calculate_cosine_similarity(&query, node.value());
                (id, distance)
            })
            .collect();

        // Sort by distance (assuming higher cosine similarity is better)
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Return top k results
        results.into_iter().take(k).map(|(id, _)| id).collect()
    }

    fn beam_search(&self, query: &Vec<f32>, beam_width: usize) -> Vec<u32> {
        let mut candidates: BinaryHeap<(OrderedFloat<f32>, u32)> = BinaryHeap::new();
        // add the entrypoint node to the candidates
        let entrypoint_node = self.get_entrypoint_node();
        candidates.push((OrderedFloat(0.0), entrypoint_node.id()));

        // for every level,
        for level in (0..=self.max_level).rev() {
            let mut next_candidates = Vec::new();
            while let Some((_, candidate_id)) = candidates.pop() {
                // convert the ordered float back to a float
                let candidate = self.nodes.get(&candidate_id).unwrap();
                let local_best = self.greedy_search(&query, candidate, level);
                for &id in &local_best {
                    let dist =
                        calculate_cosine_similarity(&query, self.nodes.get(&id).unwrap().value());
                    next_candidates.push((OrderedFloat(dist), id));
                }
            }

            // keep the top beam width candidates
            candidates = next_candidates
                .into_iter()
                .collect::<BinaryHeap<_>>()
                .into_iter()
                .take(beam_width)
                .collect();
        }

        // return the best candidate
        candidates.into_iter().map(|(_, id)| id).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_world_insert_and_search() -> Result<()> {
        let mut world = World::new(5, 10, 10, 3)?;

        let test_vectors = vec![
            (1, vec![1.0, 0.0, 0.0]),
            (2, vec![0.0, 1.0, 0.0]),
            (3, vec![0.0, 0.0, 1.0]),
            (4, vec![0.7, 0.7, 0.0]),
        ];

        for (id, vector) in test_vectors {
            let level = world.pick_node_level();
            let mut node = Node::new(id, vector, level);
            world.insert_node(&mut node)?;
        }

        let query = vec![0.8, 0.8, 0.0];
        let results = world.search(&query, 2, 5);

        assert_eq!(results.len(), 2);
        assert!(results.contains(&4), "Should find vector [0.7, 0.7, 0.0]");

        Ok(())
    }
}
