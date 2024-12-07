use crate::{
    metrics::{calculate_cosine_similarity, calculate_euclidean_distance},
    primitives::vector::Vector,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub enum DistanceMetric {
    Cosine(CosineDistance),
    Euclidean(EuclideanDistance),
}

impl DistanceMetric {
    pub fn distance(&self, a: &Vector, b: &Vector) -> f32 {
        match self {
            DistanceMetric::Cosine(metric) => metric.distance(a, b),
            DistanceMetric::Euclidean(metric) => metric.distance(a, b),
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct CosineDistance;

impl CosineDistance {
    fn distance(&self, a: &Vector, b: &Vector) -> f32 {
        1.0 - calculate_cosine_similarity(a, b)
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct EuclideanDistance;

impl EuclideanDistance {
    fn distance(&self, a: &Vector, b: &Vector) -> f32 {
        calculate_euclidean_distance(a, b)
    }
}
