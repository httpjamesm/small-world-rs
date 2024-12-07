use crate::{
    metrics::{calculate_cosine_similarity, calculate_euclidean_distance},
    primitives::vector::Vector,
};

pub trait DistanceMetric {
    fn distance(&self, a: &Vector, b: &Vector) -> f32;
}

pub struct CosineDistance;

impl DistanceMetric for CosineDistance {
    fn distance(&self, a: &Vector, b: &Vector) -> f32 {
        1.0 - calculate_cosine_similarity(a, b)
    }
}

pub struct EuclideanDistance;

impl DistanceMetric for EuclideanDistance {
    fn distance(&self, a: &Vector, b: &Vector) -> f32 {
        calculate_euclidean_distance(a, b)
    }
}
