use crate::primitives::vector::Vector;
use simsimd::SpatialSimilarity;

pub(crate) fn calculate_cosine_similarity(a: &Vector, b: &Vector) -> f32 {
    1.0 - f32::cosine(a.as_slice().as_ref(), b.as_slice().as_ref()).unwrap() as f32
}

#[cfg(test)]
mod tests {

    use super::*;
    use assert_approx_eq::assert_approx_eq;

    // test cosine similarity
    #[test]
    fn test_cosine_similarity() {
        // Test case 1: Basic vectors with known similarity
        let v1 = Vector::new_f32(&[1.0, 2.0, 3.0]);
        let v2 = Vector::new_f32(&[4.0, 5.0, 6.0]);
        let result = calculate_cosine_similarity(&v1, &v2);
        assert_approx_eq!(result, 0.9746318, 1e-6);

        // Test case 2: Identical vectors (should be 1.0)
        let v3 = Vector::new_f32(&[1.0, 0.0, 0.0]);
        let v4 = Vector::new_f32(&[1.0, 0.0, 0.0]);
        assert_approx_eq!(calculate_cosine_similarity(&v3, &v4), 1.0, 1e-6);

        // Test case 3: Orthogonal vectors (should be 0.0)
        let v5 = Vector::new_f32(&[1.0, 0.0, 0.0]);
        let v6 = Vector::new_f32(&[0.0, 1.0, 0.0]);
        assert_approx_eq!(calculate_cosine_similarity(&v5, &v6), 0.0, 1e-6);

        // Test case 4: Opposite vectors (should be -1.0)
        let v7 = Vector::new_f32(&[1.0, 0.0, 0.0]);
        let v8 = Vector::new_f32(&[-1.0, 0.0, 0.0]);
        assert_approx_eq!(calculate_cosine_similarity(&v7, &v8), -1.0, 1e-6);
    }
}
