fn calculate_dot_product(a: &Vec<f32>, b: &Vec<f32>) -> f32 {
    a.iter().zip(b.iter()).map(|(a, b)| a * b).sum()
}

fn get_vector_magnitude(a: &Vec<f32>) -> f32 {
    a.iter().map(|a| a * a).sum::<f32>().sqrt()
}

pub(crate) fn calculate_cosine_similarity(a: &Vec<f32>, b: &Vec<f32>) -> f32 {
    calculate_dot_product(a, b) / (get_vector_magnitude(a) * get_vector_magnitude(b))
}

#[cfg(test)]
mod tests {

    use super::*;
    use assert_approx_eq::assert_approx_eq;

    // test the dot product of two vectors
    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert_eq!(calculate_dot_product(&a, &b), 32.0);
    }

    // test getting vector magnitude
    #[test]
    fn test_vector_magnitude() {
        let a = vec![1.0, 2.0, 3.0];
        let magnitude = ((1.0 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0) as f32).sqrt();
        assert_eq!(get_vector_magnitude(&a), magnitude);
    }

    // test cosine similarity
    #[test]
    fn test_cosine_similarity() {
        // Test case 1: Basic vectors with known similarity
        let v1 = vec![1.0, 2.0, 3.0];
        let v2 = vec![4.0, 5.0, 6.0];
        let result = calculate_cosine_similarity(&v1, &v2);
        assert_approx_eq!(result, 0.9746318, 1e-6);

        // Test case 2: Identical vectors (should be 1.0)
        let v3 = vec![1.0, 0.0, 0.0];
        let v4 = vec![1.0, 0.0, 0.0];
        assert_approx_eq!(calculate_cosine_similarity(&v3, &v4), 1.0, 1e-6);

        // Test case 3: Orthogonal vectors (should be 0.0)
        let v5 = vec![1.0, 0.0, 0.0];
        let v6 = vec![0.0, 1.0, 0.0];
        assert_approx_eq!(calculate_cosine_similarity(&v5, &v6), 0.0, 1e-6);

        // Test case 4: Opposite vectors (should be -1.0)
        let v7 = vec![1.0, 0.0, 0.0];
        let v8 = vec![-1.0, 0.0, 0.0];
        assert_approx_eq!(calculate_cosine_similarity(&v7, &v8), -1.0, 1e-6);
    }
}
