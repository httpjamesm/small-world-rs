fn calculate_dot_product(a: &Vec<f32>, b: &Vec<f32>) -> f32 {
    a.iter().zip(b.iter()).map(|(a, b)| a * b).sum()
}

fn get_vector_magnitude(a: &Vec<f32>) -> f32 {
    a.iter().map(|a| a * a).sum::<f32>().sqrt()
}

pub(crate) fn calculate_cosine_similarity(a: &Vec<f32>, b: &Vec<f32>) -> f32 {
    calculate_dot_product(a, b) / (get_vector_magnitude(a) * get_vector_magnitude(b))
}
