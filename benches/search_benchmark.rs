use criterion::{black_box, criterion_group, criterion_main, Criterion};
use small_world_rs::{
    distance_metric::{CosineDistance, DistanceMetric},
    primitives::vector::Vector,
    world::world::World,
};

fn create_random_vector(dim: usize) -> Vector {
    let values: Vec<f32> = (0..dim).map(|_| fastrand::f32()).collect();
    Vector::new_f32(&values)
}

fn create_test_world(size: usize, dim: usize) -> World {
    let mut world = World::new(32, 64, 64, DistanceMetric::Cosine(CosineDistance)).unwrap();

    for i in 0..size {
        world
            .insert_vector(i as u32, create_random_vector(dim))
            .unwrap();
    }

    world
}

fn bench_search(c: &mut Criterion) {
    let dimensions = 384; // typical embedding dimension
    let world_sizes = [1000, 10000, 100000];
    let beam_widths = [5, 10, 20];
    let k_values = [1, 5, 10];

    for &size in &world_sizes {
        let world = create_test_world(size, dimensions);
        let query = create_random_vector(dimensions);

        for &beam_width in &beam_widths {
            for &k in &k_values {
                let bench_name = format!("search_n{}_k{}_beam{}", size, k, beam_width);
                c.bench_function(&bench_name, |b| {
                    b.iter(|| {
                        world
                            .search(black_box(&query), black_box(k), black_box(beam_width))
                            .unwrap()
                    })
                });
            }
        }
    }
}

criterion_group!(benches, bench_search);
criterion_main!(benches);
