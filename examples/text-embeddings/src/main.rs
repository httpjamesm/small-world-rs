use anyhow::Result;
use reqwest::Client;
use serde_json::{json, Value};
use small_world_rs::{
    distance_metric::{CosineDistance, DistanceMetric},
    primitives::vector::Vector,
    world::world::World,
};
use std::{
    fs,
    io::{BufRead, BufReader, Write},
    path::Path,
};
use tokio;

const CACHE_FILE: &str = ".embeddings-cache";
const DATASET_FILE: &str = "dataset.txt";
const WORLD_FILE: &str = "world.smallworld";

async fn get_embedding(client: &Client, text: &str) -> Result<Vec<f32>> {
    let response = client
        .post("http://localhost:11434/api/embeddings")
        .json(&json!({
            "model": "nomic-embed-text",
            "prompt": text
        }))
        .send()
        .await?;

    let response: Value = response.json().await?;
    let embedding: Vec<f32> = response["embedding"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();

    Ok(embedding)
}

#[tokio::main]
async fn main() -> Result<()> {
    let client = Client::new();
    let dataset = fs::read_to_string(DATASET_FILE)?
        .lines()
        .map(String::from)
        .collect::<Vec<_>>();

    let mut world = if Path::new(WORLD_FILE).exists() {
        let world_data = fs::read(WORLD_FILE)?;
        World::new_from_dump(&world_data)?
    } else {
        let mut world = World::new(16, 32, 32, DistanceMetric::Cosine(CosineDistance))?;
        let cache_path = Path::new(CACHE_FILE);

        if !cache_path.exists() {
            let mut file = fs::File::create(cache_path)?;
            for text in &dataset {
                let embedding = get_embedding(&client, text).await?;
                let json = serde_json::to_string(&embedding)?;
                writeln!(file, "{}", json)?;
            }
        }

        let file = fs::File::open(cache_path)?;
        let reader = BufReader::new(file);

        for (id, line) in reader.lines().enumerate() {
            let line = line?;
            println!("Processing line {}", id);
            let embedding: Vec<f32> = serde_json::from_str(&line)?;
            let vector = Vector::new_f32(&embedding);
            world.insert_vector(id as u32, vector)?;
        }

        // Dump the world after construction
        let world_data = world.dump()?;
        fs::write(WORLD_FILE, world_data)?;

        world
    };

    println!("Successfully processed {} texts", dataset.len());
    println!("Available commands:");
    println!("  search <query> - Search for similar texts");
    println!("  delete <id>   - Delete a node by ID");
    println!("  quit         - Exit the program");

    loop {
        println!("\nEnter a command: ");
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        let input = input.trim();

        let parts: Vec<&str> = input.splitn(2, ' ').collect();
        match parts.get(0).map(|s| *s) {
            Some("search") => {
                let query = parts
                    .get(1)
                    .ok_or_else(|| anyhow::anyhow!("Missing query"))?;
                let embedding = get_embedding(&client, query).await?;
                let vector = Vector::new_f32(&embedding);
                let nearest_neighbours = world.search(&vector, 5, 5)?;

                println!("\nNearest neighbours:");
                for (idx, neighbour) in nearest_neighbours.iter().enumerate() {
                    println!("ID {}: {}", neighbour, dataset[*neighbour as usize]);
                }
            }
            Some("delete") => {
                let id = parts
                    .get(1)
                    .and_then(|s| s.parse::<u32>().ok())
                    .ok_or_else(|| anyhow::anyhow!("Invalid ID"))?;

                match world.delete_node(id) {
                    Ok(_) => {
                        println!("Successfully deleted node {}", id);
                        // Save the updated world
                        let world_data = world.dump()?;
                        fs::write(WORLD_FILE, world_data)?;
                    }
                    Err(e) => println!("Failed to delete node: {}", e),
                }
            }
            Some("quit") => break,
            _ => println!("Invalid command. Use 'search <query>', 'delete <id>', or 'quit'"),
        }
    }

    Ok(())
}
