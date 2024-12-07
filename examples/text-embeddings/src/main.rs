use anyhow::Result;
use reqwest::Client;
use serde_json::{json, Value};
use small_world_rs::world::world::World;
use std::{
    fs,
    io::{BufRead, BufReader, Write},
    path::Path,
};
use tokio;

const CACHE_FILE: &str = ".embeddings-cache";
const DATASET_FILE: &str = "dataset.txt";

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
    let embedding = response["embedding"]
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

    let mut world = World::new(16, 32, 32, 2)?;
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
        world.insert_vector(id as u32, embedding)?;
    }

    println!("Successfully processed {} texts", dataset.len());

    // now do interactive terminal to search semantically
    loop {
        println!("Enter a query: ");
        let mut query = String::new();
        std::io::stdin().read_line(&mut query)?;
        query = query.trim().to_string();

        let embedding = get_embedding(&client, &query).await?;
        let nearest_neighbours = world.search(&embedding, 5, 5);
        println!("Nearest neighbours: {:?}", nearest_neighbours);

        // show the lines from the dataset that correspond to the nearest neighbours
        for neighbour in nearest_neighbours {
            println!("{}", dataset[neighbour as usize]);
        }
    }

    Ok(())
}