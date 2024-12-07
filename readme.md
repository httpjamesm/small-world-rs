# small-world-rs

small-world-rs is an HNSW vector database written in Rust.

## Features

- Fast, accurate and easy to implement
- Choose your precision (16 or 32 bit floats)
- Choose your distance metric or bring your own
  - Out of the box, supports cosine distance (recommended for text) and euclidean distance (recommended for images)
- Serialize and deserialize for persistence

## Example

See the [text-embeddings example](./examples/text-embeddings/src/main.rs) for a simple example of how to use small-world-rs to perform semantic search over a set of text embeddings.

Basically, it works like this:

1. Get your embeddings, be that from OpenAI, Ollama, or wherever
2. Create a `World` with `World::new` or `World::new_from_dump`
3. Insert your vectors into the world with `world.insert_vector`
4. Perform a search with `world.search`
5. Dump the world with `world.dump` to save for later

## What config values should I use?

Key Parameters:

- `m`: Connections per layer

  - Recommended: 16-64
  - Sweet spot: 32
  - Higher values increase recall but consume more memory

- `ef_construction`: Construction-time exploration factor

  - Recommended: 100-500
  - Trade-off: Higher values = better recall but slower build time
  - Rule of thumb: 2-4× your target `ef_search`

- `ef_search`: Query-time exploration factor

  - Recommended: 50-150
  - Adjustable at search time
  - Higher values increase accuracy but slow down search
  - Tune based on recall requirements
