# Text Embeddings Example

This example shows how to use small-world-rs to perform semantic search over a set of text embeddings.

## Setup

You'll need to have [Ollama](https://ollama.com/) installed to run this example.

Pull the embeddings model from Ollama:

```bash
ollama pull nomic-embed-text
```

## Running the example

```bash
cargo run
```

If it's your first time running the example, it will grab all the embeddings from the local Ollama instance, cache them, then create the World index.

## Searching

It'll pop up with an interactive prompt to search the index. Just type in something like "cryptocurrency" and hit enter. You'll likely see the result from the dataset saying:

"The blockchain technology ensures transparency and security in transactions."
