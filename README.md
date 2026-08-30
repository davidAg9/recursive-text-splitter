# recursive-text-splitter

A high-performance pure Rust port of Python LangChain's `RecursiveCharacterTextSplitter`.

Splits text into chunks using a recursive strategy with custom separators. Tries each separator in priority order and splits on the first viable one, then recursively splits oversized chunks with remaining separators. Merges splits with configurable overlap.

**5-7x faster** than the Python version while producing identical output.

## Why not just use `text-splitter` or `character_text_splitter`?

| Crate | Version | Custom Separators | Recursive Fallback | keep_separator | Python Compatible |
|---|---|---|---|---|---|
| `text-splitter` | 0.32 | ❌ (hardcoded ICU semantic boundaries) | ❌ | N/A | ❌ |
| `character_text_splitter` | 0.1.3 | ❌ (single separator only) | ❌ | ✅ | ❌ |
| `recursive-text-splitter` | 0.1.0 | ✅ | ✅ | ✅ | ✅ |

Both existing crates use Unicode ICU segmenters (sentences, words, graphemes) for semantic splitting. Neither supports injecting custom separator hierarchies like `["\nINT.", "\nEXT.", "\n\n", "\n", " ", ""]`. `recursive-text-splitter` implements LangChain's exact algorithm from scratch.

## Benchmark

### Results — Rust vs Python

Benchmarked on Star Wars script-style text and a 21.7M character real-text corpus
(Shakespeare + War and Peace + Les Misérables + KJV Bible + other PG works combined)
with separators `["\nINT.", "\nEXT.", "\n\n", "\n", " ", ""]`, chunk_size=200, chunk_overlap=50:

| Input | Text Length | Rust (ms) | Python (ms) | Speedup |
|--------|:-----------:|:---------:|:-----------:|:-------:|
| generated_100 | 30,670 | 0.05 | 0.30 | 6.0x |
| generated_500 | 154,670 | 0.28 | 1.44 | 5.1x |
| generated_1_000 | 309,670 | 0.49 | 2.89 | 5.9x |
| generated_5_000 | 1,561,670 | 2.32 | 14.7 | 6.3x |
| **large_20m_plus** | **21,711,529** | **76** | **250** | **3.3x** |

Tested with a real 20M+ character text file combining works from Project Gutenberg
(Shakespeare, War and Peace, Les Misérables, KJV Bible, etc.). Rust processes 21.7M
characters in **76ms** versus Python's **250ms**.

### Find the Graph

The benchmark comparison graph is saved as an interactive HTML file with two charts:

```
docs/benchmark_comparison.html
```

This includes **both** a Rust-vs-Python comparison and a Sequential-vs-Parallel Rust comparison.

View it by opening the file:
```bash
open docs/benchmark_comparison.html
# or
python3 -m http.server && open http://localhost:8000/docs/benchmark_comparison.html
```

Re-generate it with your own runs:
```bash
# Build both benchmarks
cargo build --release --bench cross_lang_bench
cargo build --release --bench parallel_bench --features rayon

# Rust vs Python benchmark
target/release/deps/cross_lang_bench-* > rust_results.txt
PYTHONPATH="" python3 benches/cross_lang_bench.py --large-file /tmp/combined_20m_final.txt > python_results.txt

# Parallel vs sequential benchmark
BENCH_TEXT_FILE=/tmp/combined_20m_final.txt target/release/deps/parallel_bench-* > parallel_bench.txt

# Generate graph with both comparisons
python3 benches/generate_graph.py \
    --rust-output rust_results.txt \
    --python-output python_results.txt \
    --parallel-output parallel_bench.txt \
    --output docs/benchmark_comparison.html
```

### Efficiency Advantages over Python

1. **`&str` byte slicing** — no string cloning until final chunk output. Python creates many intermediate `str` objects during `re.split` and `text.split`.
2. **Pre-allocated output capacity** — `Vec::with_capacity` based on estimated chunk count.
3. **Single-pass separator search** — uses `str::find` (Boyer-Moore-like) instead of Python's `re.escape` + `re.search` (regex compilation overhead).
4. **Minimal allocations** — `.clone()` only happens when pushing into `Vec<String>`. The hot path operates on references.
5. **Zero-copy `split_by_separator`** — scans separator positions in a single pass, no regex allocation.

## Usage

### Basic

```rust
use recursive_text_splitter::RecursiveCharacterTextSplitter;

let splitter = RecursiveCharacterTextSplitter::new()
    .with_separators(vec!["\nINT.", "\nEXT.", "\n\n", "\n", " ", ""])
    .with_chunk_size(200)
    .with_chunk_overlap(50);

let text = "INT. COCKPIT - DAY\n\nLUKE: I am a Jedi.\n\nEXT. FOREST - NIGHT\n\nVADER: Join me.";
let chunks = splitter.split_text_content(text);

for chunk in &chunks {
    println!("{}", chunk);
}
```

### With metadata (start_index)

```rust
let splitter = RecursiveCharacterTextSplitter::new()
    .with_chunk_size(200)
    .with_chunk_overlap(50)
    .with_add_start_index(true);

let chunks = splitter.split_text(text);
for chunk in &chunks {
    println!("[{}] {}", chunk.start_index.unwrap_or(0), chunk.page_content);
}
```

### Configuration options

| Method | Default | Description |
|---|---|---|
| `with_separators(Vec<&'static str>)` | `["\n\n", "\n", " ", ""]` | Separators tried in priority order |
| `with_chunk_size(usize)` | `4096` | Maximum characters per chunk |
| `with_chunk_overlap(usize)` | `0` | Characters to overlap between chunks |
| `with_add_start_index(bool)` | `false` | Include start_index in Chunk metadata |
| `with_strip_whitespace(bool)` | `true` | Trim whitespace from chunk edges |
| `with_keep_separator(bool)` | `true` | Keep separators attached to splits |
| `with_byte_length_function()` | (char count) | Use byte count instead of char count |
| `with_strategy::<S: SplitterStrategy>()` | `SequentialStrategy` | Select splitting strategy |

### SplitterStrategy trait

The `SplitterStrategy` trait allows selecting between sequential and parallel implementations at compile time:

```rust
use recursive_text_splitter::{RecursiveCharacterTextSplitter, SequentialStrategy, ParallelStrategy};

// Default sequential (works without any features)
let splitter = RecursiveCharacterTextSplitter::new()
    .with_chunk_size(200)
    .with_strategy::<SequentialStrategy>();

// Parallel via rayon (requires --features rayon)
let splitter = RecursiveCharacterTextSplitter::new()
    .with_chunk_size(200)
    .with_strategy::<ParallelStrategy>();

let chunks = splitter.split_text_content(text);
```

| Strategy | Feature | Description |
|---|---|---|
| `SequentialStrategy` | (default) | Single-threaded recursive splitting |
| `ParallelStrategy` | `rayon` | Multi-core splitting via rayon, same output |

### Parallel processing (optional `rayon` feature)

For large texts that produce many oversized splits, enable the `rayon` feature to leverage multi-core parallelism:

```toml
[dependencies]
recursive-text-splitter = { version = "0.1", features = ["rayon"] }
```

```rust
use recursive_text_splitter::{RecursiveCharacterTextSplitter, ParallelStrategy};

let splitter = RecursiveCharacterTextSplitter::new()
    .with_chunk_size(200)
    .with_chunk_overlap(50)
    .with_strategy::<ParallelStrategy>();

// Or use the explicit parallel API methods:
let chunks = splitter.split_text_content(text);  // uses parallel strategy
let chunks = splitter.split_text_content_par(&text);  // explicit parallel method
let chunks = splitter.split_text_par(text);  // explicit parallel with metadata
```

The parallel API also exposes convenience methods (all gated behind `#[cfg(feature = "rayon")]`):

| Method | Description |
|---|---|
| `split_text_par(&str) -> Vec<Chunk>` | Parallel version of `split_text()` |
| `split_text_content_par(&str) -> Vec<String>` | Parallel version of `split_text_content()` |
| `is_parallel() -> bool` | Returns `true` when the `rayon` feature is enabled |

**Algorithm**: The parallel version (`_split_text_par`) uses a 3-phase approach that preserves identical output ordering while parallelizing independent work:
1. **Sequential classification pass**: Walk through splits in order, flushing `good_splits` via `merge_splits_with_sep` as in Python's `_split_text`. Oversized splits that need recursion are collected as "deferred" work items.
2. **Parallel recursion**: All deferred oversized splits are processed concurrently via `rayon::par_iter`, dispatching recursive `_split_text_par` calls to the thread pool (cross-recursion parallelism — each recursive level also parallelizes).
3. **Ordered reassembly**: Deferred result placeholders are replaced with parallel results (in original order) and flattened.

This is correct because oversized splits are independent recursive calls on disjoint subtexts — their results can be computed in any order and still assembled correctly. Verified by 3 `par_correctness` tests comparing byte-for-byte identical output across simple, medium (500 scenes), and overlap-heavy configurations.

### Parallel Benchmark

The parallel API is benchmarked by `benches/parallel_bench.rs`:

```bash
cargo build --release --bench parallel_bench --features rayon
target/release/deps/parallel_bench-*  # auto-generates test texts
```

**Results (rayon on multi-core CPU)**:

| Input | Sequential | Parallel | Speedup | Chunks (identical?) |
|---|---|---|---|---|
| 100 scenes (30K chars) | 0.05ms | 0.12ms | 0.41x | ✅ 200 = 200 |
| 500 scenes (154K chars) | 0.23ms | 0.30ms | 0.77x | ✅ 1000 = 1000 |
| 1000 scenes (309K chars) | 0.46ms | 0.46ms | 0.99x | ✅ 2000 = 2000 |
| 5000 scenes (1.56M chars) | 2.42ms | 1.84ms | 1.31x | ✅ 10000 = 10000 |
| 20M+ chars | 73.8ms | 51.1ms | **1.44x** | ✅ 145116 = 145116 |

Parallelism overhead (~0.07ms thread pool spin-up) makes small texts slower, but large texts benefit from multi-core distribution — 21.7M characters process in **51ms instead of 74ms**.

### Graph

Open `docs/benchmark_comparison.html` in a browser to view an interactive Chart.js chart with two panels:
- **Rust vs Python** — bar chart comparing recursive-text-splitter vs langchain-text-splitters
- **Sequential vs Parallel** — bar chart comparing single-core vs multi-core (rayon) Rust splitting

## With Ollama + Local Embeddings (RAG)

This crate integrates with [Ollama](https://ollama.com/) for local embeddings — no external API keys needed. Use it with any local embedding model:

### Prerequisites

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama serve

# Pull a local embedding model
ollama pull nomic-embed-text    # 768-dim (default)
ollama pull qwen3-embed:latest  # 128-dim (Qwen3)
```

### Cargo.toml

```toml
[dependencies]
recursive-text-splitter = { version = "0.1", features = ["rayon"] }
reqwest       = { version = "0.12", features = ["json"] }
serde_json    = "1"
anyhow        = "1"
tokio         = { version = "1", features = ["full"] }
```

### Full RAG pipeline

```rust
use recursive_text_splitter::{RecursiveCharacterTextSplitter, ParallelStrategy};
use serde::Deserialize;

#[derive(Deserialize)]
struct OllamaEmbedResponse {
    embedding: Vec<f32>,
}

async fn embed_text(http: &reqwest::Client, text: &str) -> Vec<f32> {
    http
        .post("http://localhost:11434/api/embed")
        .json(&serde_json::json!({
            "model": "nomic-embed-text",
            "input": text,
        }))
        .send()
        .await
        .unwrap()
        .json::<OllamaEmbedResponse>()
        .await
        .unwrap()
        .embedding
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let text = std::fs::read_to_string("data/my_documents.txt")?;

    // 1. Split text with custom separators
    let splitter = RecursiveCharacterTextSplitter::new()
        .with_separators(vec!["\nINT.", "\nEXT.", "\n\n", "\n", " ", ""])
        .with_chunk_size(2500)
        .with_chunk_overlap(250)
        .with_add_start_index(true)
        .with_strategy::<ParallelStrategy>();

    let chunks = splitter.split_text(&text);

    // 2. Generate embeddings via Ollama
    let http = reqwest::Client::new();
    let embedded: Vec<(usize, Vec<f32>)> = futures::future::join_all(
        chunks.iter().enumerate().map(|(i, chunk)| async {
            let embedding = embed_text(&http, &chunk.page_content).await;
            (i, embedding)
        })
    ).await;

    // 3. Store in your vector database of choice (Qdrant, Pinecone, etc.)
    println!("{} chunks embedded and ready to index", embedded.len());

    Ok(())
}
```

### With `rig` (optional feature)

Enable the `rig` feature for integration with the rig framework:

```toml
[dependencies]
recursive-text-splitter = { version = "0.1", features = ["rayon", "rig"] }
```

## License

MIT
