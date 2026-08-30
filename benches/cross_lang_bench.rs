/// Cross-language benchmark comparing Rust recursive-text-splitter vs Python langchain.
///
/// This binary runs the Rust implementation on identical inputs and reports
/// timing data that can be compared against the Python benchmark output.
///
/// Usage:
///   cargo bench --bench cross_lang_bench --release
///   # Then run the Python side:
///   PYTHONPATH="" python3 benches/cross_lang_bench.py
///
/// Environment variables:
///   BENCH_TEXT_FILE — path to a large text file for the mega-benchmark (20M+ chars)
///   BENCH_ITERATIONS — number of iterations per size (default 10)
use recursive_text_splitter::RecursiveCharacterTextSplitter;
use std::time::Instant;

fn generate_text(num_scenes: usize) -> String {
    let mut text = String::new();
    for i in 0..num_scenes {
        text.push_str(&format!("INT. ROOM {i} - DAY\n\n"));
        text.push_str(&format!(
            "LUKE: This is scene {i}. It has some dialogue that goes on for quite a \
             while to make the chunk interesting. The force is strong with this one.\n\n"
        ));
        text.push_str(&format!("EXT. FOREST {i} - NIGHT\n\n"));
        text.push_str(&format!(
            "VADER: Join me. The dark side of the force is compelling and powerful \
             beyond measure. We can rule the galaxy together.\n\n"
        ));
    }
    text
}

fn bench_generated(
    splitter: &RecursiveCharacterTextSplitter,
    label: &str,
    text: &str,
    iterations: usize,
) {
    let text_len = text.chars().count();

    // Warmup
    let _ = splitter.split_text_content(text);

    let start = Instant::now();
    let mut total_chunks = 0;
    for _ in 0..iterations {
        let chunks = splitter.split_text_content(text);
        total_chunks = chunks.len();
    }
    let elapsed = start.elapsed();
    let avg_ms = elapsed.as_secs_f64() / iterations as f64 * 1000.0;

    println!(
        "{:<20} {:<15} {:<15.3} {:<10}",
        label, text_len, avg_ms, total_chunks
    );
}

fn main() {
    let splitter = RecursiveCharacterTextSplitter::new()
        .with_separators(vec!["\nINT.", "\nEXT.", "\n\n", "\n", " ", ""])
        .with_chunk_size(200)
        .with_chunk_overlap(50);

    let iterations: usize = std::env::var("BENCH_ITERATIONS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);

    println!("Rust recursive-text-splitter benchmark");
    println!("=====================================");
    println!(
        "{:<20} {:<15} {:<15} {:<10}",
        "input", "text_chars", "avg_ms", "chunks"
    );

    // Generated Star Wars script texts
    for &size in &[100, 500, 1000, 5000] {
        let text = generate_text(size);
        bench_generated(&splitter, &format!("generated_{size}"), &text, iterations);
    }

    // Large real text file (20M+ chars)
    if let Ok(path) = std::env::var("BENCH_TEXT_FILE") {
        if std::path::Path::new(&path).exists() {
            let text = std::fs::read_to_string(&path).unwrap_or_default();
            bench_generated(&splitter, "large_20m_plus", &text, 1);
        }
    }
}
