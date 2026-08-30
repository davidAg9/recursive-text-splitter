/// Benchmark comparing sequential vs parallel splitting performance.
///
/// Usage:
///   BENCH_TEXT_FILE=/tmp/combined_20m_final.txt cargo run --release --bench parallel_bench
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

fn bench(
    label: &str,
    text: &str,
    splitter: &RecursiveCharacterTextSplitter,
    iterations: usize,
) -> (usize, f64) {
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
        "{:<25} {:<15} {:<12.3} {:<10}",
        label,
        text.chars().count(),
        avg_ms,
        total_chunks
    );
    (total_chunks, avg_ms)
}

#[cfg(feature = "rayon")]
fn bench_par(
    label: &str,
    text: &str,
    splitter: &RecursiveCharacterTextSplitter,
    iterations: usize,
) -> (usize, f64) {
    use rayon::prelude::*;
    // Warmup
    let _ = splitter.split_text_content_par(text);

    let start = Instant::now();
    let mut total_chunks = 0;
    for _ in 0..iterations {
        let chunks = splitter.split_text_content_par(text);
        total_chunks = chunks.len();
    }
    let elapsed = start.elapsed();
    let avg_ms = elapsed.as_secs_f64() / iterations as f64 * 1000.0;
    println!(
        "{:<25} {:<15} {:<12.3} {:<10}",
        label,
        text.chars().count(),
        avg_ms,
        total_chunks
    );
    (total_chunks, avg_ms)
}

fn main() {
    let splitter_seq = RecursiveCharacterTextSplitter::new()
        .with_separators(vec!["\nINT.", "\nEXT.", "\n\n", "\n", " ", ""])
        .with_chunk_size(200)
        .with_chunk_overlap(50);

    #[cfg(feature = "rayon")]
    let splitter_par = RecursiveCharacterTextSplitter::new()
        .with_separators(vec!["\nINT.", "\nEXT.", "\n\n", "\n", " ", ""])
        .with_chunk_size(200)
        .with_chunk_overlap(50);

    let iterations: usize = 10;

    println!("Sequential vs Parallel recursive-text-splitter");
    println!("=============================================");
    println!(
        "{:<25} {:<15} {:<12} {:<10}",
        "input", "text_chars", "avg_ms", "chunks"
    );
    println!();

    // Test with generated Star Wars texts
    for &size in &[100, 500, 1000, 5000] {
        let text = generate_text(size);

        print!("--- {} scenes ---\n", size);
        bench("sequential", &text, &splitter_seq, iterations);
        #[cfg(feature = "rayon")]
        bench_par("parallel (rayon)", &text, &splitter_par, iterations);
        println!();
    }

    // Test with large real text file
    if let Ok(path) = std::env::var("BENCH_TEXT_FILE") {
        if std::path::Path::new(&path).exists() {
            let text = std::fs::read_to_string(&path).unwrap_or_default();
            print!("--- large_20m_plus ---\n");
            bench("sequential", &text, &splitter_seq, 1);
            #[cfg(feature = "rayon")]
            bench_par("parallel (rayon)", &text, &splitter_par, 1);
        }
    }

    println!();
    println!("Note: parallelism overhead may make small texts slower. ");
    println!("Benefits appear on large texts with many oversized splits.");
}
