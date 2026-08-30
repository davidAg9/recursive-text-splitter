use criterion::{Criterion, black_box, criterion_group, criterion_main};
use recursive_text_splitter::{LengthFunction, RecursiveCharacterTextSplitter};

/// Generate a realistic Star Wars script text for benchmarking.
fn generate_sw_script(num_scenes: usize) -> String {
    let mut text = String::new();
    for i in 0..num_scenes {
        text.push_str(&format!("INT. ROOM {} - DAY\n\n", i));
        text.push_str(&format!("LUKE: This is scene {}. It has some dialogue that goes on for quite a while to make the chunk interesting.\n\n", i));
        text.push_str(&format!("EXT. FOREST {} - NIGHT\n\n", i));
        text.push_str(&format!("VADER: Join me. The dark side of the force is compelling and powerful beyond measure.\n\n"));
    }
    text
}

fn bench_recursive_splitter(c: &mut Criterion) {
    let splitter = RecursiveCharacterTextSplitter::new()
        .with_separators(vec!["\nINT.", "\nEXT.", "\n\n", "\n", " ", ""])
        .with_chunk_size(200)
        .with_chunk_overlap(50);

    let text = generate_sw_script(100);

    c.bench_function("recursive_split_100_scenes", |b| {
        b.iter(|| {
            let chunks = splitter.split_text_content(black_box(&text));
            black_box(chunks.len());
        })
    });
}

fn bench_recursive_splitter_large(c: &mut Criterion) {
    let splitter = RecursiveCharacterTextSplitter::new()
        .with_separators(vec!["\nINT.", "\nEXT.", "\n\n", "\n", " ", ""])
        .with_chunk_size(200)
        .with_chunk_overlap(50);

    let text = generate_sw_script(1000);

    c.bench_function("recursive_split_1000_scenes", |b| {
        b.iter(|| {
            let chunks = splitter.split_text_content(black_box(&text));
            black_box(chunks.len());
        })
    });
}

fn bench_char_count_vs_byte(c: &mut Criterion) {
    let text = generate_sw_script(50);

    let char_splitter = RecursiveCharacterTextSplitter::new()
        .with_separators(vec!["\nINT.", "\nEXT.", "\n\n", "\n", " ", ""])
        .with_chunk_size(200)
        .with_chunk_overlap(50);

    let byte_splitter = char_splitter.clone().with_byte_length_function();

    c.bench_function("char_length", |b| {
        b.iter(|| {
            let chunks = char_splitter.split_text_content(black_box(&text));
            black_box(chunks.len());
        })
    });

    c.bench_function("byte_length", |b| {
        b.iter(|| {
            let chunks = byte_splitter.split_text_content(black_box(&text));
            black_box(chunks.len());
        })
    });
}

fn bench_keep_separator_on_off(c: &mut Criterion) {
    let text = generate_sw_script(100);

    let keep_on = RecursiveCharacterTextSplitter::new()
        .with_chunk_size(200)
        .with_chunk_overlap(50)
        .with_keep_separator(true);

    let keep_off = RecursiveCharacterTextSplitter::new()
        .with_chunk_size(200)
        .with_chunk_overlap(50)
        .with_keep_separator(false);

    c.bench_function("keep_separator_true", |b| {
        b.iter(|| {
            let chunks = keep_on.split_text_content(black_box(&text));
            black_box(chunks.len());
        })
    });

    c.bench_function("keep_separator_false", |b| {
        b.iter(|| {
            let chunks = keep_off.split_text_content(black_box(&text));
            black_box(chunks.len());
        })
    });
}

criterion_group!(
    benches,
    bench_recursive_splitter,
    bench_recursive_splitter_large,
    bench_char_count_vs_byte,
    bench_keep_separator_on_off,
);
criterion_main!(benches);
