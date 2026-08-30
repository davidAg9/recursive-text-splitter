#!/usr/bin/env python3
"""
Python side of the cross-language benchmark.
Run with: PYTHONPATH="" python3 benches/cross_lang_bench.py [--large-file PATH]

Generates the same text inputs and times Python's langchain-text-splitters.
Results can be compared against the Rust benchmark output.
"""

import timeit
import sys
import argparse
from pathlib import Path


def generate_text(num_scenes):
    text = ""
    for i in range(num_scenes):
        text += f"INT. ROOM {i} - DAY\n\n"
        text += (
            f"LUKE: This is scene {i}. It has some dialogue that goes on for quite a "
            f"while to make the chunk interesting. The force is strong with this one.\n\n"
        )
        text += f"EXT. FOREST {i} - NIGHT\n\n"
        text += (
            f"VADER: Join me. The dark side of the force is compelling and powerful "
            f"beyond measure. We can rule the galaxy together.\n\n"
        )
    return text


def main():
    parser = argparse.ArgumentParser(description="Python benchmark for text splitting")
    parser.add_argument("--large-file", default=None, help="Path to a 20M+ char text file")
    parser.add_argument("--iterations", type=int, default=10, help="Iterations per generated size")
    args = parser.parse_args()

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        print("langchain-text-splitters not installed. Run: pip install langchain-text-splitters")
        sys.exit(1)

    splitter = RecursiveCharacterTextSplitter(
        separators=["\nINT.", "\nEXT.", "\n\n", "\n", " ", ""],
        chunk_size=200,
        chunk_overlap=50,
    )

    print("Python langchain-text-splitters benchmark")
    print("=========================================")
    print(f"{'input':<20} {'text_chars':<15} {'avg_ms':<15} {'chunks':<10}")

    # Generated Star Wars script texts
    for size in [100, 500, 1000, 5000]:
        text = generate_text(size)
        text_len = len(text)

        # Warmup
        splitter.split_text(text)

        elapsed = timeit.timeit(lambda: splitter.split_text(text), number=args.iterations)
        avg_ms = (elapsed / args.iterations) * 1000

        chunks = splitter.split_text(text)
        print(f"generated_{size:<14} {text_len:<15} {avg_ms:<15.3} {len(chunks):<10}")

    # Large real text file (20M+ chars)
    if args.large_file:
        large_path = Path(args.large_file)
        if large_path.exists():
            text = large_path.read_text()
            text_len = len(text)

            # Warmup
            splitter.split_text(text)

            elapsed = timeit.timeit(lambda: splitter.split_text(text), number=1)
            avg_ms = elapsed * 1000

            chunks = splitter.split_text(text)
            print(f"large_20m_plus    {text_len:<15} {avg_ms:<15.3} {len(chunks):<10}")


if __name__ == "__main__":
    main()
