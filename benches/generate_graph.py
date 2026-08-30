#!/usr/bin/env python3
"""
Generate a comparison graph of Rust vs Python text splitter performance.
Also includes sequential vs parallel Rust comparison.

Usage:
    # Run Rust benchmark (release mode for fair comparison)
    cargo build --release --bench parallel_bench --features rayon
    target/release/deps/parallel_bench-* > parallel_bench.txt

    # Run Rust vs Python benchmark
    cargo build --release --bench cross_lang_bench
    BENCH_TEXT_FILE=/tmp/combined_20m_final.txt target/release/deps/cross_lang_bench-* > rust_results.txt

    # Run Python benchmark
    PYTHONPATH="" python3 benches/cross_lang_bench.py --large-file /tmp/combined_20m_final.txt > python_results.txt

    # Generate graph
    python3 benches/generate_graph.py --rust-output rust_results.txt --python-output python_results.txt --parallel-output parallel_bench.txt --output docs/benchmark_comparison.html
"""
import sys
import os
import argparse
from pathlib import Path


def parse_bench_output(filepath):
    """
    Parse benchmark output lines.
    """
    results = []
    with open(filepath) as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 4:
            continue

        # Skip header/separator lines
        if parts[0].startswith("---") or parts[0].startswith("Note:") or parts[0].startswith("input"):
            continue

        try:
            if parts[0] in ("sequential", "parallel"):
                label = parts[0]
                if parts[1] == "(rayon)":
                    chars = int(parts[2])
                    avg_ms = float(parts[3])
                    chunks = int(parts[4])
                else:
                    chars = int(parts[1])
                    avg_ms = float(parts[2])
                    chunks = int(parts[3])
                results.append({
                    "label": label,
                    "chars": chars,
                    "avg_ms": avg_ms,
                    "chunks": chunks,
                })
            else:
                label = parts[0]
                chars = int(parts[1])
                avg_ms = float(parts[2])
                chunks = int(parts[3])
                results.append({
                    "label": label,
                    "chars": chars,
                    "avg_ms": avg_ms,
                    "chunks": chunks,
                })
        except (ValueError, IndexError):
            continue

    return results


def generate_html(rust_results, python_results, parallel_results, output_path):
    """Generate an HTML bar chart comparing Rust vs Python performance."""
    # Build chart data - use human-readable labels
    labels = []
    for r in rust_results:
        if r["label"].startswith("generated_"):
            labels.append(f"gen ({r['chars']:,} chars)")
        elif r["label"].startswith("large_"):
            labels.append(f"{r['chars'] / 1_000_000:.1f}M chars")
        else:
            labels.append(r["label"])

    rust_times = [r["avg_ms"] for r in rust_results]
    python_times = [r["avg_ms"] for r in python_results] if python_results else [0.0] * len(rust_results)

    # Calculate speedup
    speedups = []
    for r, p in zip(rust_results, python_results):
        if p["avg_ms"] > 0 and r["avg_ms"] > 0:
            speedups.append(round(p["avg_ms"] / r["avg_ms"], 2))
        else:
            speedups.append(0)

    # Build parallel vs sequential data
    par_labels = []
    seq_times = []
    par_times = []
    par_speedups = []

    if parallel_results:
        seq_data = [r for r in parallel_results if r["label"] == "sequential"]
        par_data = [r for r in parallel_results if r["label"] == "parallel"]

        for i in range(len(seq_data)):
            s = seq_data[i]
            p = par_data[i] if i < len(par_data) else s
            if s["chars"] > 1_000_000:
                par_labels.append(f"large ({s['chars']:,} chars)")
            else:
                par_labels.append(f"gen ({s['chars']:,} chars)")
            seq_times.append(s["avg_ms"])
            par_times.append(p["avg_ms"])
            if p["avg_ms"] > 0 and s["avg_ms"] > 0:
                par_speedups.append(round(s["avg_ms"] / p["avg_ms"], 2))
            else:
                par_speedups.append(0)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>recursive-text-splitter: Benchmark Results</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1000px;
            margin: 0 auto;
            padding: 24px;
            background: var(--card, #fff);
            color: var(--foreground, #1a1a1a);
        }}
        h1 {{
            font-size: 1.75rem;
            margin-bottom: 8px;
        }}
        .subtitle {{
            color: var(--muted-foreground, #666);
            margin-bottom: 24px;
        }}
        .summary {{
            display: flex;
            gap: 24px;
            margin-bottom: 32px;
            flex-wrap: wrap;
        }}
        .stat {{
            background: var(--muted, #f5f5f5);
            padding: 16px 24px;
            border-radius: 8px;
            text-align: center;
            min-width: 140px;
        }}
        .stat-value {{
            font-size: 1.75rem;
            font-weight: 700;
            color: var(--accent, #0070f3);
        }}
        .stat-label {{
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-top: 4px;
        }}
        .chart-container {{
            margin-bottom: 48px;
        }}
        h2 {{
            font-size: 1.25rem;
            margin-bottom: 12px;
        }}
    </style>
</head>
<body>
    <h1>recursive-text-splitter Benchmark</h1>
    <p class="subtitle">Rust vs Python (langchain-text-splitters) and Sequential vs Parallel Rust — {len(rust_results)} test cases, largest: {rust_results[-1]['chars']:,} chars</p>

    <div class="summary">
        <div class="stat">
            <div class="stat-value">{speedups[-1] if speedups else 0}x</div>
            <div class="stat-label">Rust Speedup vs Python (largest)</div>
        </div>
        <div class="stat">
            <div class="stat-value">{min(speedups) if speedups else 0}x</div>
            <div class="stat-label">Min Speedup vs Python</div>
        </div>
        <div class="stat">
            <div class="stat-value">{rust_times[-1]:.1f}ms</div>
            <div class="stat-label">Rust Time (21M chars)</div>
        </div>
        <div class="stat">
            <div class="stat-value">{python_times[-1]:.0f}ms</div>
            <div class="stat-label">Python Time (21M chars)</div>
        </div>
    </div>

    <div class="chart-container">
        <h2>Rust vs Python (langchain-text-splitters)</h2>
        <canvas id="rustVsPythonChart" height="400"></canvas>
    </div>

    <div class="chart-container">
        <h2>Parallel Speedup (rayon) vs Sequential</h2>
        <p><em>Same output, multi-core — parallelism helps on large texts with many oversized splits.</em></p>
        <canvas id="parallelChart" height="300"></canvas>
    </div>

    <script>
        // Rust vs Python chart
        new Chart(document.getElementById('rustVsPythonChart'), {{
            type: 'bar',
            data: {{
                labels: {labels},
                datasets: [
                    {{
                        label: 'Rust (recursive-text-splitter)',
                        data: {rust_times},
                        backgroundColor: 'rgba(10, 173, 137, 0.8)',
                        borderColor: 'rgba(10, 173, 137, 1)',
                        borderWidth: 1
                    }},
                    {{
                        label: 'Python (langchain-text-splitters)',
                        data: {python_times},
                        backgroundColor: 'rgba(255, 159, 64, 0.8)',
                        borderColor: 'rgba(255, 159, 64, 1)',
                        borderWidth: 1
                    }}
                ]
            }},
            options: {{
                responsive: true,
                indexAxis: 'y',
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Average time per split (ms, lower is better) — 1 iteration for 20M+ char file'
                    }},
                    tooltip: {{
                        callbacks: {{
                            label: function(context) {{
                                return context.dataset.label + ': ' + context.raw.toFixed(2) + 'ms';
                            }}
                        }}
                    }}
                }},
                scales: {{
                    x: {{
                        beginAtZero: true,
                        title: {{
                            display: true,
                            text: 'Average time (ms)'
                        }}
                    }},
                    y: {{
                        title: {{
                            display: true,
                            text: 'Input size'
                        }}
                    }}
                }}
            }}
        }});

        // Parallel vs Sequential chart
        new Chart(document.getElementById('parallelChart'), {{
            type: 'bar',
            data: {{
                labels: {par_labels},
                datasets: [
                    {{
                        label: 'Sequential (single-core)',
                        data: {seq_times},
                        backgroundColor: 'rgba(10, 173, 137, 0.8)',
                        borderColor: 'rgba(10, 173, 137, 1)',
                        borderWidth: 1
                    }},
                    {{
                        label: 'Parallel (rayon, multi-core)',
                        data: {par_times},
                        backgroundColor: 'rgba(96, 175, 255, 0.8)',
                        borderColor: 'rgba(33, 150, 243, 1)',
                        borderWidth: 1
                    }}
                ]
            }},
            options: {{
                responsive: true,
                indexAxis: 'y',
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Sequential vs Parallel Rust (ms, lower is better)'
                    }},
                    tooltip: {{
                        callbacks: {{
                            label: function(context) {{
                                return context.dataset.label + ': ' + context.raw.toFixed(2) + 'ms';
                            }}
                        }}
                    }}
                }},
                scales: {{
                    x: {{
                        beginAtZero: true,
                        title: {{
                            display: true,
                            text: 'Average time (ms)'
                        }}
                    }},
                    y: {{
                        title: {{
                            display: true,
                            text: 'Input size'
                        }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)

    print(f"Graph generated: {output_path}")
    print(f"Speedups (Rust vs Python): {speedups}")
    if par_speedups:
        print(f"Parallel speedups: {par_speedups}")
    return speedups


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark comparison graph")
    parser.add_argument("--rust-output", default=None, help="Path to Rust benchmark output")
    parser.add_argument("--python-output", default=None, help="Path to Python benchmark output")
    parser.add_argument("--parallel-output", default=None, help="Path to parallel benchmark output")
    parser.add_argument("--output", default="benchmark_results.html", help="Output HTML file")
    args = parser.parse_args()

    rust_results = []
    if args.rust_output and Path(args.rust_output).exists():
        rust_results = parse_bench_output(args.rust_output)

    python_results = []
    if args.python_output and Path(args.python_output).exists():
        python_results = parse_bench_output(args.python_output)

    parallel_results = []
    if args.parallel_output and Path(args.parallel_output).exists():
        parallel_results = parse_bench_output(args.parallel_output)

    if not rust_results:
        print("No Rust results found! Run cross_lang_bench first.")
        sys.exit(1)

    output_path = args.output
    generate_html(rust_results, python_results, parallel_results, output_path)


if __name__ == "__main__":
    main()
