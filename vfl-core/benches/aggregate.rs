//! Macro-benchmarks for the aggregation hot path.
//!
//! One bench per (strategy, shape tier). Shapes mirror realistic FL model
//! sizes so the numbers mean something:
//!   - `tiny`   ≈ 1K params, 4 layers  (smoke / CI baseline)
//!   - `medium` ≈ 1M params, 10 layers (ResNet-18-scale)
//!   - `large`  ≈ 10M params, 16 layers (ResNet-50-scale)
//!
//! All tiers use 10 clients. Weights are deterministic pseudo-random f32s
//! so reruns on the same tier are comparable. We measure aggregation only —
//! no Python, no PyO3 — so this is the best case for the Rust core.

use std::collections::HashMap;

use divan::{black_box, Bencher};
use vfl_core::strategy::{aggregate, ClientUpdate, Strategy};

fn main() {
    divan::main();
}

const CLIENTS: usize = 10;
const TIERS: &[&str] = &["tiny", "medium", "large"];

/// Deterministic PRNG (xorshift32) — avoids pulling a noise source into the
/// bench path and keeps each tier reproducible run-to-run.
fn make_updates(tier: &str) -> Vec<ClientUpdate> {
    let layers: Vec<(&str, usize)> = match tier {
        "tiny" => vec![
            ("fc1.weight", 512),
            ("fc1.bias", 64),
            ("fc2.weight", 384),
            ("fc2.bias", 10),
        ],
        "medium" => {
            // ~1M params across 10 layers
            (0..10)
                .map(|i| {
                    let name: &'static str = Box::leak(format!("layer{i}.weight").into_boxed_str());
                    (name, 100_000)
                })
                .collect()
        }
        "large" => {
            // ~10M params across 16 layers
            (0..16)
                .map(|i| {
                    let name: &'static str = Box::leak(format!("block{i}.weight").into_boxed_str());
                    (name, 625_000)
                })
                .collect()
        }
        _ => panic!("unknown tier: {tier}"),
    };

    let mut state: u32 = 0x9E37_79B9;
    let mut next = || -> f32 {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        (state as f32 / u32::MAX as f32) * 2.0 - 1.0
    };

    (0..CLIENTS)
        .map(|_| {
            let weights: HashMap<String, Vec<f32>> = layers
                .iter()
                .map(|(name, size)| {
                    let v: Vec<f32> = (0..*size).map(|_| next() * 0.1).collect();
                    (name.to_string(), v)
                })
                .collect();
            ClientUpdate {
                num_samples: 100,
                weights,
            }
        })
        .collect()
}

#[divan::bench(args = TIERS)]
fn fed_avg(bencher: Bencher, tier: &str) {
    let updates = make_updates(tier);
    bencher.bench_local(|| aggregate(black_box(&updates), &Strategy::FedAvg).unwrap());
}

#[divan::bench(args = TIERS)]
fn fed_prox(bencher: Bencher, tier: &str) {
    let updates = make_updates(tier);
    bencher
        .bench_local(|| aggregate(black_box(&updates), &Strategy::FedProx { mu: 0.01 }).unwrap());
}

#[divan::bench(args = TIERS)]
fn fed_median(bencher: Bencher, tier: &str) {
    let updates = make_updates(tier);
    bencher.bench_local(|| aggregate(black_box(&updates), &Strategy::FedMedian).unwrap());
}

#[divan::bench(args = TIERS)]
fn trimmed_mean(bencher: Bencher, tier: &str) {
    let updates = make_updates(tier);
    bencher
        .bench_local(|| aggregate(black_box(&updates), &Strategy::TrimmedMean { k: 1 }).unwrap());
}

// f = 1 keeps Krum within `n >= 2f + 3 = 5` at 10 clients and matches the
// conservative "one attacker tolerated" scenario in docs/strategies.md.
#[divan::bench(args = TIERS)]
fn krum(bencher: Bencher, tier: &str) {
    let updates = make_updates(tier);
    bencher.bench_local(|| aggregate(black_box(&updates), &Strategy::Krum { f: 1 }).unwrap());
}

#[divan::bench(args = TIERS)]
fn multi_krum(bencher: Bencher, tier: &str) {
    let updates = make_updates(tier);
    bencher.bench_local(|| {
        aggregate(black_box(&updates), &Strategy::MultiKrum { f: 1, m: None }).unwrap()
    });
}

// n=10, f=1 satisfies Bulyan's `n >= 4f + 3 = 7` bound. Default m = n - 2f = 8.
#[divan::bench(args = TIERS)]
fn bulyan(bencher: Bencher, tier: &str) {
    let updates = make_updates(tier);
    bencher.bench_local(|| {
        aggregate(black_box(&updates), &Strategy::Bulyan { f: 1, m: None }).unwrap()
    });
}
