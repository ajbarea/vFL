//! Property-based tests for the aggregation kernel.
//!
//! Same algebraic invariants as `tests/test_aggregation_properties.py` —
//! verified here on the Rust side so a regression surfaces under both
//! `cargo test` and `pytest`. Running the properties on the native code
//! (no PyO3 in the loop) also lets us crank counts higher than we can
//! afford from Python.
//!
//! Invariants checked:
//!   - FedAvg of N identical updates == that update
//!   - FedAvg of one update == that update
//!   - FedAvg matches a naive weighted-mean reference implementation
//!   - FedMedian of N identical updates == that update
//!   - FedMedian is unmoved by a single extreme outlier (Byzantine majority)
//!   - FedProx produces the same weights as FedAvg (mu is client-side only)

use std::collections::HashMap;

use approx::assert_relative_eq;
use proptest::prelude::*;
// Alias the aggregation enum so it doesn't shadow `proptest::Strategy` (trait).
use vfl_core::strategy::{aggregate, ClientUpdate, Strategy as Agg};

const LAYER_NAMES: &[&str] = &["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"];

/// A single `ClientUpdate` with `n_layers` layers of size in `[1, 8]`.
fn arb_update_with_shapes(shapes: Vec<(String, usize)>) -> impl Strategy<Value = ClientUpdate> {
    let sample_count = 1usize..1000;
    let coord = -10.0f32..10.0;

    let per_layer: Vec<_> = shapes
        .iter()
        .map(|(name, len)| {
            let name = name.clone();
            proptest::collection::vec(coord.clone(), *len..=*len)
                .prop_map(move |vs| (name.clone(), vs))
        })
        .collect();

    (sample_count, per_layer).prop_map(|(ns, pairs)| ClientUpdate {
        num_samples: ns,
        weights: pairs.into_iter().collect::<HashMap<String, Vec<f32>>>(),
    })
}

/// A shared layer shape spec — drawn once, then every client in a round uses it.
fn arb_shapes() -> impl Strategy<Value = Vec<(String, usize)>> {
    let names = proptest::sample::subsequence(LAYER_NAMES.to_vec(), 1..=LAYER_NAMES.len());
    names.prop_flat_map(|names| {
        let layers: Vec<_> = names
            .into_iter()
            .map(|n| (Just(n.to_string()), 1usize..=8usize))
            .collect();
        layers
    })
}

/// N updates sharing one set of layer shapes — the precondition the aggregator assumes.
fn arb_updates(n_clients: usize) -> impl Strategy<Value = Vec<ClientUpdate>> {
    arb_shapes().prop_flat_map(move |shapes| {
        let clients: Vec<_> = (0..n_clients)
            .map(|_| arb_update_with_shapes(shapes.clone()))
            .collect();
        clients
    })
}

// ---------------------------------------------------------------------------
// FedAvg invariants
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { cases: 64, .. ProptestConfig::default() })]

    #[test]
    fn fedavg_singleton_returns_input(update in arb_updates(1)) {
        let result = aggregate(&update, &Agg::FedAvg).unwrap();
        for (name, values) in &update[0].weights {
            let got = &result.weights[name];
            prop_assert_eq!(got.len(), values.len());
            for (g, w) in got.iter().zip(values.iter()) {
                prop_assert!((g - w).abs() < 1e-5, "{}: {} != {}", name, g, w);
            }
        }
    }

    #[test]
    fn fedavg_identical_updates_is_identity(
        n in 2usize..=6,
        template in arb_updates(1)
    ) {
        let copies: Vec<ClientUpdate> = (0..n).map(|_| template[0].clone()).collect();
        let result = aggregate(&copies, &Agg::FedAvg).unwrap();
        for (name, values) in &template[0].weights {
            let got = &result.weights[name];
            for (g, w) in got.iter().zip(values.iter()) {
                prop_assert!((g - w).abs() < 1e-4, "{}: {} != {}", name, g, w);
            }
        }
    }

    #[test]
    fn fedavg_matches_weighted_mean_reference(updates in arb_updates(3)) {
        let total: f64 = updates.iter().map(|u| u.num_samples as f64).sum();
        let result = aggregate(&updates, &Agg::FedAvg).unwrap();
        for (name, layer) in &result.weights {
            for (i, got) in layer.iter().enumerate() {
                let expected: f64 = updates
                    .iter()
                    .map(|u| u.weights[name][i] as f64 * (u.num_samples as f64 / total))
                    .sum();
                assert_relative_eq!(
                    *got as f64,
                    expected,
                    epsilon = 1e-3,
                    max_relative = 1e-3,
                );
            }
        }
    }

    #[test]
    fn fedavg_preserves_layer_shape(updates in arb_updates(3)) {
        let result = aggregate(&updates, &Agg::FedAvg).unwrap();
        prop_assert_eq!(
            result.weights.keys().collect::<std::collections::BTreeSet<_>>(),
            updates[0].weights.keys().collect::<std::collections::BTreeSet<_>>()
        );
        for (name, values) in &updates[0].weights {
            prop_assert_eq!(result.weights[name].len(), values.len());
        }
    }
}

// ---------------------------------------------------------------------------
// FedMedian invariants
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { cases: 64, .. ProptestConfig::default() })]

    #[test]
    fn fedmedian_singleton_returns_input(update in arb_updates(1)) {
        let result = aggregate(&update, &Agg::FedMedian).unwrap();
        for (name, values) in &update[0].weights {
            let got = &result.weights[name];
            for (g, w) in got.iter().zip(values.iter()) {
                prop_assert!((g - w).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn fedmedian_identical_updates_is_identity(
        n in 2usize..=5,
        template in arb_updates(1)
    ) {
        let copies: Vec<ClientUpdate> = (0..n).map(|_| template[0].clone()).collect();
        let result = aggregate(&copies, &Agg::FedMedian).unwrap();
        for (name, values) in &template[0].weights {
            let got = &result.weights[name];
            for (g, w) in got.iter().zip(values.iter()) {
                prop_assert!((g - w).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn fedmedian_resists_one_extreme_outlier(template in arb_updates(1)) {
        // 4 honest clients + 1 attacker with +100 in every coordinate.
        // FedMedian majority > N/2 ⇒ attacker cannot move the median.
        let base = &template[0];
        let mut round: Vec<ClientUpdate> = (0..4).map(|_| base.clone()).collect();
        let attacker_weights: HashMap<String, Vec<f32>> = base
            .weights
            .iter()
            .map(|(n, vs)| (n.clone(), vs.iter().map(|v| v + 100.0).collect()))
            .collect();
        round.push(ClientUpdate {
            num_samples: base.num_samples,
            weights: attacker_weights,
        });

        let result = aggregate(&round, &Agg::FedMedian).unwrap();
        for (name, values) in &base.weights {
            for (g, w) in result.weights[name].iter().zip(values.iter()) {
                prop_assert!(
                    (g - w).abs() < 1e-4,
                    "median moved under 1 outlier: {} vs {}",
                    g,
                    w
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// FedProx — same kernel as FedAvg
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { cases: 64, .. ProptestConfig::default() })]

    #[test]
    fn fedprox_matches_fedavg_output(updates in arb_updates(3)) {
        let fedavg = aggregate(&updates, &Agg::FedAvg).unwrap();
        let fedprox = aggregate(&updates, &Agg::FedProx { mu: 0.01 }).unwrap();
        for name in fedavg.weights.keys() {
            for (a, p) in fedavg.weights[name].iter().zip(fedprox.weights[name].iter()) {
                prop_assert!((a - p).abs() < 1e-6);
            }
        }
    }
}
