/// Aggregation strategy for federated learning rounds.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[allow(clippy::enum_variant_names)] // Public API — variant names mirror FL literature.
pub enum Strategy {
    FedAvg,
    FedProx { mu: f64 },
    FedMedian,
}

/// A model update from a single client, represented as named weight tensors.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ClientUpdate {
    /// Number of training samples used to produce this update.
    pub num_samples: usize,
    /// Flattened parameter arrays keyed by layer name.
    pub weights: std::collections::HashMap<String, Vec<f32>>,
}

/// Aggregate a list of client updates into a global model using the chosen strategy.
///
/// Accepts any slice element that borrows a `ClientUpdate` — so `&[ClientUpdate]`
/// (owned) and `&[&ClientUpdate]` (refs across the PyO3 boundary) both compile
/// without the caller cloning weight data.
pub fn aggregate<U: std::borrow::Borrow<ClientUpdate>>(
    updates: &[U],
    strategy: &Strategy,
) -> Result<std::collections::HashMap<String, Vec<f32>>, String> {
    if updates.is_empty() {
        return Err("No client updates to aggregate".to_string());
    }

    match strategy {
        Strategy::FedAvg | Strategy::FedProx { .. } => fedavg(updates),
        Strategy::FedMedian => fed_median(updates),
    }
}

/// Weighted average of client weights, weighted by the number of local training samples.
///
/// Uses an f64 accumulator to bound rounding error when summing many small f32
/// contributions, then downcasts to f32 at the end.
fn fedavg<U: std::borrow::Borrow<ClientUpdate>>(
    updates: &[U],
) -> Result<std::collections::HashMap<String, Vec<f32>>, String> {
    let total_samples: usize = updates.iter().map(|u| u.borrow().num_samples).sum();
    if total_samples == 0 {
        return Err("Total sample count is zero".to_string());
    }
    let total_samples_f64 = total_samples as f64;

    let first = updates[0].borrow();
    let mut global: std::collections::HashMap<String, Vec<f32>> =
        std::collections::HashMap::with_capacity(first.weights.len());

    for (name, first_vec) in first.weights.iter() {
        let len = first_vec.len();
        let mut agg = vec![0.0f64; len];

        for update in updates {
            let w = match update.borrow().weights.get(name) {
                Some(v) => v,
                None => return Err(format!("Client update missing layer '{}'", name)),
            };
            if w.len() != len {
                return Err(format!(
                    "Layer '{}' size mismatch: expected {}, got {}",
                    name,
                    len,
                    w.len()
                ));
            }
            let scale = update.borrow().num_samples as f64 / total_samples_f64;
            for (a, &val) in agg.iter_mut().zip(w.iter()) {
                *a += val as f64 * scale;
            }
        }

        global.insert(name.clone(), agg.into_iter().map(|x| x as f32).collect());
    }

    Ok(global)
}

/// Element-wise coordinate-wise median across client updates.
///
/// Hoists the per-layer client slices out of the inner coordinate loop so the
/// hot path touches only contiguous `&[f32]` slices — no HashMap lookups per
/// coordinate. Uses `select_nth_unstable_by` (O(C) on average) instead of a
/// full sort (O(C log C)) and reuses a single scratch buffer per layer.
fn fed_median<U: std::borrow::Borrow<ClientUpdate>>(
    updates: &[U],
) -> Result<std::collections::HashMap<String, Vec<f32>>, String> {
    let first = updates[0].borrow();
    let mut global: std::collections::HashMap<String, Vec<f32>> =
        std::collections::HashMap::with_capacity(first.weights.len());

    let mut scratch: Vec<f32> = Vec::with_capacity(updates.len());

    for (name, first_vec) in first.weights.iter() {
        let len = first_vec.len();

        let layer_slices: Vec<&[f32]> = updates
            .iter()
            .map(|u| {
                u.borrow()
                    .weights
                    .get(name)
                    .map(|v| v.as_slice())
                    .ok_or_else(|| format!("Client update missing layer '{}'", name))
            })
            .collect::<Result<_, _>>()?;
        for s in &layer_slices {
            if s.len() != len {
                return Err(format!(
                    "Layer '{}' size mismatch: expected {}, got {}",
                    name,
                    len,
                    s.len()
                ));
            }
        }

        let mut agg = vec![0.0f32; len];
        let n = layer_slices.len();
        let mid = n / 2;
        let even = n.is_multiple_of(2);

        let cmp = |a: &f32, b: &f32| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal);

        for (i, slot) in agg.iter_mut().enumerate() {
            scratch.clear();
            scratch.extend(layer_slices.iter().map(|s| s[i]));

            scratch.select_nth_unstable_by(mid, cmp);
            // After select_nth, scratch[mid] is the k-th order statistic and
            // scratch[..mid] holds the `mid` smallest elements (unordered).
            let median_hi = scratch[mid];
            *slot = if even {
                // even ⇒ n ≥ 2 ⇒ mid ≥ 1, so the left partition is non-empty.
                let lo = scratch[..mid]
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max);
                (lo + median_hi) / 2.0
            } else {
                median_hi
            };
        }

        global.insert(name.clone(), agg);
    }

    Ok(global)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_update(num_samples: usize, weights: &[(&str, Vec<f32>)]) -> ClientUpdate {
        ClientUpdate {
            num_samples,
            weights: weights
                .iter()
                .map(|(k, v)| (k.to_string(), v.clone()))
                .collect(),
        }
    }

    #[test]
    fn fedavg_single_client() {
        let update = make_update(10, &[("layer1", vec![1.0, 2.0, 3.0])]);
        let result = aggregate(&[update], &Strategy::FedAvg).unwrap();
        let layer = &result["layer1"];
        assert!((layer[0] - 1.0).abs() < 1e-5);
        assert!((layer[1] - 2.0).abs() < 1e-5);
        assert!((layer[2] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn fedavg_equal_weights() {
        let u1 = make_update(5, &[("w", vec![0.0, 2.0])]);
        let u2 = make_update(5, &[("w", vec![2.0, 4.0])]);
        let result = aggregate(&[u1, u2], &Strategy::FedAvg).unwrap();
        let w = &result["w"];
        assert!((w[0] - 1.0).abs() < 1e-5);
        assert!((w[1] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn fedavg_weighted_by_samples() {
        // u1 has 3x the samples of u2, so result should be 3/4 * u1 + 1/4 * u2
        let u1 = make_update(3, &[("w", vec![0.0])]);
        let u2 = make_update(1, &[("w", vec![4.0])]);
        let result = aggregate(&[u1, u2], &Strategy::FedAvg).unwrap();
        let w = &result["w"];
        assert!((w[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn fedmedian_odd_clients() {
        let u1 = make_update(1, &[("w", vec![1.0])]);
        let u2 = make_update(1, &[("w", vec![3.0])]);
        let u3 = make_update(1, &[("w", vec![2.0])]);
        let result = aggregate(&[u1, u2, u3], &Strategy::FedMedian).unwrap();
        assert!((result["w"][0] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn fedmedian_even_clients_averages_middle_pair() {
        // Sorted: [1, 2, 3, 4] → median = (2 + 3) / 2 = 2.5
        let u1 = make_update(1, &[("w", vec![3.0, 10.0])]);
        let u2 = make_update(1, &[("w", vec![1.0, 40.0])]);
        let u3 = make_update(1, &[("w", vec![4.0, 20.0])]);
        let u4 = make_update(1, &[("w", vec![2.0, 30.0])]);
        let result = aggregate(&[u1, u2, u3, u4], &Strategy::FedMedian).unwrap();
        let w = &result["w"];
        assert!((w[0] - 2.5).abs() < 1e-5, "got {}", w[0]);
        assert!((w[1] - 25.0).abs() < 1e-5, "got {}", w[1]);
    }

    #[test]
    fn fedmedian_missing_layer_returns_error() {
        let u1 = make_update(1, &[("a", vec![1.0])]);
        let u2 = make_update(1, &[("b", vec![1.0])]);
        let result = aggregate(&[u1, u2], &Strategy::FedMedian);
        assert!(result.is_err());
    }

    #[test]
    fn aggregate_empty_returns_error() {
        let empty: [ClientUpdate; 0] = [];
        let result = aggregate(&empty, &Strategy::FedAvg);
        assert!(result.is_err());
    }

    #[test]
    fn aggregate_missing_layer_returns_error() {
        let u1 = make_update(1, &[("a", vec![1.0])]);
        let u2 = make_update(1, &[("b", vec![1.0])]);
        // u2 is missing "a", aggregation should fail
        let result = aggregate(&[u1, u2], &Strategy::FedAvg);
        assert!(result.is_err());
    }
}
