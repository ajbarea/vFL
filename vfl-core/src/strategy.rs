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
/// Returns the aggregated weights or an error string if aggregation fails.
pub fn aggregate(
    updates: &[ClientUpdate],
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
fn fedavg(updates: &[ClientUpdate]) -> Result<std::collections::HashMap<String, Vec<f32>>, String> {
    let total_samples: usize = updates.iter().map(|u| u.num_samples).sum();
    if total_samples == 0 {
        return Err("Total sample count is zero".to_string());
    }

    let layer_names: Vec<String> = updates[0].weights.keys().cloned().collect();
    let mut global: std::collections::HashMap<String, Vec<f32>> = std::collections::HashMap::new();

    for name in &layer_names {
        let len = updates[0].weights.get(name).map(|v| v.len()).unwrap_or(0);

        let mut agg = vec![0.0f32; len];

        for update in updates {
            let w = match update.weights.get(name) {
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
            let scale = update.num_samples as f32 / total_samples as f32;
            for (a, &val) in agg.iter_mut().zip(w.iter()) {
                *a += val * scale;
            }
        }

        global.insert(name.clone(), agg);
    }

    Ok(global)
}

/// Element-wise coordinate-wise median across client updates.
fn fed_median(
    updates: &[ClientUpdate],
) -> Result<std::collections::HashMap<String, Vec<f32>>, String> {
    let layer_names: Vec<String> = updates[0].weights.keys().cloned().collect();
    let mut global: std::collections::HashMap<String, Vec<f32>> = std::collections::HashMap::new();

    for name in &layer_names {
        let len = updates[0].weights.get(name).map(|v| v.len()).unwrap_or(0);

        let mut agg = vec![0.0f32; len];

        for (i, slot) in agg.iter_mut().enumerate() {
            let mut values: Vec<f32> = updates
                .iter()
                .filter_map(|u| u.weights.get(name).and_then(|v| v.get(i)).copied())
                .collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mid = values.len() / 2;
            *slot = if values.len().is_multiple_of(2) {
                (values[mid - 1] + values[mid]) / 2.0
            } else {
                values[mid]
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
    fn aggregate_empty_returns_error() {
        let result = aggregate(&[], &Strategy::FedAvg);
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
