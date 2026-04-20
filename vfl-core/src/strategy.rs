use std::borrow::Borrow;
use std::collections::HashMap;

/// Aggregation strategy for federated learning rounds.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[allow(clippy::enum_variant_names)] // Public API — variant names mirror FL literature.
pub enum Strategy {
    FedAvg,
    FedProx {
        mu: f64,
    },
    FedMedian,
    /// Krum (Blanchard et al. 2017, arXiv:1703.02757) — picks the single
    /// client whose sum of `n - f - 2` smallest squared distances to others
    /// is minimal. Byzantine-robust when `n >= 2*f + 3`.
    Krum {
        f: usize,
    },
    /// Multi-Krum (El Mhamdi et al. 2018) — averages the top-`m` clients by
    /// Krum score. `m = None` resolves to `n - f` ("largest non-Byzantine
    /// group") at aggregation time. Requires `n >= 2*f + 3` and
    /// `1 <= m <= n - f`.
    MultiKrum {
        f: usize,
        m: Option<usize>,
    },
}

/// A model update from a single client, represented as named weight tensors.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ClientUpdate {
    /// Number of training samples used to produce this update.
    pub num_samples: usize,
    /// Flattened parameter arrays keyed by layer name.
    pub weights: HashMap<String, Vec<f32>>,
}

/// Result of aggregating a round of client updates.
///
/// `selected_client_ids` is always populated: non-robust aggregators (FedAvg,
/// FedProx, FedMedian) return `0..n`; Krum/Multi-Krum return the subset that
/// contributed to the aggregate (1 client for Krum, `m` for Multi-Krum).
#[derive(Debug, Clone)]
pub struct Aggregation {
    pub weights: HashMap<String, Vec<f32>>,
    pub selected_client_ids: Vec<usize>,
}

/// Aggregate a list of client updates into a global model using the chosen strategy.
///
/// Accepts any slice element that borrows a `ClientUpdate` — so `&[ClientUpdate]`
/// (owned) and `&[&ClientUpdate]` (refs across the PyO3 boundary) both compile
/// without the caller cloning weight data.
pub fn aggregate<U: Borrow<ClientUpdate>>(
    updates: &[U],
    strategy: &Strategy,
) -> Result<Aggregation, String> {
    if updates.is_empty() {
        return Err("No client updates to aggregate".to_string());
    }

    let all_ids = || (0..updates.len()).collect::<Vec<usize>>();

    match strategy {
        Strategy::FedAvg | Strategy::FedProx { .. } => Ok(Aggregation {
            weights: fedavg(updates)?,
            selected_client_ids: all_ids(),
        }),
        Strategy::FedMedian => Ok(Aggregation {
            weights: fed_median(updates)?,
            selected_client_ids: all_ids(),
        }),
        Strategy::Krum { f } => krum_select(updates, *f, Some(1)),
        Strategy::MultiKrum { f, m } => krum_select(updates, *f, *m),
    }
}

/// Weighted average of client weights, weighted by the number of local training samples.
///
/// Uses an f64 accumulator to bound rounding error when summing many small f32
/// contributions, then downcasts to f32 at the end.
fn fedavg<U: Borrow<ClientUpdate>>(updates: &[U]) -> Result<HashMap<String, Vec<f32>>, String> {
    let total_samples: usize = updates.iter().map(|u| u.borrow().num_samples).sum();
    if total_samples == 0 {
        return Err("Total sample count is zero".to_string());
    }
    let total_samples_f64 = total_samples as f64;

    let first = updates[0].borrow();
    let mut global: HashMap<String, Vec<f32>> = HashMap::with_capacity(first.weights.len());

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
fn fed_median<U: Borrow<ClientUpdate>>(updates: &[U]) -> Result<HashMap<String, Vec<f32>>, String> {
    let first = updates[0].borrow();
    let mut global: HashMap<String, Vec<f32>> = HashMap::with_capacity(first.weights.len());

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

/// Krum / Multi-Krum kernel.
///
/// Flattens each client's weights into a shared coordinate space (layer order
/// taken from `updates[0]`), computes the pairwise squared-Euclidean distance
/// matrix, scores each client by the sum of its `n - f - 2` smallest distances
/// to other clients, then selects the `m` smallest-score clients and averages
/// them with uniform weights (not sample-weighted — matches El Mhamdi et al.).
///
/// `m = Some(1)` reproduces Blanchard's original Krum. `m = None` resolves to
/// `n - f` at call time ("largest non-Byzantine group" interpretation).
fn krum_select<U: Borrow<ClientUpdate>>(
    updates: &[U],
    f: usize,
    m: Option<usize>,
) -> Result<Aggregation, String> {
    let n = updates.len();

    // Blanchard's breakdown bound: needs n - f - 2 >= 1 honest neighbours to
    // score against, and f Byzantine + (f+1) "too far" exclusions + 1 winner.
    if n < 2 * f + 3 {
        return Err(format!(
            "Krum/Multi-Krum requires n >= 2*f + 3; got n={n}, f={f}"
        ));
    }

    let m = m.unwrap_or(n - f);
    if m == 0 || m > n - f {
        return Err(format!(
            "Multi-Krum requires 1 <= m <= n - f; got m={m}, n={n}, f={f}"
        ));
    }

    // Fix a layer order from the first update; every other client must match.
    let first = updates[0].borrow();
    let layer_names: Vec<String> = first.weights.keys().cloned().collect();
    let layer_sizes: Vec<usize> = layer_names
        .iter()
        .map(|name| first.weights[name].len())
        .collect();
    let flat_len: usize = layer_sizes.iter().sum();

    // Flatten every client into a contiguous f32 vector in `layer_names` order.
    let mut flats: Vec<Vec<f32>> = Vec::with_capacity(n);
    for u in updates {
        let u = u.borrow();
        let mut flat: Vec<f32> = Vec::with_capacity(flat_len);
        for (name, &size) in layer_names.iter().zip(&layer_sizes) {
            let w = u
                .weights
                .get(name)
                .ok_or_else(|| format!("Client update missing layer '{}'", name))?;
            if w.len() != size {
                return Err(format!(
                    "Layer '{}' size mismatch: expected {}, got {}",
                    name,
                    size,
                    w.len()
                ));
            }
            flat.extend_from_slice(w);
        }
        flats.push(flat);
    }

    // Pairwise squared Euclidean distances (symmetric; diagonal = 0). f64
    // accumulator bounds rounding when d is large (e.g. 10⁶ coords × f32 diffs).
    let mut dist = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d: f64 = flats[i]
                .iter()
                .zip(flats[j].iter())
                .map(|(&a, &b)| {
                    let diff = a as f64 - b as f64;
                    diff * diff
                })
                .sum();
            dist[i][j] = d;
            dist[j][i] = d;
        }
    }

    // Krum score: sum of the `n - f - 2` smallest distances to *other* clients.
    // The `n >= 2f + 3` check above guarantees `k_terms >= f + 1 >= 1`.
    let k_terms = n - f - 2;
    let mut scores: Vec<(usize, f64)> = Vec::with_capacity(n);
    let mut buf: Vec<f64> = Vec::with_capacity(n.saturating_sub(1));
    let cmp_f64 = |a: &f64, b: &f64| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal);
    for (i, row) in dist.iter().enumerate() {
        buf.clear();
        for (j, &d) in row.iter().enumerate() {
            if j != i {
                buf.push(d);
            }
        }
        buf.select_nth_unstable_by(k_terms - 1, cmp_f64);
        let score: f64 = buf[..k_terms].iter().sum();
        scores.push((i, score));
    }

    // Ascending sort — smallest score is the Krum winner. Stable on ties so
    // equal-score clients are selected in index order (makes tests deterministic).
    scores.sort_by(|a, b| cmp_f64(&a.1, &b.1));
    let selected_client_ids: Vec<usize> = scores[..m].iter().map(|&(i, _)| i).collect();

    // Uniform average over the selected clients, layer by layer.
    let mut weights: HashMap<String, Vec<f32>> = HashMap::with_capacity(layer_names.len());
    let inv_m = 1.0f64 / m as f64;
    for (name, &size) in layer_names.iter().zip(&layer_sizes) {
        let mut agg = vec![0.0f64; size];
        for &idx in &selected_client_ids {
            let u = updates[idx].borrow();
            let w = &u.weights[name];
            for (a, &val) in agg.iter_mut().zip(w.iter()) {
                *a += val as f64 * inv_m;
            }
        }
        weights.insert(name.clone(), agg.into_iter().map(|x| x as f32).collect());
    }

    Ok(Aggregation {
        weights,
        selected_client_ids,
    })
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
        let layer = &result.weights["layer1"];
        assert!((layer[0] - 1.0).abs() < 1e-5);
        assert!((layer[1] - 2.0).abs() < 1e-5);
        assert!((layer[2] - 3.0).abs() < 1e-5);
        assert_eq!(result.selected_client_ids, vec![0]);
    }

    #[test]
    fn fedavg_equal_weights() {
        let u1 = make_update(5, &[("w", vec![0.0, 2.0])]);
        let u2 = make_update(5, &[("w", vec![2.0, 4.0])]);
        let result = aggregate(&[u1, u2], &Strategy::FedAvg).unwrap();
        let w = &result.weights["w"];
        assert!((w[0] - 1.0).abs() < 1e-5);
        assert!((w[1] - 3.0).abs() < 1e-5);
        assert_eq!(result.selected_client_ids, vec![0, 1]);
    }

    #[test]
    fn fedavg_weighted_by_samples() {
        let u1 = make_update(3, &[("w", vec![0.0])]);
        let u2 = make_update(1, &[("w", vec![4.0])]);
        let result = aggregate(&[u1, u2], &Strategy::FedAvg).unwrap();
        let w = &result.weights["w"];
        assert!((w[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn fedmedian_odd_clients() {
        let u1 = make_update(1, &[("w", vec![1.0])]);
        let u2 = make_update(1, &[("w", vec![3.0])]);
        let u3 = make_update(1, &[("w", vec![2.0])]);
        let result = aggregate(&[u1, u2, u3], &Strategy::FedMedian).unwrap();
        assert!((result.weights["w"][0] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn fedmedian_even_clients_averages_middle_pair() {
        // Sorted: [1, 2, 3, 4] → median = (2 + 3) / 2 = 2.5
        let u1 = make_update(1, &[("w", vec![3.0, 10.0])]);
        let u2 = make_update(1, &[("w", vec![1.0, 40.0])]);
        let u3 = make_update(1, &[("w", vec![4.0, 20.0])]);
        let u4 = make_update(1, &[("w", vec![2.0, 30.0])]);
        let result = aggregate(&[u1, u2, u3, u4], &Strategy::FedMedian).unwrap();
        let w = &result.weights["w"];
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
        let result = aggregate(&[u1, u2], &Strategy::FedAvg);
        assert!(result.is_err());
    }

    // ----- Krum / Multi-Krum -----

    #[test]
    fn krum_3client_fixture_picks_closest_pair() {
        // n=3, f=0, k_terms = n-f-2 = 1. Each client scores with its single
        // closest neighbour.
        //   c0=[0,0], c1=[0,1], c2=[10,10]
        //   d(0,1)=1, d(0,2)=200, d(1,2)=(10-0)^2+(10-1)^2=100+81=181
        //   score(0)=min(1,200)=1, score(1)=min(1,181)=1, score(2)=min(200,181)=181
        // Tie between 0 and 1; stable sort keeps index order → Krum picks c0.
        let u0 = make_update(1, &[("w", vec![0.0, 0.0])]);
        let u1 = make_update(1, &[("w", vec![0.0, 1.0])]);
        let u2 = make_update(1, &[("w", vec![10.0, 10.0])]);
        let result = aggregate(&[u0, u1, u2], &Strategy::Krum { f: 0 }).unwrap();
        assert_eq!(result.selected_client_ids, vec![0]);
        let w = &result.weights["w"];
        assert!((w[0] - 0.0).abs() < 1e-6, "got {}", w[0]);
        assert!((w[1] - 0.0).abs() < 1e-6, "got {}", w[1]);
    }

    #[test]
    fn krum_equals_multikrum_m1() {
        let u0 = make_update(1, &[("w", vec![1.0, 1.0])]);
        let u1 = make_update(1, &[("w", vec![1.1, 1.1])]);
        let u2 = make_update(1, &[("w", vec![5.0, 5.0])]);
        let updates = [u0, u1, u2];
        let krum = aggregate(&updates, &Strategy::Krum { f: 0 }).unwrap();
        let mk = aggregate(&updates, &Strategy::MultiKrum { f: 0, m: Some(1) }).unwrap();
        assert_eq!(krum.selected_client_ids, mk.selected_client_ids);
        let a = &krum.weights["w"];
        let b = &mk.weights["w"];
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-6);
        }
    }

    #[test]
    fn krum_excludes_byzantine_outlier() {
        // n=5, f=1 → needs n >= 2*1+3 = 5 ✓. k_terms = 5-1-2 = 2.
        // 4 honest clients around [1,1], 1 outlier at [100,100].
        let mut clients: Vec<ClientUpdate> = [1.0f32, 1.1, 0.9, 1.0]
            .iter()
            .map(|&v| make_update(1, &[("w", vec![v, v])]))
            .collect();
        clients.push(make_update(1, &[("w", vec![100.0, 100.0])]));
        let result = aggregate(&clients, &Strategy::Krum { f: 1 }).unwrap();
        assert_eq!(result.selected_client_ids.len(), 1);
        assert_ne!(
            result.selected_client_ids[0], 4,
            "Krum picked the Byzantine outlier"
        );
    }

    #[test]
    fn multikrum_default_m_excludes_outlier_and_averages_honest() {
        // n=5, f=1, m=None → m = n-f = 4. All 4 honest clients selected,
        // outlier excluded. Uniform mean of 4 identical vectors is itself.
        let u0 = make_update(1, &[("w", vec![2.0, 2.0])]);
        let u1 = make_update(1, &[("w", vec![2.0, 2.0])]);
        let u2 = make_update(1, &[("w", vec![2.0, 2.0])]);
        let u3 = make_update(1, &[("w", vec![2.0, 2.0])]);
        let u4 = make_update(1, &[("w", vec![100.0, 100.0])]);
        let result = aggregate(
            &[u0, u1, u2, u3, u4],
            &Strategy::MultiKrum { f: 1, m: None },
        )
        .unwrap();
        assert_eq!(result.selected_client_ids.len(), 4);
        assert!(!result.selected_client_ids.contains(&4));
        let w = &result.weights["w"];
        assert!((w[0] - 2.0).abs() < 1e-5, "got {}", w[0]);
        assert!((w[1] - 2.0).abs() < 1e-5, "got {}", w[1]);
    }

    #[test]
    fn multikrum_uniform_weighting_ignores_sample_counts() {
        // If Multi-Krum were sample-weighted, u0 (100 samples) would dominate.
        // Paper says uniform — so the mean is 1.0, not 0.02.
        let u0 = make_update(100, &[("w", vec![0.0])]);
        let u1 = make_update(1, &[("w", vec![1.0])]);
        let u2 = make_update(1, &[("w", vec![2.0])]);
        let result = aggregate(&[u0, u1, u2], &Strategy::MultiKrum { f: 0, m: Some(3) }).unwrap();
        // Uniform mean: (0 + 1 + 2) / 3 = 1.0
        assert!((result.weights["w"][0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn krum_rejects_insufficient_clients() {
        // n=2, f=0 → needs n >= 3. Should error.
        let u0 = make_update(1, &[("w", vec![1.0])]);
        let u1 = make_update(1, &[("w", vec![2.0])]);
        let result = aggregate(&[u0, u1], &Strategy::Krum { f: 0 });
        assert!(result.is_err());
    }

    #[test]
    fn multikrum_rejects_bad_m() {
        let u0 = make_update(1, &[("w", vec![1.0])]);
        let u1 = make_update(1, &[("w", vec![2.0])]);
        let u2 = make_update(1, &[("w", vec![3.0])]);
        // n=3, f=0 → m must be in 1..=3.
        assert!(aggregate(
            &[u0.clone(), u1.clone(), u2.clone()],
            &Strategy::MultiKrum { f: 0, m: Some(4) },
        )
        .is_err());
        assert!(aggregate(&[u0, u1, u2], &Strategy::MultiKrum { f: 0, m: Some(0) },).is_err());
    }

    #[test]
    fn krum_missing_layer_returns_error() {
        let u0 = make_update(1, &[("a", vec![1.0]), ("b", vec![1.0])]);
        let u1 = make_update(1, &[("a", vec![1.0]), ("b", vec![1.0])]);
        let u2 = make_update(1, &[("a", vec![1.0])]); // missing "b"
        let result = aggregate(&[u0, u1, u2], &Strategy::Krum { f: 0 });
        assert!(result.is_err());
    }
}
