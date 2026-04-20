//! vfl-core: Rust core for VelocityFL.
//!
//! FL orchestration, aggregation strategies, and attack simulation,
//! exposed to Python via PyO3 bindings.

mod orchestrator;
mod security;
pub mod strategy;

use std::collections::HashMap;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// Strategy
// ---------------------------------------------------------------------------

/// Python-facing FL aggregation strategy.
#[pyclass(name = "Strategy")]
#[derive(Debug, Clone)]
pub struct PyStrategy(strategy::Strategy);

#[pymethods]
impl PyStrategy {
    /// Federated Averaging — weighted mean by number of local samples.
    #[staticmethod]
    fn fed_avg() -> Self {
        PyStrategy(strategy::Strategy::FedAvg)
    }

    /// FedProx — FedAvg with a proximal term (mu controls regularisation).
    #[staticmethod]
    fn fed_prox(mu: f64) -> Self {
        PyStrategy(strategy::Strategy::FedProx { mu })
    }

    /// Coordinate-wise median — robust to Byzantine clients.
    #[staticmethod]
    fn fed_median() -> Self {
        PyStrategy(strategy::Strategy::FedMedian)
    }

    /// Krum (Blanchard et al. 2017) — Byzantine-robust selection of a single
    /// client. ``f`` is the tolerated number of Byzantine clients; requires
    /// ``n >= 2*f + 3`` at aggregation time.
    #[staticmethod]
    fn krum(f: usize) -> Self {
        PyStrategy(strategy::Strategy::Krum { f })
    }

    /// Multi-Krum (El Mhamdi et al. 2018) — averages the top-``m`` clients by
    /// Krum score. ``m=None`` resolves to ``n - f`` at aggregation time.
    /// Requires ``n >= 2*f + 3`` and ``1 <= m <= n - f``.
    #[staticmethod]
    #[pyo3(signature = (f, m=None))]
    fn multi_krum(f: usize, m: Option<usize>) -> Self {
        PyStrategy(strategy::Strategy::MultiKrum { f, m })
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}

// ---------------------------------------------------------------------------
// ClientUpdate
// ---------------------------------------------------------------------------

/// A model update submitted by a single federated client.
#[pyclass(name = "ClientUpdate")]
#[derive(Debug, Clone)]
pub struct PyClientUpdate(strategy::ClientUpdate);

#[pymethods]
impl PyClientUpdate {
    /// Create a new client update.
    ///
    /// Args:
    ///     num_samples: Number of local training samples.
    ///     weights: Dict mapping layer names to flat lists of float32 weights.
    #[new]
    fn new(num_samples: usize, weights: HashMap<String, Vec<f32>>) -> Self {
        PyClientUpdate(strategy::ClientUpdate {
            num_samples,
            weights,
        })
    }

    #[getter]
    fn num_samples(&self) -> usize {
        self.0.num_samples
    }

    #[getter]
    fn weights(&self) -> HashMap<String, Vec<f32>> {
        self.0.weights.clone()
    }
}

// ---------------------------------------------------------------------------
// RoundSummary
// ---------------------------------------------------------------------------

/// Summary of a completed federated learning round.
#[pyclass(name = "RoundSummary")]
#[derive(Debug, Clone)]
pub struct PyRoundSummary(orchestrator::RoundSummary);

#[pymethods]
impl PyRoundSummary {
    #[getter]
    fn round(&self) -> usize {
        self.0.round
    }

    #[getter]
    fn num_clients(&self) -> usize {
        self.0.num_clients
    }

    #[getter]
    fn global_loss(&self) -> f64 {
        self.0.global_loss
    }

    /// JSON-serialised list of attack results for this round.
    #[getter]
    fn attack_results(&self) -> PyResult<String> {
        serde_json::to_string(&self.0.attack_results)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Indices of clients that contributed to this round's aggregate.
    ///
    /// For non-robust aggregators (FedAvg / FedProx / FedMedian) this is
    /// every participating client. Krum returns a single index; Multi-Krum
    /// returns ``m`` indices. Always populated — never ``None``.
    #[getter]
    fn selected_client_ids(&self) -> Vec<usize> {
        self.0.selected_client_ids.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "RoundSummary(round={}, num_clients={}, global_loss={:.4})",
            self.0.round, self.0.num_clients, self.0.global_loss
        )
    }
}

// ---------------------------------------------------------------------------
// Orchestrator
// ---------------------------------------------------------------------------

/// Rust-backed FL orchestrator.
///
/// Manages training rounds, client aggregation, and attack simulations.
#[pyclass(name = "Orchestrator")]
pub struct PyOrchestrator(orchestrator::Orchestrator);

#[pymethods]
impl PyOrchestrator {
    /// Create a new orchestrator.
    ///
    /// Args:
    ///     model_id: Hugging Face model identifier (e.g. "meta-llama/Llama-3-8B").
    ///     dataset: Dataset name or path.
    ///     strategy: Aggregation strategy (use :class:`Strategy` helpers).
    ///     storage: Storage URI (e.g. "hf-xet://namespace/repo").
    ///     min_clients: Minimum clients required to proceed with a round.
    ///     rounds: Number of federated learning rounds to run.
    ///     layer_shapes: Dict mapping layer names to their flat parameter count.
    #[new]
    #[pyo3(signature = (
        model_id,
        dataset,
        strategy,
        storage,
        min_clients,
        rounds,
        layer_shapes
    ))]
    fn new(
        model_id: String,
        dataset: String,
        strategy: &PyStrategy,
        storage: String,
        min_clients: usize,
        rounds: usize,
        layer_shapes: HashMap<String, usize>,
    ) -> Self {
        let config = orchestrator::ExperimentConfig {
            model_id,
            dataset,
            strategy: strategy.0.clone(),
            storage,
            min_clients,
            rounds,
        };
        PyOrchestrator(orchestrator::Orchestrator::new(config, &layer_shapes))
    }

    /// Register an attack to be simulated in the next round.
    ///
    /// Args:
    ///     attack_type: One of ``"model_poisoning"``, ``"sybil_nodes"``,
    ///                  ``"gaussian_noise"``, ``"label_flipping"``.
    ///     intensity: For ``model_poisoning`` / ``gaussian_noise`` — magnitude ∈ [0, 1].
    ///     count: For ``sybil_nodes`` — number of Byzantine clients to inject.
    ///     fraction: For ``label_flipping`` — fraction of clients to affect.
    #[pyo3(signature = (attack_type, intensity=0.1, count=1, fraction=0.1))]
    fn register_attack(
        &mut self,
        attack_type: &str,
        intensity: f64,
        count: usize,
        fraction: f64,
    ) -> PyResult<()> {
        let attack = match attack_type {
            "model_poisoning" => security::AttackType::ModelPoisoning { intensity },
            "sybil_nodes" => security::AttackType::SybilNodes { count },
            "gaussian_noise" => security::AttackType::GaussianNoise { std_dev: intensity },
            "label_flipping" => security::AttackType::LabelFlipping { fraction },
            other => {
                return Err(PyRuntimeError::new_err(format!(
                    "Unknown attack type: '{other}'. Valid types: model_poisoning, \
                     sybil_nodes, gaussian_noise, label_flipping"
                )))
            }
        };
        self.0.register_attack(attack);
        Ok(())
    }

    /// Execute a single federated learning round.
    ///
    /// Args:
    ///     client_updates: List of :class:`ClientUpdate` objects.
    ///     reported_loss: Loss the caller computed on a held-out set after
    ///         the previous round's aggregation. Stored on the round summary;
    ///         omit (or pass ``None``) to record ``NaN``.
    ///
    /// Returns:
    ///     :class:`RoundSummary` for the completed round.
    #[pyo3(signature = (client_updates, reported_loss=None))]
    fn run_round(
        &mut self,
        client_updates: Vec<PyRef<PyClientUpdate>>,
        reported_loss: Option<f64>,
    ) -> PyResult<PyRoundSummary> {
        // Fast path: no attacks pending, so client weights are read-only —
        // hand aggregation a slice of `&ClientUpdate` and skip the
        // per-round deep clone of every f32 across every layer.
        let result = if self.0.has_pending_attacks() {
            let owned: Vec<strategy::ClientUpdate> =
                client_updates.iter().map(|u| u.0.clone()).collect();
            self.0.run_round(owned, reported_loss)
        } else {
            let refs: Vec<&strategy::ClientUpdate> = client_updates.iter().map(|u| &u.0).collect();
            self.0.run_round_readonly(&refs, reported_loss)
        };
        result.map(PyRoundSummary).map_err(PyRuntimeError::new_err)
    }

    /// Current global model weights as a Python dict.
    fn global_weights(&self) -> HashMap<String, Vec<f32>> {
        self.0.global_weights.clone()
    }

    /// Overwrite the global model weights.
    ///
    /// Use this to seed from a real PyTorch initialisation rather than the
    /// default zeros. Layer names and lengths must match the ``layer_shapes``
    /// passed at construction time.
    fn set_global_weights(&mut self, weights: HashMap<String, Vec<f32>>) {
        self.0.set_global_weights(weights);
    }

    /// JSON-serialised experiment history (all completed round summaries).
    fn history_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.0.history).map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> String {
        format!(
            "Orchestrator(model='{}', rounds={}, history_len={})",
            self.0.config.model_id,
            self.0.config.rounds,
            self.0.history.len()
        )
    }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Aggregate a list of :class:`ClientUpdate` objects using a given strategy.
///
/// This free function is a diagnostic / testing helper. It returns only the
/// aggregated weights; for the full :class:`RoundSummary` (including
/// ``selected_client_ids``), drive aggregation through :class:`Orchestrator`.
///
/// Args:
///     updates: List of :class:`ClientUpdate` objects.
///     strategy: Aggregation :class:`Strategy`.
///
/// Returns:
///     Dict mapping layer names to aggregated weight lists.
#[pyfunction]
fn aggregate(
    updates: Vec<PyRef<PyClientUpdate>>,
    strategy: &PyStrategy,
) -> PyResult<HashMap<String, Vec<f32>>> {
    let refs: Vec<&strategy::ClientUpdate> = updates.iter().map(|u| &u.0).collect();
    crate::strategy::aggregate(&refs, &strategy.0)
        .map(|agg| agg.weights)
        .map_err(PyRuntimeError::new_err)
}

/// Apply Gaussian noise to a weight dict (in-place simulation).
///
/// Args:
///     weights: Dict mapping layer names to float lists.
///     std_dev: Standard deviation of the noise.
///
/// Returns:
///     JSON-encoded :class:`AttackResult`.
#[pyfunction]
fn apply_gaussian_noise(
    mut weights: HashMap<String, Vec<f32>>,
    std_dev: f64,
) -> PyResult<(HashMap<String, Vec<f32>>, String)> {
    let result = security::simulate_gaussian_noise(&mut weights, std_dev);
    let json =
        serde_json::to_string(&result).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok((weights, json))
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

/// velocity._core — Rust-backed VelocityFL engine.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyStrategy>()?;
    m.add_class::<PyClientUpdate>()?;
    m.add_class::<PyRoundSummary>()?;
    m.add_class::<PyOrchestrator>()?;
    m.add_function(wrap_pyfunction!(aggregate, m)?)?;
    m.add_function(wrap_pyfunction!(apply_gaussian_noise, m)?)?;
    Ok(())
}
