//! vfl-core: Rust core for VelocityFL.
//!
//! This crate exposes high-performance FL orchestration, aggregation strategies,
//! and security/attack simulation via PyO3 Python bindings.

mod orchestrator;
mod security;
mod strategy;

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

/// High-performance Rust-backed FL orchestrator.
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
    ///
    /// Returns:
    ///     :class:`RoundSummary` for the completed round.
    fn run_round(
        &mut self,
        client_updates: Vec<PyRef<PyClientUpdate>>,
    ) -> PyResult<PyRoundSummary> {
        let updates: Vec<strategy::ClientUpdate> =
            client_updates.iter().map(|u| u.0.clone()).collect();
        self.0
            .run_round(updates)
            .map(PyRoundSummary)
            .map_err(PyRuntimeError::new_err)
    }

    /// Current global model weights as a Python dict.
    fn global_weights(&self) -> HashMap<String, Vec<f32>> {
        self.0.global_weights.clone()
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
    let raw: Vec<strategy::ClientUpdate> = updates.iter().map(|u| u.0.clone()).collect();
    crate::strategy::aggregate(&raw, &strategy.0).map_err(PyRuntimeError::new_err)
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
