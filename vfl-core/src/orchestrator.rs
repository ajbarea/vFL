use std::collections::HashMap;

use crate::security::AttackResult;
use crate::strategy::{aggregate, ClientUpdate, Strategy};

/// Configuration for a federated learning experiment.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ExperimentConfig {
    pub model_id: String,
    pub dataset: String,
    pub strategy: Strategy,
    pub storage: String,
    pub min_clients: usize,
    pub rounds: usize,
}

/// Per-round summary produced by the orchestrator.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RoundSummary {
    pub round: usize,
    pub num_clients: usize,
    pub global_loss: f64,
    pub attack_results: Vec<AttackResult>,
}

/// The central FL orchestrator.
///
/// Manages rounds, collects client updates, applies aggregation, and
/// optionally injects attack simulations.
#[derive(Debug)]
pub struct Orchestrator {
    pub config: ExperimentConfig,
    /// Current global model weights.
    pub global_weights: HashMap<String, Vec<f32>>,
    /// Summaries of completed rounds.
    pub history: Vec<RoundSummary>,
    /// Pending attacks to simulate in the next round.
    pending_attacks: Vec<crate::security::AttackType>,
}

impl Orchestrator {
    /// Create a new orchestrator with zero-initialised weights for the given layer shapes.
    pub fn new(config: ExperimentConfig, layer_shapes: &HashMap<String, usize>) -> Self {
        let global_weights = layer_shapes
            .iter()
            .map(|(name, &len)| (name.clone(), vec![0.0f32; len]))
            .collect();

        Orchestrator {
            config,
            global_weights,
            history: Vec::new(),
            pending_attacks: Vec::new(),
        }
    }

    /// Register an attack to be simulated during the next training round.
    pub fn register_attack(&mut self, attack: crate::security::AttackType) {
        self.pending_attacks.push(attack);
    }

    /// Run a single federated learning round.
    ///
    /// `client_updates` must contain at least `config.min_clients` updates.
    pub fn run_round(
        &mut self,
        mut client_updates: Vec<ClientUpdate>,
    ) -> Result<RoundSummary, String> {
        let round = self.history.len() + 1;
        if client_updates.len() < self.config.min_clients {
            return Err(format!(
                "Insufficient clients: need {}, got {}",
                self.config.min_clients,
                client_updates.len()
            ));
        }

        let mut attack_results: Vec<AttackResult> = Vec::new();

        // Apply pending attack simulations
        for attack in self.pending_attacks.drain(..) {
            match attack {
                crate::security::AttackType::ModelPoisoning { intensity } => {
                    // Poison the first client's weights as a demonstration
                    if let Some(u) = client_updates.first_mut() {
                        let result =
                            crate::security::simulate_model_poisoning(&mut u.weights, intensity);
                        attack_results.push(result);
                    }
                }
                crate::security::AttackType::SybilNodes { count } => {
                    let layer_shapes: HashMap<String, usize> = self
                        .global_weights
                        .iter()
                        .map(|(k, v)| (k.clone(), v.len()))
                        .collect();
                    let (sybil_updates, result) =
                        crate::security::simulate_sybil_nodes(&layer_shapes, count);
                    client_updates.extend(sybil_updates);
                    attack_results.push(result);
                }
                crate::security::AttackType::GaussianNoise { std_dev } => {
                    let result = crate::security::simulate_gaussian_noise(
                        &mut self.global_weights,
                        std_dev,
                    );
                    attack_results.push(result);
                }
                crate::security::AttackType::LabelFlipping { fraction } => {
                    attack_results.push(AttackResult {
                        attack_type: "label_flipping".to_string(),
                        clients_affected: (client_updates.len() as f64 * fraction) as usize,
                        severity: fraction,
                        description: format!(
                            "Label flipping on {:.0}% of clients",
                            fraction * 100.0
                        ),
                    });
                }
            }
        }

        // Aggregate
        let new_weights = aggregate(&client_updates, &self.config.strategy)?;
        self.global_weights = new_weights;

        // Compute a simple mock loss (L2 norm of global weights as proxy)
        let global_loss = self
            .global_weights
            .values()
            .flat_map(|v| v.iter())
            .map(|&x| (x as f64).powi(2))
            .sum::<f64>()
            .sqrt();

        let summary = RoundSummary {
            round,
            num_clients: client_updates.len(),
            global_loss,
            attack_results,
        };
        self.history.push(summary.clone());
        Ok(summary)
    }

    /// Run all configured rounds, providing client updates via the supplied callback.
    ///
    /// `client_provider` is called once per round and must return at least
    /// `config.min_clients` updates.
    #[allow(dead_code)]
    pub fn run<F>(&mut self, mut client_provider: F) -> Result<Vec<RoundSummary>, String>
    where
        F: FnMut(usize, &HashMap<String, Vec<f32>>) -> Vec<ClientUpdate>,
    {
        for r in 0..self.config.rounds {
            let updates = client_provider(r, &self.global_weights);
            self.run_round(updates)?;
        }
        Ok(self.history.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn make_config(rounds: usize) -> ExperimentConfig {
        ExperimentConfig {
            model_id: "test-model".to_string(),
            dataset: "test-dataset".to_string(),
            strategy: Strategy::FedAvg,
            storage: "local://".to_string(),
            min_clients: 2,
            rounds,
        }
    }

    fn make_layer_shapes() -> HashMap<String, usize> {
        let mut m = HashMap::new();
        m.insert("fc1".to_string(), 4);
        m
    }

    fn make_update(val: f32) -> ClientUpdate {
        let mut weights = HashMap::new();
        weights.insert("fc1".to_string(), vec![val; 4]);
        ClientUpdate {
            num_samples: 10,
            weights,
        }
    }

    #[test]
    fn single_round_aggregates_correctly() {
        let mut orch = Orchestrator::new(make_config(1), &make_layer_shapes());
        let updates = vec![make_update(1.0), make_update(3.0)];
        let summary = orch.run_round(updates).unwrap();
        assert_eq!(summary.round, 1);
        // FedAvg of equal-sample updates: (1+3)/2 = 2
        let w = &orch.global_weights["fc1"];
        assert!((w[0] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn insufficient_clients_returns_error() {
        let mut orch = Orchestrator::new(make_config(1), &make_layer_shapes());
        let result = orch.run_round(vec![make_update(1.0)]);
        assert!(result.is_err());
    }

    #[test]
    fn multi_round_run() {
        let mut orch = Orchestrator::new(make_config(3), &make_layer_shapes());
        let summaries = orch
            .run(|_, _| vec![make_update(1.0), make_update(1.0)])
            .unwrap();
        assert_eq!(summaries.len(), 3);
    }

    #[test]
    fn attack_registration_is_applied() {
        let mut orch = Orchestrator::new(make_config(1), &make_layer_shapes());
        orch.register_attack(crate::security::AttackType::ModelPoisoning { intensity: 1.0 });
        let updates = vec![make_update(1.0), make_update(1.0)];
        let summary = orch.run_round(updates).unwrap();
        assert!(!summary.attack_results.is_empty());
        assert_eq!(summary.attack_results[0].attack_type, "model_poisoning");
    }
}
