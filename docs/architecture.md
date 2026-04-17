# Architecture

## Layers

1. **Rust core (`vfl-core`)**
   - Aggregation strategies (FedAvg/FedProx/FedMedian)
   - Round orchestration
   - Attack simulation primitives
2. **PyO3 bindings**
   - Exposes Rust types as Python classes/functions
3. **Python package (`velocity`)**
   - Researcher-facing API (`VelocityServer`)
   - Prefect flow/task wrappers
   - CLI for quick operator workflows
