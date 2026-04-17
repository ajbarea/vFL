# CLI Reference

VelocityFL ships with a Typer-based CLI.

```bash
velocity --help
```

## Commands

### `velocity version`
Prints the installed package version.

### `velocity strategies`
Prints supported aggregation strategies.

### `velocity run`
Runs a local orchestrated experiment.

```bash
velocity run --model-id meta-llama/Llama-3-8B --dataset huggingface/ultrafeedback --rounds 5 --min-clients 10
```

### `velocity simulate-attack`
Registers an attack and runs a single validation round.

```bash
velocity simulate-attack model_poisoning --intensity 0.2
```
