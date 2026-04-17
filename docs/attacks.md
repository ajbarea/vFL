# Attacks

Built-in simulation attacks:

- `model_poisoning`
- `sybil_nodes`
- `gaussian_noise`
- `label_flipping`

Use Python API:

```python
vfl.simulate_attack("sybil_nodes", count=5)
```

Or CLI:

```bash
velocity simulate-attack sybil_nodes --count 5
```
