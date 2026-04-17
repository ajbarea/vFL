# Configuration

## Core server fields

- `model_id`: Hugging Face model identifier
- `dataset`: dataset source
- `strategy`: `FedAvg`, `FedProx`, `FedMedian`
- `storage`: storage URI
- `layer_shapes`: optional layer-size map

## Runtime fields

- `min_clients`
- `rounds`
