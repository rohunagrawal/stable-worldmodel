# Pusht Checkpoint Comparison

- **ADV**: `lewm-pusht-adv` (adversarially fine-tuned)
- **Base**: `lewm-pusht` (pretrained)
- **Seeds**: [42, 123, 456, 789, 1024]
- **Solver**: adam (n_steps=30, lr=0.1, horizon=5)
- **num_eval**: 50

| Epoch | ADV Mean | ADV Std | ADV Rates | Base Mean | Base Std | Base Rates |
|------:|---------:|--------:|-----------|----------:|--------:|------------|
| 1 | 64.4% | 8.2% | 80%, 62%, 56%, 60%, 64% | 66.4% | 9.1% | 80%, 60%, 54%, 72%, 66% |
