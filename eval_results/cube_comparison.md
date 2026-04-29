# Cube Checkpoint Comparison

- **ADV**: `lewm-ogbcube-adv` (adversarially fine-tuned)
- **Base**: `lewm-cube` (pretrained)
- **Seeds**: [42, 123, 456, 789, 1024]
- **Solver**: adam (n_steps=30, lr=0.1, horizon=5)
- **num_eval**: 50
- **goal_offset_steps**: 50

| Epoch | ADV Mean | ADV Std | ADV Rates (seeds) | Base Mean | Base Std | Base Rates |
|------:|---------:|--------:|-------------------|----------:|---------:|------------|
| 2 | 56.0% | 5.7% | 52%, 64%, 52% (42,456,789; 123+1024 failed) | 51.6% | 5.0% | 54%, 44%, 58%, 54%, 48% |
