# Changelog

## 0.1.1 - 2026-11-11

Apply the stratum gate cap before reciprocal exponentiation and the desired
rank cap before integer conversion. Valid tiny positive inhibition values now
saturate instead of raising OverflowError. Zero remains invalid. Ordinary
finite configurations retain their behavior; loss descent is unchanged.

Four regression tests cover subnormals, budgets, ordinary values and large
inhibition. Includes the VAL revision impact analysis. Previous v0.1.0 benchmark
records and release remain historical.

## 0.1.0 - 2026-11-11

Initial open reference package. Includes NumPy LoRA optimization, log-domain OT,
identity penalties, transactional governance, bounded signatures, two seed
consensus heuristics, checkpoints, CLI, optional HF/PEFT trainer, independent
numerical tests, synthetic ablations and explicit failure examples.

Paper empirical results are not reproduced by this release.
