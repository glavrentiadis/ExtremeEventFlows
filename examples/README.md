# SA(0.2 s) rare-event training and estimator comparison

`train_sa_0p2.py` trains the autoregressive proposal for

\[
\max_{i:\,t_i\leq T} SA_i(0.2\,\mathrm{s}) > 0.5g.
\]

Run from the extracted package root:

```bash
python examples/train_sa_0p2.py
```

The script performs four steps:

1. Warm-starts the proposal by maximum likelihood using samples from the original physical model.
2. Adapts the proposal using staged weighted cross-entropy updates.
3. Estimates the exceedance probability with defensive-mixture importance sampling.
4. Estimates the same probability with naïve Monte Carlo and prints a side-by-side comparison.

Both estimators report:

- probability estimate;
- Monte Carlo standard error;
- relative standard error;
- confidence interval;
- number of samples; and
- number of sampled exceedances.

The comparison also reports wall-clock time and an empirical estimator-variance ratio. The variance ratio does not by itself account for the different computational cost per sample.

A fast execution test is:

```bash
python examples/train_sa_0p2.py \
    --pretrain-updates 1 \
    --pretrain-samples 8 \
    --updates-per-stage 1 \
    --samples-per-update 8 \
    --importance-samples 20 \
    --naive-samples 20
```

`--crude-samples` remains accepted as a backward-compatible alias for `--naive-samples`.

The source and ground-motion coefficients are illustrative. Replace them with calibrated source and GMM objects before scientific use.
