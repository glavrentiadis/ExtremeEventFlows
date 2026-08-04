# ExtremeEventFlows

This package separates the physical ground-motion application from the trainable normalizing-flow proposal.

## Files

**Source Files**
- `src/containers.py`: generic variable-length sequence containers.
- `src/transforms.py`: invertible scalar transforms.
- `src/site.py`: site with arbitrary model inputs.
- `src/source.py`: joint Markov source interface and a general latent-Gaussian implementation.
- `src/ground_motion.py`: arbitrary-input, multiple-output ground-motion interfaces.
- `src/application.py`: composition of source, site, and ground-motion models.
- `src/performance.py`: joint conditions involving multiple ground motions.
- `src/target.py`: physical and rare-event target density.
- `src/flow.py`: event-by-event autoregressive normalizing flow.
- `src/estimators.py`: crude Monte Carlo, importance sampling, and cross-entropy adaptation.
**Examples**
- `examples/example_single_gm.py` simple example
- `examples/example_multi_gm.py`: complete example with source location and three ground-motion outputs.

## Mathematical structure

The source target is a variable-length Markov process with a joint event density

\[
p(\mathcal X_N)
=
\prod_{i=1}^{N}
p(\mathbf s_i\mid \mathbf s_{i-1})
\;P(\Delta t_{N+1}>T-t_N\mid \mathbf s_N),
\]

where each event vector may contain any number of source parameters. The included example uses

\[
\mathbf s_i=(\Delta t_i,M_i,x_i,y_i,d_i).
\]

The ground-motion model may return any number of intensity measures. The example returns PGA, SA(0.2 s), and SA(1.0 s) with correlated within-event residuals.

The proposal factorizes autoregressively within each event and across events, but it represents a joint distribution because every current variable is conditioned on the previous variables and the sequence history.

## Run

```bash
python example.py
```

## Adding source variables

1. Add a named transform to `event_transforms`.
2. Extend `base_mean`, `transition_matrix`, `latent_cholesky`, `reference_event`, and `state_scale` consistently.
3. Update or replace the ground-motion feature builder if the new variable is a GMM input.

No changes are needed in the source model, target density, or normalizing flow.

## Adding ground-motion outputs

1. Add an output name.
2. Add a coefficient row.
3. Increase the residual Cholesky matrix.
4. Add a threshold or a custom performance function.

The flow automatically adds one latent residual variable for every GMM output.

## Custom physical models

For a different source distribution, subclass `JointMarkovSourceKernel` and implement joint sampling, joint log density, time survival, and state updates.

For a different ground-motion model, subclass `GroundMotionModel`. A complex neural, empirical, or physics-informed model can consume the complete `SourceSequence` and `Site` objects and return multiple named outputs.

For a system-level failure condition, subclass `PerformanceFunction` or use `CallablePerformanceFunction`; the callable receives the complete source sequence and all ground-motion outputs together.
