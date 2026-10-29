# baum-welch-parameter-estimation-tools

Estimates the emission and transition probabilities of a discrete hidden Markov model from an observation sequence using the expectation-maximization algorithm. Implemented in pure Python with no third-party dependencies.

## Usage

```python
from baum_welch_parameter_estimation_tools import BaumWelch, estimate_hmm_parameters

observations = [0, 0, 1, 1, 0, 0, 1, 1, 0, 1]

# Long form: construct once, fit many sequences.
bw = BaumWelch(
    n_states=2,
    n_symbols=2,
    initial_probs=[0.5, 0.5],
    max_iter=100,
    tol=1e-4,
)
result = bw.fit(
    observations,
    transition=[[0.9, 0.1], [0.1, 0.9]],
    emission=[[0.8, 0.2], [0.2, 0.8]],
)
print(result.transition)   # re-estimated transition matrix
print(result.emission)     # re-estimated emission matrix
print(result.log_likelihood)
print(result.converged, result.n_iter)

# Short form: one-shot convenience wrapper.
result = estimate_hmm_parameters(
    observations,
    initial_transition=[[0.9, 0.1], [0.1, 0.9]],
    initial_emission=[[0.8, 0.2], [0.2, 0.8]],
    initial_probs=[0.5, 0.5],
    n_symbols=2,
)
```

`forward_backward` and `normalize_log` are also exported for callers who need the E-step quantities (state posteriors and expected transition counts) without running a full E-M loop.

## Why this exists

The specific problem: you have an observation sequence and a guess at the structure of a hidden Markov model, and you want better transition and emission probabilities without pulling in a numerical computing stack. Standard-library Python keeps deployment trivial — the test suite runs with nothing installed.

The trade-off: initial state probabilities are held fixed across iterations rather than re-estimated. Re-estimating the priors from the first time-step is notoriously unstable on short sequences, so this library makes the caller own them. If you need learned priors, run the estimator on many sequences and average the first-step posteriors yourself.

## The awkward edge

Observation sequences are integer indices into `n_symbols`, not raw tokens. A symbol index outside `range(n_symbols)` raises `ValueError` immediately — the library will not silently produce NaNs. Convergence is judged on the absolute change in log-likelihood between iterations (`tol`), not a relative threshold, so pick `tol` with that in mind.

## Performance

The window keeps a bounded buffer, so `push` is constant time and memory does not
grow with the length of the stream. `peak` and `trough` are linear in the window
size, which is the trade that keeps `push` cheap.

