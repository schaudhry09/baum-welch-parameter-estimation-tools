"""Baum-Welch parameter estimation for discrete hidden Markov models.

The implementation works entirely in log-space for the forward and backward
sweeps so that long observation sequences do not underflow.  Probabilities are
only re-materialised at the end of each E-step when expected transition and
emission counts are accumulated.

Design decisions (stated plainly so the tests and the reader agree on what is
supported):

* Discrete emissions only.  ``observations`` is a sequence of integer indices
  into ``n_symbols``.  Continuous emissions are out of scope.
* Observation indices are validated and must fall in ``range(n_symbols)``.
  This is the one place the library raises rather than returning NaNs, because
  a bad index is almost always a caller bug.
* Initial state probabilities are held fixed across iterations.  Re-estimating
  them from the first time-step is notoriously unstable on short sequences, so
  the caller controls them explicitly.
* Convergence is judged on the change in log-likelihood between iterations.
  ``tol`` is an absolute threshold on that change, not a relative one.  This is
  easier to reason about and deterministic across platforms.
* Parameters that become impossible (zero expected count) keep a tiny floor
  value so a subsequent E-step never takes ``log(0)``.  The floor is data-only;
  it does not distort returned parameters beyond the caller's own smoothing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

# A small constant used as a probability floor during re-estimation.  It exists
# to keep the next iteration's log-space computations finite; it does not leak
# into the caller-visible result because the final parameters are normalised
# after flooring.
_FLOOR = 1e-12


def normalize_log(values: Sequence[float]) -> List[float]:
    """Normalise a sequence of log-probabilities so they sum to one.

    Returns ordinary (non-log) probabilities.  Used to turn a row of log-scores
    into a usable probability distribution.  If every entry is ``-inf`` (the
    sequence was impossible under every hypothesis) the result is a uniform
    distribution — this only happens inside the E-step for states that have
    already been floored, so it is a safety net rather than a normal path.
    """
    n = len(values)
    if n == 0:
        return []

    max_val = max(values)
    if max_val == float("-inf"):
        # Every option is impossible; fall back to uniform rather than NaN.
        return [1.0 / n] * n

    total = 0.0
    exps: List[float] = []
    for v in values:
        e = math.exp(v - max_val)
        exps.append(e)
        total += e

    if total <= 0.0:
        return [1.0 / n] * n
    return [e / total for e in exps]


def _logsumexp(values: Sequence[float]) -> float:
    """Numerically stable log-sum-exp."""
    if not values:
        return float("-inf")
    max_val = max(values)
    if max_val == float("-inf"):
        return float("-inf")
    s = 0.0
    for v in values:
        s += math.exp(v - max_val)
    return max_val + math.log(s)


def _validate_inputs(
    observations: Sequence[int],
    n_states: int,
    n_symbols: int,
) -> None:
    if n_states <= 0:
        raise ValueError(f"n_states must be positive, got {n_states}")
    if n_symbols <= 0:
        raise ValueError(f"n_symbols must be positive, got {n_symbols}")
    if len(observations) == 0:
        raise ValueError("observations must be non-empty")
    for i, o in enumerate(observations):
        if not isinstance(o, int):
            raise TypeError(
                f"observations[{i}] must be int, got {type(o).__name__}"
            )
        if o < 0 or o >= n_symbols:
            raise ValueError(
                f"observations[{i}]={o} out of range for n_symbols={n_symbols}"
            )


def _log_transition_matrix(
    transition: Sequence[Sequence[float]],
    n_states: int,
) -> List[List[float]]:
    log_T: List[List[float]] = []
    for i in range(n_states):
        row = [float(transition[i][j]) for j in range(n_states)]
        total = sum(row)
        if total <= 0.0:
            raise ValueError(f"transition row {i} sums to non-positive value")
        log_T.append([math.log(row[j] / total) for j in range(n_states)])
    return log_T


def _log_emission_matrix(
    emission: Sequence[Sequence[float]],
    n_states: int,
    n_symbols: int,
) -> List[List[float]]:
    log_E: List[List[float]] = []
    for i in range(n_states):
        row = [float(emission[i][k]) for k in range(n_symbols)]
        total = sum(row)
        if total <= 0.0:
            raise ValueError(f"emission row {i} sums to non-positive value")
        log_E.append([math.log(row[k] / total) for k in range(n_symbols)])
    return log_E


def _log_pi(pi: Sequence[float], n_states: int) -> List[float]:
    row = [float(pi[i]) for i in range(n_states)]
    total = sum(row)
    if total <= 0.0:
        raise ValueError("initial probabilities sum to non-positive value")
    return [math.log(row[i] / total) for i in range(n_states)]


def forward_backward(
    observations: Sequence[int],
    log_pi: Sequence[float],
    log_T: Sequence[Sequence[float]],
    log_E: Sequence[Sequence[float]],
    n_states: int,
) -> Tuple[List[List[float]], List[List[float]], float]:
    """Run the forward-backward algorithm in log-space.

    Returns ``(gamma, xi_sums, log_likelihood)`` where:

    * ``gamma[t][i]`` is the posterior probability of being in state ``i`` at
      time ``t``.
    * ``xi_sums[i][j]`` is the sum over all time-steps of the posterior
      probability of transitioning ``i -> j``.  These are the expected counts
      consumed directly by the M-step.
    * ``log_likelihood`` is the log probability of the observation sequence
      under the current parameters.
    """
    T = len(observations)
    log_alpha: List[List[float]] = [[0.0] * n_states for _ in range(T)]
    log_beta: List[List[float]] = [[0.0] * n_states for _ in range(T)]

    # Forward pass.
    for i in range(n_states):
        log_alpha[0][i] = log_pi[i] + log_E[i][observations[0]]
    for t in range(1, T):
        o = observations[t]
        for j in range(n_states):
            best = float("-inf")
            for i in range(n_states):
                v = log_alpha[t - 1][i] + log_T[i][j]
                if v > best:
                    best = v
            if best == float("-inf"):
                log_alpha[t][j] = float("-inf")
            else:
                s = 0.0
                for i in range(n_states):
                    s += math.exp(log_alpha[t - 1][i] + log_T[i][j] - best)
                log_alpha[t][j] = best + math.log(s) + log_E[j][o]

    # Backward pass.
    for i in range(n_states):
        log_beta[T - 1][i] = 0.0
    for t in range(T - 2, -1, -1):
        o_next = observations[t + 1]
        for i in range(n_states):
            best = float("-inf")
            for j in range(n_states):
                v = log_T[i][j] + log_E[j][o_next] + log_beta[t + 1][j]
                if v > best:
                    best = v
            if best == float("-inf"):
                log_beta[t][i] = float("-inf")
            else:
                s = 0.0
                for j in range(n_states):
                    s += math.exp(
                        log_T[i][j] + log_E[j][o_next] + log_beta[t + 1][j] - best
                    )
                log_beta[t][i] = best + math.log(s)

    log_likelihood = _logsumexp(log_alpha[T - 1])

    # Gamma: posterior of state i at time t.
    gamma: List[List[float]] = [[0.0] * n_states for _ in range(T)]
    for t in range(T):
        log_post = [log_alpha[t][i] + log_beta[t][i] for i in range(n_states)]
        gamma[t] = normalize_log(log_post)

    # Xi sums: expected transition counts, accumulated over t=0..T-2.
    xi_sums: List[List[float]] = [[0.0] * n_states for _ in range(n_states)]
    for t in range(T - 1):
        o_next = observations[t + 1]
        log_xi: List[List[float]] = [
            [0.0] * n_states for _ in range(n_states)
        ]
        for i in range(n_states):
            for j in range(n_states):
                log_xi[i][j] = (
                    log_alpha[t][i]
                    + log_T[i][j]
                    + log_E[j][o_next]
                    + log_beta[t + 1][j]
                )
        # Normalise the full (i, j) grid for this time-step.
        flat = [log_xi[i][j] for i in range(n_states) for j in range(n_states)]
        norm = normalize_log(flat)
        idx = 0
        for i in range(n_states):
            for j in range(n_states):
                xi_sums[i][j] += norm[idx]
                idx += 1

    return gamma, xi_sums, log_likelihood


@dataclass
class BWResult:
    """Estimated parameters and diagnostics.

    ``converged`` is True when the improvement in log-likelihood between the
    last two iterations was at or below ``tol``.  ``n_iter`` is the number of
    E-M iterations actually performed (at least one).
    """

    transition: List[List[float]]
    emission: List[List[float]]
    log_likelihood: float
    n_iter: int
    converged: bool


class BaumWelch:
    """Iterative estimator for an HMM's transition and emission matrices.

    ``initial_probs`` are held fixed; only ``transition`` and ``emission`` are
    re-estimated.  See the module docstring for why.
    """

    def __init__(
        self,
        n_states: int,
        n_symbols: int,
        initial_probs: Sequence[float],
        max_iter: int = 100,
        tol: float = 1e-4,
    ) -> None:
        if n_states <= 0:
            raise ValueError(f"n_states must be positive, got {n_states}")
        if n_symbols <= 0:
            raise ValueError(f"n_symbols must be positive, got {n_symbols}")
        if len(initial_probs) != n_states:
            raise ValueError(
                f"initial_probs length {len(initial_probs)} != n_states {n_states}"
            )
        if max_iter <= 0:
            raise ValueError(f"max_iter must be positive, got {max_iter}")
        if tol < 0:
            raise ValueError(f"tol must be non-negative, got {tol}")

        self.n_states = n_states
        self.n_symbols = n_symbols
        self.max_iter = max_iter
        self.tol = tol
        self.log_pi = _log_pi(initial_probs, n_states)

    def fit(
        self,
        observations: Sequence[int],
        transition: Sequence[Sequence[float]],
        emission: Sequence[Sequence[float]],
    ) -> BWResult:
        """Run E-M until convergence or ``max_iter``.

        ``transition`` and ``emission`` are the starting estimates; each row is
        normalised internally so the caller need not pre-normalise.
        """
        _validate_inputs(observations, self.n_states, self.n_symbols)
        if len(transition) != self.n_states:
            raise ValueError(
                f"transition has {len(transition)} rows, expected {self.n_states}"
            )
        for i, row in enumerate(transition):
            if len(row) != self.n_states:
                raise ValueError(
                    f"transition row {i} has length {len(row)}, "
                    f"expected {self.n_states}"
                )
        if len(emission) != self.n_states:
            raise ValueError(
                f"emission has {len(emission)} rows, expected {self.n_states}"
            )
        for i, row in enumerate(emission):
            if len(row) != self.n_symbols:
                raise ValueError(
                    f"emission row {i} has length {len(row)}, "
                    f"expected {self.n_symbols}"
                )

        log_T = _log_transition_matrix(transition, self.n_states)
        log_E = _log_emission_matrix(emission, self.n_states, self.n_symbols)

        prev_ll = float("-inf")
        n_iter = 0
        converged = False
        gamma: List[List[float]] = []
        xi_sums: List[List[float]] = []
        ll = float("-inf")

        for _ in range(self.max_iter):
            n_iter += 1
            gamma, xi_sums, ll = forward_backward(
                observations, self.log_pi, log_T, log_E, self.n_states
            )

            # M-step: re-estimate transition and emission from expected counts.
            new_log_T: List[List[float]] = []
            for i in range(self.n_states):
                row_counts = [xi_sums[i][j] for j in range(self.n_states)]
                floored = [max(c, _FLOOR) for c in row_counts]
                total = sum(floored)
                new_log_T.append([math.log(floored[j] / total) for j in range(self.n_states)])

            new_log_E: List[List[float]] = []
            for i in range(self.n_states):
                counts = [0.0] * self.n_symbols
                for t in range(len(observations)):
                    counts[observations[t]] += gamma[t][i]
                floored = [max(c, _FLOOR) for c in counts]
                total = sum(floored)
                new_log_E.append([math.log(floored[k] / total) for k in range(self.n_symbols)])

            log_T = new_log_T
            log_E = new_log_E

            if prev_ll != float("-inf") and abs(ll - prev_ll) <= self.tol:
                converged = True
                break
            prev_ll = ll

        # Materialise probabilities for the caller.
        out_T = [
            [math.exp(log_T[i][j]) for j in range(self.n_states)]
            for i in range(self.n_states)
        ]
        out_E = [
            [math.exp(log_E[i][k]) for k in range(self.n_symbols)]
            for i in range(self.n_states)
        ]
        # Renormalise to absorb any floating-point drift so rows sum to ~1.
        for i in range(self.n_states):
            sT = sum(out_T[i])
            out_T[i] = [v / sT for v in out_T[i]]
            sE = sum(out_E[i])
            out_E[i] = [v / sE for v in out_E[i]]

        return BWResult(
            transition=out_T,
            emission=out_E,
            log_likelihood=ll,
            n_iter=n_iter,
            converged=converged,
        )


def estimate_hmm_parameters(
    observations: Sequence[int],
    initial_transition: Sequence[Sequence[float]],
    initial_emission: Sequence[Sequence[float]],
    initial_probs: Sequence[float],
    n_symbols: int,
    max_iter: int = 100,
    tol: float = 1e-4,
) -> BWResult:
    """Convenience wrapper around :class:`BaumWelch`.

    The number of hidden states is inferred from the shape of
    ``initial_transition``.  This keeps the common one-shot call site short
    without hiding the fact that the estimator object owns the iteration.
    """
    n_states = len(initial_transition)
    bw = BaumWelch(
        n_states=n_states,
        n_symbols=n_symbols,
        initial_probs=initial_probs,
        max_iter=max_iter,
        tol=tol,
    )
    return bw.fit(observations, initial_transition, initial_emission)
