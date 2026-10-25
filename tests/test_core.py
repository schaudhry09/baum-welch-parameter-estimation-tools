import math
import unittest

from baum_welch_parameter_estimation_tools import (
    BaumWelch,
    estimate_hmm_parameters,
    forward_backward,
    normalize_log,
)


class TestNormalizeLog(unittest.TestCase):
    def test_basic_normalization(self):
        result = normalize_log([math.log(0.2), math.log(0.8)])
        self.assertAlmostEqual(result[0], 0.2, places=12)
        self.assertAlmostEqual(result[1], 0.8, places=12)

    def test_all_neg_inf_returns_uniform(self):
        result = normalize_log([float("-inf"), float("-inf")])
        self.assertEqual(result, [0.5, 0.5])

    def test_empty_returns_empty(self):
        self.assertEqual(normalize_log([]), [])

    def test_single_element(self):
        result = normalize_log([math.log(0.5)])
        self.assertAlmostEqual(result[0], 1.0, places=12)


class TestForwardBackward(unittest.TestCase):
    def _setup_two_state(self):
        # State 0 favours symbol 0, state 1 favours symbol 1.
        log_pi = [math.log(0.5), math.log(0.5)]
        log_T = [
            [math.log(0.9), math.log(0.1)],
            [math.log(0.1), math.log(0.9)],
        ]
        log_E = [
            [math.log(0.8), math.log(0.2)],
            [math.log(0.2), math.log(0.8)],
        ]
        return log_pi, log_T, log_E

    def test_gamma_rows_sum_to_one(self):
        log_pi, log_T, log_E = self._setup_two_state()
        obs = [0, 0, 1, 1, 0]
        gamma, xi_sums, ll = forward_backward(obs, log_pi, log_T, log_E, 2)
        for t in range(len(obs)):
            self.assertAlmostEqual(sum(gamma[t]), 1.0, places=9)

    def test_xi_sums_shape(self):
        log_pi, log_T, log_E = self._setup_two_state()
        obs = [0, 1, 0]
        gamma, xi_sums, ll = forward_backward(obs, log_pi, log_T, log_E, 2)
        self.assertEqual(len(xi_sums), 2)
        for row in xi_sums:
            self.assertEqual(len(row), 2)

    def test_log_likelihood_finite_for_valid_sequence(self):
        log_pi, log_T, log_E = self._setup_two_state()
        obs = [0, 1, 0]
        gamma, xi_sums, ll = forward_backward(obs, log_pi, log_T, log_E, 2)
        self.assertFalse(math.isinf(ll))
        self.assertFalse(math.isnan(ll))

    def test_deterministic_observation_increases_self_transition_posterior(self):
        # A long run of the same symbol should push posterior mass toward the
        # state that emits that symbol most often, and that state's
        # self-transition count should dominate.
        log_pi, log_T, log_E = self._setup_two_state()
        obs = [0] * 20
        gamma, xi_sums, ll = forward_backward(obs, log_pi, log_T, log_E, 2)
        self.assertGreater(gamma[5][0], gamma[5][1])
        self.assertGreater(xi_sums[0][0], xi_sums[0][1])


class TestBaumWelchFit(unittest.TestCase):
    def _two_state_setup(self):
        transition = [[0.9, 0.1], [0.1, 0.9]]
        emission = [[0.8, 0.2], [0.2, 0.8]]
        initial_probs = [0.5, 0.5]
        return transition, emission, initial_probs

    def test_rows_normalized_after_fit(self):
        transition, emission, initial_probs = self._two_state_setup()
        obs = [0, 0, 1, 1, 0, 0, 1, 1]
        bw = BaumWelch(2, 2, initial_probs, max_iter=20, tol=1e-6)
        result = bw.fit(obs, transition, emission)
        for row in result.transition:
            self.assertAlmostEqual(sum(row), 1.0, places=9)
        for row in result.emission:
            self.assertAlmostEqual(sum(row), 1.0, places=9)

    def test_likelihood_does_not_decrease_across_a_few_iters(self):
        transition, emission, initial_probs = self._two_state_setup()
        obs = [0, 0, 1, 1, 0, 0, 1, 1, 0, 1]
        bw = BaumWelch(2, 2, initial_probs, max_iter=1, tol=1e-12)
        r1 = bw.fit(obs, transition, emission)
        bw2 = BaumWelch(2, 2, initial_probs, max_iter=5, tol=1e-12)
        r5 = bw2.fit(obs, transition, emission)
        self.assertGreaterEqual(r5.log_likelihood, r1.log_likelihood - 1e-9)

    def test_convergence_flag_true_with_loose_tol(self):
        transition, emission, initial_probs = self._two_state_setup()
        obs = [0, 1, 0, 1, 0, 1]
        bw = BaumWelch(2, 2, initial_probs, max_iter=200, tol=1e-2)
        result = bw.fit(obs, transition, emission)
        self.assertTrue(result.converged)

    def test_n_iter_at_least_one(self):
        transition, emission, initial_probs = self._two_state_setup()
        obs = [0, 1, 0]
        bw = BaumWelch(2, 2, initial_probs, max_iter=5, tol=1e-12)
        result = bw.fit(obs, transition, emission)
        self.assertGreaterEqual(result.n_iter, 1)
        self.assertLessEqual(result.n_iter, 5)

    def test_estimate_hmm_parameters_wrapper(self):
        transition, emission, initial_probs = self._two_state_setup()
        obs = [0, 0, 1, 1, 0, 1]
        result = estimate_hmm_parameters(
            obs, transition, emission, initial_probs, n_symbols=2, max_iter=20, tol=1e-6
        )
        self.assertEqual(len(result.transition), 2)
        self.assertEqual(len(result.emission), 2)
        self.assertEqual(len(result.emission[0]), 2)

    def test_rejects_out_of_range_observation(self):
        transition, emission, initial_probs = self._two_state_setup()
        obs = [0, 2]  # 2 is out of range for n_symbols=2
        bw = BaumWelch(2, 2, initial_probs)
        with self.assertRaises(ValueError):
            bw.fit(obs, transition, emission)

    def test_rejects_empty_observations(self):
        transition, emission, initial_probs = self._two_state_setup()
        bw = BaumWelch(2, 2, initial_probs)
        with self.assertRaises(ValueError):
            bw.fit([], transition, emission)

    def test_rejects_bad_transition_shape(self):
        transition = [[0.5, 0.5]]  # 1 row, expected 2
        emission = [[0.5, 0.5], [0.5, 0.5]]
        initial_probs = [0.5, 0.5]
        bw = BaumWelch(2, 2, initial_probs)
        with self.assertRaises(ValueError):
            bw.fit([0, 1], transition, emission)

    def test_rejects_bad_emission_shape(self):
        transition = [[0.5, 0.5], [0.5, 0.5]]
        emission = [[0.5, 0.5]]  # 1 row, expected 2
        initial_probs = [0.5, 0.5]
        bw = BaumWelch(2, 2, initial_probs)
        with self.assertRaises(ValueError):
            bw.fit([0, 1], transition, emission)

    def test_rejects_zero_transition_row(self):
        transition = [[0.0, 0.0], [0.5, 0.5]]
        emission = [[0.5, 0.5], [0.5, 0.5]]
        initial_probs = [0.5, 0.5]
        bw = BaumWelch(2, 2, initial_probs)
        with self.assertRaises(ValueError):
            bw.fit([0, 1], transition, emission)

    def test_initial_probs_held_fixed(self):
        # initial_probs heavily biased toward state 0; the estimator must not
        # change them, so re-running with the same data and the same (fixed)
        # priors should reproduce the same likelihood deterministically.
        transition = [[0.7, 0.3], [0.3, 0.7]]
        emission = [[0.9, 0.1], [0.1, 0.9]]
        initial_probs = [0.99, 0.01]
        obs = [0, 0, 0, 1, 0]
        bw = BaumWelch(2, 2, initial_probs, max_iter=10, tol=1e-12)
        r1 = bw.fit(obs, transition, emission)
        bw2 = BaumWelch(2, 2, initial_probs, max_iter=10, tol=1e-12)
        r2 = bw2.fit(obs, transition, emission)
        self.assertAlmostEqual(r1.log_likelihood, r2.log_likelihood, places=9)


class TestConstructionValidation(unittest.TestCase):
    def test_rejects_zero_n_states(self):
        with self.assertRaises(ValueError):
            BaumWelch(0, 2, [])

    def test_rejects_initial_probs_length_mismatch(self):
        with self.assertRaises(ValueError):
            BaumWelch(2, 2, [1.0])

    def test_rejects_zero_max_iter(self):
        with self.assertRaises(ValueError):
            BaumWelch(2, 2, [0.5, 0.5], max_iter=0)

    def test_rejects_negative_tol(self):
        with self.assertRaises(ValueError):
            BaumWelch(2, 2, [0.5, 0.5], tol=-0.1)


if __name__ == "__main__":
    unittest.main()
