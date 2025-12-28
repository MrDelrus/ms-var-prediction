import numpy as np
from numba import njit
from ms_var_prediction.utils import gaussian_pdf


@njit
def loglikelyhood_gaussian(params: np.ndarray, returns: np.ndarray, n_states: int):
    """
    Compute negative log-likelihood and final state probabilities for N-state Gaussian MS model.

    Parameters
    ----------
    params : np.ndarray
        Flattened array: [mus..., sigmas..., P_flat...].
    returns : np.ndarray
        Array of returns.
    n_states : int
        Number of hidden states.

    Returns
    -------
    nll : np.float64
        Negative log-likelihood.
    state_probs_last : np.ndarray
        Probabilities of each state at the last observation.
    """
    n = returns.shape[0]
    mus = params[:n_states]
    sigmas = params[n_states : 2 * n_states]
    P = params[2 * n_states :].reshape((n_states, n_states))

    rho = np.ones(n + 1, dtype=np.float64)
    state_probs = np.zeros((n + 1, n_states), dtype=np.float64)
    state_probs[0, :] = 1.0 / n_states

    for t in range(n):
        prob_next = state_probs[t, :] @ P
        likelihood_t = 0.0
        for i in range(n_states):
            likelihood_t += prob_next[i] * gaussian_pdf(returns[t], mus[i], sigmas[i])
        rho[t + 1] = likelihood_t
        for i in range(n_states):
            state_probs[t + 1, i] = (
                prob_next[i] * gaussian_pdf(returns[t], mus[i], sigmas[i]) / rho[t + 1]
            )

    nll = -np.sum(np.log(rho))
    return np.float64(nll), state_probs[-1]
