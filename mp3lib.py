"""Core numerical functions for ECE 3715 Mini-Project 3.

The notebook imports the five functions required by the handout from this
module.  The two sampling helpers at the bottom are used by Bonus A.
"""

from __future__ import annotations

import numpy as np


def sample_mean_cov(X):
    """Return the sample mean vector and unbiased sample covariance matrix.

    Parameters
    ----------
    X : array_like, shape (N, d)
        Rows are observations and columns are variables.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must have shape (N, d)")
    n_samples, _ = X.shape
    if n_samples < 2:
        raise ValueError("at least two observations are required")

    mean_vector = np.mean(X, axis=0)
    centered = X - mean_vector
    cov_matrix = (centered.T @ centered) / (n_samples - 1)
    return mean_vector, cov_matrix


def is_valid_covariance(S, tol=1e-10):
    """Return True when S is symmetric positive semidefinite within tol."""
    S = np.asarray(S, dtype=float)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        return False
    if not np.allclose(S, S.T, atol=tol, rtol=0.0):
        return False

    # eigh is the numerically appropriate solver for a real symmetric matrix.
    eigenvalues = np.linalg.eigvalsh((S + S.T) / 2.0)
    return bool(np.min(eigenvalues) >= -tol)


def gaussian_pdf(x, mu, Sigma):
    """Evaluate a nonsingular multivariate Gaussian density.

    The determinant enters through slogdet.  The quadratic form is evaluated
    with solve instead of inv(Sigma) @ delta because solving a linear system
    is generally faster and better conditioned than explicitly forming an
    inverse.
    """
    x = np.asarray(x, dtype=float)
    mu = np.asarray(mu, dtype=float).reshape(-1)
    Sigma = np.asarray(Sigma, dtype=float)

    if Sigma.ndim != 2 or Sigma.shape[0] != Sigma.shape[1]:
        raise ValueError("Sigma must be square")
    dimension = mu.size
    if Sigma.shape != (dimension, dimension):
        raise ValueError("mu and Sigma dimensions do not agree")
    if not np.allclose(Sigma, Sigma.T, atol=1e-12, rtol=0.0):
        raise ValueError("Sigma must be symmetric")

    sign, logdet = np.linalg.slogdet(Sigma)
    if sign <= 0:
        raise ValueError(
            "Sigma must be positive definite for a density with respect to "
            "Lebesgue measure"
        )

    delta = x - mu
    if delta.shape[-1] != dimension:
        raise ValueError("the last axis of x must have length len(mu)")

    flat_delta = delta.reshape(-1, dimension)
    # Solve Sigma * z = delta rather than explicitly forming Sigma^{-1}.
    solved = np.linalg.solve(Sigma, flat_delta.T).T
    mahalanobis_squared = np.einsum(
        "ij,ij->i", flat_delta, solved, optimize=True
    )
    log_normalizer = -0.5 * (dimension * np.log(2.0 * np.pi) + logdet)
    density = np.exp(log_normalizer - 0.5 * mahalanobis_squared)
    return density.reshape(delta.shape[:-1])


def whitening_matrix(Sigma):
    """Return W such that W @ Sigma @ W.T is the identity matrix."""
    Sigma = np.asarray(Sigma, dtype=float)
    if Sigma.ndim != 2 or Sigma.shape[0] != Sigma.shape[1]:
        raise ValueError("Sigma must be square")
    if not np.allclose(Sigma, Sigma.T, atol=1e-12, rtol=0.0):
        raise ValueError("Sigma must be symmetric")

    eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
    if np.min(eigenvalues) <= 0:
        raise ValueError("Sigma must be positive definite to whiten it")

    inverse_square_roots = 1.0 / np.sqrt(eigenvalues)
    return np.diag(inverse_square_roots) @ eigenvectors.T


def lmmse_weights(Sigma_xx, Sigma_sx):
    """Return w in the scalar LMMSE estimate S_hat = w.T @ x."""
    Sigma_xx = np.asarray(Sigma_xx, dtype=float)
    Sigma_sx = np.asarray(Sigma_sx, dtype=float).reshape(-1)
    if Sigma_xx.ndim != 2 or Sigma_xx.shape[0] != Sigma_xx.shape[1]:
        raise ValueError("Sigma_xx must be square")
    if Sigma_xx.shape[0] != Sigma_sx.size:
        raise ValueError("Sigma_xx and Sigma_sx dimensions do not agree")

    # Solve Sigma_xx @ w = Sigma_sx instead of forming an explicit inverse.
    return np.linalg.solve(Sigma_xx, Sigma_sx)


def cholesky_samples(rng, covariance, n_samples):
    """Bonus A: generate zero-mean correlated rows using a Cholesky factor."""
    covariance = np.asarray(covariance, dtype=float)
    L = np.linalg.cholesky(covariance)
    z = rng.standard_normal((n_samples, covariance.shape[0]))
    return z @ L.T


def eigen_samples(rng, covariance, n_samples):
    """Bonus A: generate zero-mean correlated rows using an eigendecomposition."""
    covariance = np.asarray(covariance, dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if np.min(eigenvalues) < 0:
        raise ValueError("covariance cannot have a negative eigenvalue")
    transform = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0.0)))
    z = rng.standard_normal((n_samples, covariance.shape[0]))
    return z @ transform.T
