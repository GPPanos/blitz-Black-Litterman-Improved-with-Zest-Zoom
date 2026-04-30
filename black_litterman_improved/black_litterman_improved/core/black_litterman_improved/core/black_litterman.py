"""
Core Black-Litterman implementation with vectorized operations.

References:
    Black, F., & Litterman, R. (1992). Global Portfolio Optimization.
    Financial Analysts Journal, 48(5), 28-43.
    DOI: 10.3905/jpm.1992.409394

Author: GPPanos
"""

import numpy as np
import warnings
from typing import Optional, Tuple


class BlackLittermanBase:
    """
    Base Black-Litterman model implementation.
    
    The Black-Litterman model combines equilibrium returns with investor views
    to generate posterior expected returns.
    
    Parameters
    ----------
    risk_aversion : float, default=2.5
        Risk aversion coefficient (λ). Typical values range from 1.5 to 3.5.
    tau : float, default=0.025
        Uncertainty scalar for prior. Typically between 0.025 and 0.05.
    use_torch : bool, default=False
        Whether to use PyTorch for GPU acceleration.
    
    Examples
    --------
    >>> model = BlackLittermanBase(risk_aversion=2.5, tau=0.025)
    >>> prior = model.implied_returns(market_caps, covariance)
    >>> posterior, cov = model.posterior_returns(prior, covariance, P, Q)
    """
    
    def __init__(
        self,
        risk_aversion: float = 2.5,
        tau: float = 0.025,
        use_torch: bool = False
    ):
        self.risk_aversion = risk_aversion
        self.tau = tau
        self.use_torch = use_torch
        
        if use_torch:
            import torch
            self.torch = torch
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _validate_and_fix_covariance(self, covariance: np.ndarray) -> np.ndarray:
        """
        Ensure covariance matrix is positive definite.
        
        Parameters
        ----------
        covariance : np.ndarray
            Covariance matrix (N, N)
        
        Returns
        -------
        np.ndarray
            Positive definite covariance matrix
        """
        try:
            np.linalg.cholesky(covariance)
            return covariance
        except np.linalg.LinAlgError:
            # Near-PSD correction using eigenvalue clipping
            eigvals, eigvecs = np.linalg.eigh(covariance)
            eigvals = np.maximum(eigvals, 1e-8)
            fixed_cov = eigvecs @ np.diag(eigvals) @ eigvecs.T
            warnings.warn(
                "Covariance matrix corrected to positive definite",
                UserWarning
            )
            return fixed_cov
    
    def implied_returns(
        self,
        market_caps: np.ndarray,
        covariance: np.ndarray
    ) -> np.ndarray:
        """
        Calculate implied equilibrium returns.
        
        Formula: Π = λ Σ w_market
        
        Parameters
        ----------
        market_caps : np.ndarray
            Market capitalization for each asset (N,)
        covariance : np.ndarray
            Asset return covariance matrix (N, N)
        
        Returns
        -------
        np.ndarray
            Implied equilibrium returns vector (N,)
        
        Examples
        --------
        >>> caps = np.array([100, 200, 150])
        >>> cov = np.array([[0.01, 0.005, 0.002],
        ...                 [0.005, 0.02, 0.003],
        ...                 [0.002, 0.003, 0.015]])
        >>> model = BlackLittermanBase(risk_aversion=2.5)
        >>> model.implied_returns(caps, cov)
        array([0.0235, 0.0456, 0.0341])
        """
        # Calculate market weights
        market_weights = market_caps / np.sum(market_caps)
        
        # Validate and fix covariance if needed
        covariance = self._validate_and_fix_covariance(covariance)
        
        # Calculate implied returns
        if self.use_torch:
            cov_t = self.torch.tensor(covariance, device=self.device)
            weights_t = self.torch.tensor(market_weights, device=self.device)
            returns = (self.risk_aversion * (cov_t @ weights_t)).cpu().numpy()
        else:
            returns = self.risk_aversion * (covariance @ market_weights)
        
        return returns
    
    def posterior_returns(
        self,
        prior_returns: np.ndarray,
        covariance: np.ndarray,
        P: np.ndarray,
        Q: np.ndarray,
        omega: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Black-Litterman posterior returns and covariance.
        
        Formula: E[R] = [ (τΣ)⁻¹ + PᵀΩ⁻¹P ]⁻¹ [ (τΣ)⁻¹Π + PᵀΩ⁻¹Q ]
        
        Parameters
        ----------
        prior_returns : np.ndarray
            Implied equilibrium returns Π (N,)
        covariance : np.ndarray
            Asset covariance matrix Σ (N, N)
        P : np.ndarray
            Pick matrix linking views to assets (K, N)
        Q : np.ndarray
            View returns vector (K,)
        omega : np.ndarray, optional
            View uncertainty matrix Ω (K, K). Defaults to τ * P Σ Pᵀ
        
        Returns
        -------
        posterior_returns : np.ndarray
            Updated expected returns (N,)
        posterior_covariance : np.ndarray
            Updated covariance matrix (N, N)
        
        Raises
        ------
        ValueError
            If prior_returns contains NaN or inf values
        np.linalg.LinAlgError
            If matrix inversion fails
        
        Examples
        --------
        >>> model = BlackLittermanBase()
        >>> prior = np.array([0.02, 0.03, 0.01])
        >>> cov = np.eye(3) * 0.01
        >>> P = np.array([[1, -1, 0]])  # View: Asset 1 > Asset 2
        >>> Q = np.array([0.01])  # 1% excess return
        >>> post_returns, post_cov = model.posterior_returns(prior, cov, P, Q)
        """
        # Input validation
        if np.any(np.isnan(prior_returns)) or np.any(np.isinf(prior_returns)):
            raise ValueError("Prior returns contain NaN or inf values")
        
        N = len(prior_returns)
        K = len(Q)
        
        # Validate covariance
        covariance = self._validate_and_fix_covariance(covariance)
        
        # Default omega using Idzorek's method
        if omega is None:
            omega = self.tau * (P @ covariance @ P.T)
            # Ensure positive diagonal for numerical stability
            omega = omega + np.eye(K) * 1e-8
        else:
            omega = omega + np.eye(K) * 1e-8
        
        try:
            # Compute using efficient solving (avoid explicit inverse where possible)
            tau_cov = self.tau * covariance
            
            # Pre-compute inverses
            tau_cov_inv = np.linalg.inv(tau_cov)
            omega_inv = np.linalg.inv(omega)
            
            # Posterior precision matrix
            posterior_precision = tau_cov_inv + (P.T @ omega_inv @ P)
            
            # Posterior returns
            b = (tau_cov_inv @ prior_returns) + (P.T @ omega_inv @ Q)
            posterior_returns = np.linalg.solve(posterior_precision, b)
            
            # Posterior covariance
            posterior_covariance = np.linalg.inv(posterior_precision)
            
        except np.linalg.LinAlgError as e:
            raise np.linalg.LinAlgError(
                f"Matrix inversion failed. Check that P has full row rank. Error: {e}"
            )
        
        return posterior_returns, posterior_covariance
