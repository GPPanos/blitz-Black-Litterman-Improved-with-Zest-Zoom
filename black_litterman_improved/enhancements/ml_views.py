"""
Machine learning-enhanced view generation for Black-Litterman.

This module implements ML predictions based on factor exposures to generate
views for the Black-Litterman model.

References:
    Ko, H., & Lee, J. (2025). Portfolio Management Transformed: 
    An Enhanced Black–Litterman Approach Integrating Asset Pricing 
    Theory and Machine Learning. Computational Economics, 66, 3841-3887.
    DOI: 10.1007/s10614-024-10922-x

Author: GPPanos
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple, Dict, Any
import warnings

from ..core.black_litterman import BlackLittermanBase


class BlackLittermanML(BlackLittermanBase):
    """
    Black-Litterman with ML-generated views using factor predictions.
    
    This model integrates machine learning predictions based on factor
    exposures (size, book-to-market, momentum, volatility) into the
    Black-Litterman framework.
    
    Parameters
    ----------
    risk_aversion : float, default=2.5
        Risk aversion coefficient (λ)
    tau : float, default=0.025
        Uncertainty scalar for prior
    ml_model : object, optional
        Scikit-learn compatible regressor. Defaults to RandomForestRegressor.
    lookback_days : int, default=252
        Number of days for training ML model (1 trading year)
    
    Examples
    --------
    >>> import numpy as np
    >>> from black_litterman_improved import BlackLittermanML
    >>> 
    >>> prices = np.random.randn(500, 10).cumsum(axis=0) + 100
    >>> market_caps = np.random.rand(10)
    >>> 
    >>> model = BlackLittermanML(risk_aversion=2.5)
    >>> results = model.predict(prices, market_caps)
    >>> print(f"Sharpe ratio: {results['sharpe_ratio']:.2f}")
    """
    
    def __init__(
        self,
        risk_aversion: float = 2.5,
        tau: float = 0.025,
        ml_model: Optional[Any] = None,
        lookback_days: int = 252
    ):
        super().__init__(risk_aversion, tau)
        self.ml_model = ml_model or RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.lookback_days = lookback_days
        self._is_fitted = False
    
    def generate_ml_views(
        self,
        price_data: np.ndarray,
        factors: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate views using ML predictions on factors.
        
        Parameters
        ----------
        price_data : np.ndarray
            Historical prices (T, N) where T is time steps, N is assets
        factors : np.ndarray, optional
            Factor exposures (size, B/M, momentum, volatility). 
            Shape (T, N, 4) or (T, N). If None, computed automatically.
        
        Returns
        -------
        P : np.ndarray
            Pick matrix (N, N) - identity for asset-level views
        Q : np.ndarray
            Predicted returns vector (N,)
        omega : np.ndarray
            View uncertainty matrix (N, N)
        """
        T, N = price_data.shape
        
        # Calculate historical returns
        returns = np.diff(np.log(price_data), axis=0)
        
        # If no factors provided, create synthetic ones
        if factors is None:
            factors = self._compute_factors(returns)
        elif factors.ndim == 3:
            # Aggregate factor dimensions
            factors = factors.mean(axis=2)
        
        # Train ML model for each asset
        predictions = np.zeros(N)
        uncertainties = np.zeros(N)
        
        for i in range(N):
            # Prepare features (lagged returns + factors)
            X = np.column_stack([
                returns[:-1, i],  # lagged return
                factors[:-1, i] if factors.ndim == 2 else factors[:-1]
            ])
            y = returns[1:, i]  # next period return
            
            # Remove NaN rows
            mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X_clean, y_clean = X[mask], y[mask]
            
            if len(X_clean) > 10:  # Minimum samples for training
                # Standardize and train
                X_scaled = self.scaler.fit_transform(X_clean)
                self.ml_model.fit(X_scaled, y_clean)
                
                # Predict next return
                if len(X) > 0:
                    last_features = X[-1:].reshape(1, -1)
                    last_scaled = self.scaler.transform(last_features)
                    predictions[i] = float(self.ml_model.predict(last_scaled)[0])
                else:
                    predictions[i] = np.nanmean(returns[:, i])
                
                # Uncertainty as prediction std dev (10% of historical vol)
                uncertainties[i] = np.std(y_clean) * 0.1
                self._is_fitted = True
            else:
                # Fallback to historical mean
                predictions[i] = np.nanmean(returns[:, i])
                uncertainties[i] = np.std(returns[:, i])
                warnings.warn(
                    f"Insufficient data for asset {i}, using historical mean",
                    UserWarning
                )
        
        # Create pick matrix (identity for asset-level views)
        P = np.eye(N)
        Q = predictions
        omega = np.diag(uncertainties ** 2)
        
        return P, Q, omega
    
    def _compute_factors(self, returns: np.ndarray) -> np.ndarray:
        """
        Compute factor exposures for ML model.
        
        Factors computed:
        1. Size proxy (returns magnitude)
        2. B/M proxy (negative returns)
        3. Momentum (20-day moving average)
        4. Volatility (20-day rolling std)
        
        Parameters
        ----------
        returns : np.ndarray
            Historical returns (T, N)
        
        Returns
        -------
        np.ndarray
            Factor exposures (T, N)
        """
        T, N = returns.shape
        
        # Initialize factor arrays
        momentum_factor = np.zeros((T, N))
        volatility_factor = np.zeros((T, N))
        
        for i in range(N):
            # Momentum (past 20 days average return)
            for t in range(20, T):
                momentum_factor[t, i] = np.mean(returns[t-20:t, i])
            
            # Volatility (20-day rolling standard deviation)
            for t in range(20, T):
                volatility_factor[t, i] = np.std(returns[t-20:t, i])
        
        # Combine factors
        # Size proxy = absolute returns
        size_factor = np.abs(returns)
        
        # B/M proxy = negative returns (value stocks have higher returns)
        bm_factor = -returns
        
        # Aggregate to single factor score (equal weighting)
        factors = (
            size_factor + 
            bm_factor + 
            momentum_factor + 
            volatility_factor
        ) / 4
        
        # Normalize to zero mean, unit variance
        factors = (factors - np.mean(factors, axis=0)) / (np.std(factors, axis=0) + 1e-8)
        
        return factors
    
    def predict(self, price_data: np.ndarray, market_caps: np.ndarray) -> Dict[str, Any]:
        """
        Full pipeline: compute posterior returns and optimal weights.
        
        Parameters
        ----------
        price_data : np.ndarray
            Historical prices (T, N) where T is time steps, N is assets
        market_caps : np.ndarray
            Market capitalization for each asset (N,)
        
        Returns
        -------
        dict
            Dictionary containing:
            - posterior_returns: Updated expected returns (N,)
            - posterior_covariance: Updated covariance matrix (N, N)
            - weights: Optimal portfolio weights (N,)
            - ml_predictions: ML model predictions (N,)
            - sharpe_ratio: Expected Sharpe ratio of optimal portfolio
        
        Examples
        --------
        >>> prices = np.random.randn(500, 10).cumsum(axis=0) + 100
        >>> caps = np.random.rand(10)
        >>> model = BlackLittermanML()
        >>> results = model.predict(prices, caps)
        >>> results['weights'].shape
        (10,)
        """
        # Calculate returns and covariance
        returns = np.diff(np.log(price_data), axis=0)
        covariance = np.cov(returns.T)
        
        # Generate ML views
        P, Q, omega = self.generate_ml_views(price_data)
        
        # Calculate prior returns (equilibrium)
        prior = self.implied_returns(market_caps, covariance)
        
        # Compute posterior
        posterior_returns, posterior_covariance = self.posterior_returns(
            prior, covariance, P, Q, omega
        )
        
        # Compute optimal weights (mean-variance optimization)
        try:
            weights = np.linalg.solve(
                self.risk_aversion * posterior_covariance + np.eye(len(posterior_covariance)) * 1e-8,
                posterior_returns
            )
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse
            weights = np.linalg.pinv(self.risk_aversion * posterior_covariance) @ posterior_returns
        
        # Apply constraints
        weights = np.maximum(weights, 0)  # Long-only
        weights = weights / np.sum(weights)  # Normalize to sum to 1
        
        # Calculate expected Sharpe ratio
        expected_return = posterior_returns @ weights
        expected_volatility = np.sqrt(weights @ posterior_covariance @ weights)
        sharpe_ratio = expected_return / expected_volatility if expected_volatility > 0 else 0.0
        
        return {
            'posterior_returns': posterior_returns,
            'posterior_covariance': posterior_covariance,
            'weights': weights,
            'ml_predictions': Q,
            'sharpe_ratio': sharpe_ratio,
        }
