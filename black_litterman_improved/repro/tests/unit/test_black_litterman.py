"""
Unit tests for core Black-Litterman implementation.

Author: GPPanos
"""

import pytest
import numpy as np
from black_litterman_improved.core.black_litterman import BlackLittermanBase


class TestBlackLittermanBase:
    """Test suite for base Black-Litterman model."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.model = BlackLittermanBase(risk_aversion=2.5, tau=0.025)
        
        # Create test data
        np.random.seed(42)
        self.n_assets = 5
        self.market_caps = np.random.rand(self.n_assets)
        self.covariance = np.random.randn(self.n_assets, self.n_assets)
        self.covariance = self.covariance @ self.covariance.T + np.eye(self.n_assets) * 0.1
        self.prior = np.random.randn(self.n_assets)
        
    def test_implied_returns_shape(self):
        """Test implied returns shape and calculation."""
        returns = self.model.implied_returns(self.market_caps, self.covariance)
        assert returns.shape == (self.n_assets,)
        assert not np.any(np.isnan(returns))
    
    def test_implied_returns_positive_weights(self):
        """Test that implied returns are positive for positive weights."""
        returns = self.model.implied_returns(self.market_caps, self.covariance)
        # With positive covariance and positive weights, returns should be positive
        assert np.all(returns > 0)
    
    def test_posterior_returns_shape(self):
        """Test posterior returns calculation."""
        P = np.eye(self.n_assets)
        Q = np.random.randn(self.n_assets)
        
        post_returns, post_cov = self.model.posterior_returns(
            self.prior, self.covariance, P, Q
        )
        
        assert post_returns.shape == (self.n_assets,)
        assert post_cov.shape == (self.n_assets, self.n_assets)
    
    def test_posterior_covariance_psd(self):
        """Test that posterior covariance is positive semi-definite."""
        P = np.eye(self.n_assets)
        Q = np.random.randn(self.n_assets)
        
        _, post_cov = self.model.posterior_returns(
            self.prior, self.covariance, P, Q
        )
        
        # Check eigenvalues are non-negative
        eigenvalues = np.linalg.eigvals(post_cov)
        assert np.all(eigenvalues >= -1e-8)
    
    def test_view_impact(self):
        """Test that views correctly impact posterior returns."""
        # Create bullish view on first asset
        P = np.array([[1, 0, 0, 0, 0]])
        Q = np.array([0.05])  # 5% expected return
        
        post_returns, _ = self.model.posterior_returns(
            self.prior, self.covariance, P, Q
        )
        
        # First asset should have higher return than prior
        assert post_returns[0] > self.prior[0]
    
    def test_nan_handling(self):
        """Test that NaNs raise appropriate errors."""
        prior_with_nan = self.prior.copy()
        prior_with_nan[0] = np.nan
        
        P = np.eye(self.n_assets)
        Q = np.random.randn(self.n_assets)
        
        with pytest.raises(ValueError, match="NaN or inf"):
            self.model.posterior_returns(prior_with_nan, self.covariance, P, Q)
    
    def test_covariance_fixing(self):
        """Test covariance matrix correction for non-PSD matrices."""
        # Create non-PSD matrix (all zeros)
        bad_cov = np.zeros((3, 3))
        fixed = self.model._validate_and_fix_covariance(bad_cov)
        
        # Should be corrected to PSD
        eigenvalues = np.linalg.eigvals(fixed)
        assert np.all(eigenvalues >= -1e-7)
    
    def test_single_asset_case(self):
        """Test single asset scenario."""
        model = BlackLittermanBase()
        caps = np.array([100.0])
        cov = np.array([[0.01]])
        
        returns = model.implied_returns(caps, cov)
        assert returns.shape == (1,)
        assert returns[0] > 0
    
    def test_many_views(self):
        """Test with more views than assets."""
        n_views = 10
        P = np.random.randn(n_views, self.n_assets)
        Q = np.random.randn(n_views)
        
        # Should still work
        post_returns, post_cov = self.model.posterior_returns(
            self.prior, self.covariance, P, Q
        )
        
        assert post_returns.shape == (self.n_assets,)
        assert post_cov.shape == (self.n_assets, self.n_assets)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def setup_method(self):
        self.model = BlackLittermanBase()
        
    def test_extreme_covariance_values(self):
        """Test with very large covariance values."""
        caps = np.array([100, 200])
        cov = np.array([[1e10, 0], [0, 1e10]])
        returns = self.model.implied_returns(caps, cov)
        assert np.all(np.isfinite(returns))
    
    def test_highly_correlated_assets(self):
        """Test with highly correlated assets."""
        caps = np.array([100, 100, 100])
        cov = np.ones((3, 3)) * 0.95
        np.fill_diagonal(cov, 0.1)
        returns = self.model.implied_returns(caps, cov)
        assert np.all(np.isfinite(returns))
    
    def test_zero_market_cap(self):
        """Test with zero market cap."""
        caps = np.array([100, 0, 50])
        cov = np.eye(3) * 0.01
        returns = self.model.implied_returns(caps, cov)
        # Asset with zero cap should have zero weight
        assert np.isfinite(returns[1])
