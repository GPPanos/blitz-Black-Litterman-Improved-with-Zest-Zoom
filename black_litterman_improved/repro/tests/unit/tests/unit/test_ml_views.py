"""
Unit tests for ML-enhanced Black-Litterman.

Author: GPPanos
"""

import pytest
import numpy as np
from black_litterman_improved.enhancements.ml_views import BlackLittermanML


class TestBlackLittermanML:
    """Test suite for ML-enhanced Black-Litterman model."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n_assets = 5
        self.n_days = 100
        self.prices = 100 + np.random.randn(self.n_days, self.n_assets).cumsum(axis=0)
        self.market_caps = np.random.rand(self.n_assets)
        self.model = BlackLittermanML(risk_aversion=2.5, tau=0.025)
    
    def test_predict_returns_dict(self):
        """Test that predict returns all required keys."""
        results = self.model.predict(self.prices, self.market_caps)
        expected_keys = {
            'posterior_returns', 
            'posterior_covariance', 
            'weights', 
            'ml_predictions', 
            'sharpe_ratio'
        }
        assert expected_keys.issubset(results.keys())
    
    def test_weights_sum_to_one(self):
        """Test that portfolio weights sum to approximately 1."""
        results = self.model.predict(self.prices, self.market_caps)
        weights = results['weights']
        assert np.isclose(weights.sum(), 1.0, atol=1e-6)
    
    def test_weights_non_negative(self):
        """Test that weights are non-negative (long-only constraint)."""
        results = self.model.predict(self.prices, self.market_caps)
        weights = results['weights']
        assert np.all(weights >= 0)
    
    def test_sharpe_ratio_positive(self):
        """Test that Sharpe ratio is positive for reasonable data."""
        results = self.model.predict(self.prices, self.market_caps)
        assert results['sharpe_ratio'] > 0
    
    def test_ml_predictions_finite(self):
        """Test that ML predictions are all finite numbers."""
        results = self.model.predict(self.prices, self.market_caps)
        assert np.all(np.isfinite(results['ml_predictions']))
    
    def test_posterior_covariance_psd(self):
        """Test that posterior covariance is positive semi-definite."""
        results = self.model.predict(self.prices, self.market_caps)
        eigenvalues = np.linalg.eigvals(results['posterior_covariance'])
        assert np.all(eigenvalues >= -1e-8)
    
    def test_different_risk_aversion(self):
        """Test that risk aversion parameter affects results."""
        model_low = BlackLittermanML(risk_aversion=1.0)
        model_high = BlackLittermanML(risk_aversion=5.0)
        
        results_low = model_low.predict(self.prices, self.market_caps)
        results_high = model_high.predict(self.prices, self.market_caps)
        
        # Different risk aversion should produce different weights
        assert not np.allclose(results_low['weights'], results_high['weights'], atol=1e-3)
    
    def test_ml_model_custom(self):
        """Test with custom ML model."""
        from sklearn.linear_model import Ridge
        
        custom_model = Ridge(alpha=1.0)
        model = BlackLittermanML(ml_model=custom_model)
        
        results = model.predict(self.prices, self.market_caps)
        assert 'weights' in results
    
    def test_short_history(self):
        """Test with very short price history."""
        short_prices = self.prices[:20, :]  # Only 20 days
        results = self.model.predict(short_prices, self.market_caps)
        assert np.all(np.isfinite(results['weights']))
    
    def test_single_asset(self):
        """Test with single asset."""
        single_prices = self.prices[:, :1]
        single_caps = self.market_caps[:1]
        
        results = self.model.predict(single_prices, single_caps)
        
        # With one asset, weight should be 1.0
        assert np.isclose(results['weights'][0], 1.0, atol=1e-6)
