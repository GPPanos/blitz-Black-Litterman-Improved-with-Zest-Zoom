# Black-Litterman Model Improvements

Welcome to the Black-Litterman Model Improvements documentation!

## Overview

This library provides state-of-the-art implementations of the Black-Litterman model with various enhancements including:

- **ML-generated views** using random forests and factor exposures
- **LLM-generated views** (coming soon)
- **Qualitative constraints** (coming soon)

## Mathematical Foundation

The Black-Litterman model combines equilibrium returns with investor views:

$$\mathbb{E}[R] = [(\tau\Sigma)^{-1} + P^T\Omega^{-1}P]^{-1}[(\tau\Sigma)^{-1}\Pi + P^T\Omega^{-1}Q]$$

Where:
- $\Pi$ = Implied equilibrium returns
- $\Sigma$ = Covariance matrix
- $P$ = Pick matrix
- $Q$ = View returns
- $\Omega$ = View uncertainty
- $\tau$ = Scalar uncertainty

## Quick Example

```python
from black_litterman_improved import BlackLittermanML
import numpy as np

# Data
prices = np.random.randn(500, 10).cumsum(axis=0) + 100
caps = np.random.rand(10)

# Model
model = BlackLittermanML(risk_aversion=2.5)
results = model.predict(prices, caps)

print(f"Optimal Sharpe: {results['sharpe_ratio']:.2f}")
