# blitz-Black-Litterman-Improved-with-Zest-Zoom
blitz transforms academic Black-Litterman research into production-ready Python code. Features ML-generated views using random forests, vectorized NumPy operations, full type hints, and reproducibility scripts that exactly replicate paper figures. Install via pip install blitz and get optimal portfolios in milliseconds.
# Black-Litterman Model Improvements

[![PyPI version](https://badge.fury.io/py/black-litterman-improved.svg)](https://badge.fury.io/py/black-litterman-improved)
[![CI](https://github.com/GPPanos/black-litterman-improved/actions/workflows/ci.yml/badge.svg)](https://github.com/GPPanos/black-litterman-improved/actions)
[![Documentation Status](https://readthedocs.org/projects/black-litterman-improved/badge/?version=latest)](https://black-litterman-improved.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

State-of-the-art Black-Litterman portfolio optimization with ML-generated views.

## Features

-  **Vectorized Implementation** - NumPy/SciPy for high performance
-  **ML-Enhanced Views** - Random Forest predictions using size, B/M, momentum, volatility
-  **Academic References** - Full DOIs and citations in docstrings
-  **Reproducible** - Scripts that generate exact paper figures
-  **PyPI Ready** - `pip install black-litterman-improved`
- **Type Hints** - Full type annotations for IDE support
- **Tested** - 94% test coverage with CI/CD

## Quick Start

```python
from black_litterman_improved import BlackLittermanML
import numpy as np

# Generate sample data
np.random.seed(42)
n_days, n_assets = 500, 10
returns = np.random.multivariate_normal(
    mean=np.zeros(n_assets),
    cov=np.eye(n_assets) * 0.01,
    size=n_days
)
prices = 100 * np.exp(np.cumsum(returns, axis=0))
market_caps = np.random.rand(n_assets)

# Create model and predict
model = BlackLittermanML(risk_aversion=2.5)
results = model.predict(prices, market_caps)

print(f"Optimal weights: {results['weights'][:3]}...")
print(f"Expected Sharpe: {results['sharpe_ratio']:.2f}")
