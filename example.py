#!/usr/bin/env python3
"""
Quick example of using the Black-Litterman Improved library.

Author: GPPanos
"""

from black_litterman_improved import BlackLittermanML
import numpy as np

def main():
    print("=" * 60)
    print("Black-Litterman Improved Library - Example")
    print("=" * 60)
    
    # Generate realistic price data (500 days, 10 assets)
    print("\n📊 Generating synthetic market data...")
    np.random.seed(42)
    n_days, n_assets = 500, 10
    
    # Create correlated returns
    returns = np.random.multivariate_normal(
        mean=np.zeros(n_assets),
        cov=np.eye(n_assets) * 0.01,
        size=n_days
    )
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    
    # Market caps (e.g., company sizes)
    market_caps = np.array([100, 50, 30, 20, 15, 10, 8, 5, 3, 2])
    
    # Initialize model
    print("🤖 Initializing ML-enhanced Black-Litterman model...")
    model = BlackLittermanML(
        risk_aversion=2.5,
        tau=0.025,
    )
    
    # Get predictions
    print("📈 Computing posterior returns and optimal weights...")
    results = model.predict(prices, market_caps)
    
    # Display results
    print("\n" + "=" * 60)
    print("📊 Results")
    print("=" * 60)
    
    print("\n📈 Optimal Portfolio Weights:")
    for i, w in enumerate(results['weights'][:5]):  # Show first 5
        print(f"   Asset {i+1}: {w:.2%}")
    if n_assets > 5:
        print(f"   ... and {n_assets - 5} more assets")
    
    print(f"\n📊 Portfolio Statistics:")
    print(f"   Expected Annual Return: {results['posterior_returns'].mean():.2%}")
    print(f"   Portfolio Volatility: {np.sqrt(results['weights'] @ results['posterior_covariance'] @ results['weights']):.2%}")
    print(f"   Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    
    print(f"\n🤖 ML Model Predictions (first 5 assets):")
    for i, pred in enumerate(results['ml_predictions'][:5]):
        print(f"   Asset {i+1}: {pred:.4%}")
    
    print("\n" + "=" * 60)
    print("✅ Example completed successfully!")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    main()
