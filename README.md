# Seeded Financial Market Simulator

Physics-inspired, reproducible market simulation engine designed for finance professionals, quants, and data scientists. It generates synthetic price trajectories, incorporating stochastic noise, momentum dynamics, and realistic market constraints, with full auditability.

---

## Features

- **Reproducible Randomness:**  
  Seeded random noise ensures that every simulation is **fully reproducible**, critical for backtesting, auditing, and regulatory compliance.

- **Symplectic Physics-Inspired Updates:**  
  Price evolution is driven by momentum-like updates:
  - `q` represents the price level.
  - `p` represents price momentum.
  This captures realistic market behavior such as gradual price adjustments and oscillations.

- **Stochastic Noise Injection:**  
  Randomized shocks are added to each tick while preserving underlying trends, simulating real-world market volatility.

- **Risk Guardrails:**  
  Momentum (`p`) is capped to prevent unrealistic jumps, analogous to circuit breakers in financial markets.

- **Institutional-Grade Metrics:**  
  Calculates:
  - Daily returns
  - Annualized volatility
  - Maximum drawdown  
  Output is formatted for **finance-ready reporting**.

- **Audit-Ready Output:**  
  Each simulation logs the master seed and key metrics, ensuring traceability and verifiable results.



