# 🚀 Portfolio Quant Engine

*A comprehensive, modular Quantitative Analysis system designed for portfolio optimization, stress testing, and technical trend analysis. This engine integrates institutional-grade financial modeling with algorithmic trading indicators to provide a data-driven approach to market navigation.*

---

## 📊 Key Features

* **Portfolio Optimization**: Mean-Variance Optimization (MVO) utilizing the `SLSQP` algorithm to maximize Sharpe Ratios and define Efficient Frontiers.
* **Risk Management**: Dynamic VIX-adjusted returns analysis and Rolling Volatility tracking to stabilize risk during market panics.
* **Monte Carlo Projections**:
    * **Forward Projection**: Portfolio equity curve forecasting (Geometric Brownian Motion).
    * **Macro Projection**: 5-year outlooks integrating US 10-Year Treasury Yields.
    * **Gold Engine**: Dynamic Monte Carlo forecasting for precious metals.
* **Algorithmic Screening**:
    * **Sniper Scanner**: Algorithmic confluence detection (RSI + MACD + Stochastic).
    * **Market Cycle Quadrants**: Asset positioning analysis (Full Bull vs. Capitulation).
    * **Holistic Screener**: A definitive "Action Matrix" (Buy/Hold/Sell) based on confluence logic.
* **Technical Intelligence**: Professional dashboard providing SMA (20, 50, 200), RSI, MACD & Stochastic indicators.

---

## 🛠️ Architecture (Separation of Concerns)

The project follows a clean, modular design pattern to ensure maintainability and scalability:

1. **`portfolio_optimizer.py`**: The "Engine." Handles all mathematical computation, financial modeling, and data filtering.
2. **`visualizer.py`**: The "Renderer." Manages the complex Matplotlib/Seaborn visualization pipeline.
3. **`main.py`**: The "Controller." Orchestrates execution, logs terminal reports, and triggers the stress-testing pipeline.
4. **`config.py`**: Centralized configuration for tickers, budgets, and risk thresholds.

---

## ⚙️ Installation & Setup Guide

### 1. Clone the repository
```bash
git clone [https://github.com/your-username/portfolio-quant-engine.git](https://github.com/your-username/portfolio-quant-engine.git)
cd portfolio-quant-engine
```
### Prerequisites
Ensure you have **Python 3.8+** installed on your system. You can download the latest version from [python.org](https://www.python.org/).

---
## 2. Setup (Windows / macOS)

**On Windows (Command Prompt/PowerShell):**
---
1. Create and activate environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate
2.	**Install requirements:**
    ```bash
    pip install yfinance numpy pandas scipy matplotlib seaborn mplfinance
3.  **Run the analysis:**
     ```bash
    python main.py
     
**On macOS:**
---
1.  Create and activate environment:
     ```bash
    python3 -m venv venv
    source venv/bin/activate
2.	**Install requirements:**
     ```bash
    pip install yfinance numpy pandas scipy matplotlib seaborn mplfinance
3.  **Run the analysis:**
     ```bash
    python3 main.py
     
### 💡 Pro Tips for Cross-Platform Compatibility

1.   **File Paths**: If you add functionality to save data (e.g., CSV/Excel files), avoid using hardcoded paths like `C:\Users\Name\data.csv`. Use `os.path.join` to ensure compatibility across all operating systems:
     ```python
         import os
         save_path = os.path.join('data', 'results.csv')

2.   **Matplotlib Backend**: If you experience issues with plot windows not opening on macOS, ensure you have the Tkinter backend installed. You can usually fix this by running brew install python-tk (if using Homebrew) or simply upgrading your libraries:
     ```bash
        pip install --upgrade matplotlib
     
## 📈 Strategy Map

| Strategy | Objective | Methodology |
| :--- | :--- | :--- |
| **Alpha** | Max Sharpe Ratio | Quadratic Programming Optimization |
| **Beta** | Risk Control | VIX-adjusted return scaling |
| **Gamma** | Long-term Outlook | Monte Carlo (GBM) Projection |
| **Delta** | Technical Trend | SMA (20/50/200) & EMA |
| **Epsilon** | Confluence Trading | Sniper Scanner (RSI+MACD+Stoch) |
| **Zeta** | Macro Analysis | Treasury Yield & 5-Year Monte Carlo |

   *Disclaimer: This project is for educational and analytical purposes only. Financial markets involve significant risk. Always perform your own due diligence before making investment decisions.*
