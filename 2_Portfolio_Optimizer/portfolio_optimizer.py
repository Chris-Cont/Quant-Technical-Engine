import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import config
import warnings
warnings.filterwarnings('ignore')

class PortfolioQuantEngine:
    def __init__(self):
        self.old_investments = config.old_investments
        self.stock_hedge_tickers = config.stock_hedge_tickers
        self.macro_hedge_tickers = config.macro_hedge_tickers

        self.old_tickers = list(self.old_investments.keys())
        
        self.all_tickers = list(self.old_investments.keys()) + self.stock_hedge_tickers + self.macro_hedge_tickers
        self.tickers_with_vix = self.all_tickers + ['^VIX']
        
        self.old_amounts = np.array(list(self.old_investments.values()))
        self.total_budget = np.sum(self.old_amounts) + config.stock_budget + config.macro_budget
        
        self.num_stock = len(self.stock_hedge_tickers)
        self.num_macro = len(self.macro_hedge_tickers)
        self.total_new = self.num_stock + self.num_macro

    def fetch_data_and_filter(self):
        print("Λήψη δεδομένων αγοράς (Ιστορικό 6 ετών για Backtest & Stress Test)...")
        data = yf.download(self.tickers_with_vix, period="6y", progress=False)['Close'].dropna()

        stock_data = data[self.all_tickers]
        vix_data = data['^VIX']

        self.all_returns = np.log(stock_data / stock_data.shift(1)).dropna()
        aligned_vix = vix_data.loc[self.all_returns.index]
        self.filtered_returns = self.all_returns[aligned_vix < config.VIX_THRESHOLD]

        self.cov_matrix = self.filtered_returns.cov() * 252 
        self.annual_returns = self.filtered_returns.mean() * 252

        old_weights = self.old_amounts / np.sum(self.old_amounts)
        old_cov_matrix = self.filtered_returns[list(self.old_investments.keys())].cov() * 252
        self.risk_before = np.sqrt(np.dot(old_weights.T, np.dot(old_cov_matrix, old_weights)))
        self.return_before = np.sum(self.annual_returns[list(self.old_investments.keys())] * old_weights)

    def _get_portfolio_metrics(self, new_amounts):
        combined_amounts = np.concatenate((self.old_amounts, new_amounts))
        weights = combined_amounts / self.total_budget
        ret = np.sum(self.annual_returns * weights)
        var = np.dot(weights.T, np.dot(self.cov_matrix, weights))
        vol = np.sqrt(var)
        sharpe = (ret - config.risk_free_rate) / vol
        return ret, vol, sharpe

    def run_optimization(self):
        print("Εκτέλεση Βελτιστοποίησης (Max Sharpe)...")
        def objective_min_vol(new_amounts): return self._get_portfolio_metrics(new_amounts)[1] * 1000
        def objective_max_sharpe(new_amounts): return -self._get_portfolio_metrics(new_amounts)[2]

        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x[:self.num_stock]) - config.stock_budget},
            {'type': 'eq', 'fun': lambda x: np.sum(x[self.num_stock:]) - config.macro_budget}
        )
        bounds = tuple((0, max(config.stock_budget, config.macro_budget)) for _ in range(self.total_new))
        initial_guess = np.array([config.stock_budget / self.num_stock] * self.num_stock + 
                                 [config.macro_budget / self.num_macro] * self.num_macro)

        opt_max_sharpe = minimize(objective_max_sharpe, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        self.opt_amounts = opt_max_sharpe.x
        self.return_after, self.risk_after, self.sharpe_after = self._get_portfolio_metrics(self.opt_amounts)
        self.final_weights = np.concatenate((self.old_amounts, self.opt_amounts)) / self.total_budget

        # Monte Carlo Efficient Frontier
        print("Δημιουργία 15000 προσομοιώσεων και Αποτελεσματικού Συνόρου...")
        num_portfolios = 15000
        self.results = np.zeros((3, num_portfolios))

        for i in range(num_portfolios):
            w_s = np.random.random(self.num_stock); w_s = (w_s / np.sum(w_s)) * config.stock_budget
            w_m = np.random.random(self.num_macro); w_m = (w_m / np.sum(w_m)) * config.macro_budget
            rand_amts = np.concatenate((w_s, w_m))
            r, v, s = self._get_portfolio_metrics(rand_amts)
            self.results[0, i] = v; self.results[1, i] = r; self.results[2, i] = s

        target_returns = np.linspace(self.results[1,:].min(), self.results[1,:].max(), 30)
        self.frontier_vols = []; self.valid_returns = []

        for target in target_returns:
            cons = (
                {'type': 'eq', 'fun': lambda x: np.sum(x[:self.num_stock]) - config.stock_budget},
                {'type': 'eq', 'fun': lambda x: np.sum(x[self.num_stock:]) - config.macro_budget},
                {'type': 'eq', 'fun': lambda x: self._get_portfolio_metrics(x)[0] - target}
            )
            res = minimize(objective_min_vol, initial_guess, method='SLSQP', bounds=bounds, constraints=cons)
            if res.success:
                self.frontier_vols.append(self._get_portfolio_metrics(res.x)[1])
                self.valid_returns.append(target)

    def run_backtests_and_stress(self):
        print("Εκτέλεση Backtesting έναντι του SPY...")
        self.spy_data = yf.download('SPY', period="6y", progress=False)['Close'].dropna().squeeze()
        self.common_dates = self.all_returns.index.intersection(self.spy_data.index)

        recent_dates = self.common_dates[-252:]
        self.port_daily_1y = self.all_returns.loc[recent_dates].dot(self.final_weights)
        self.spy_daily_1y = np.log(self.spy_data.loc[recent_dates] / self.spy_data.shift(1).loc[recent_dates]).dropna()

        self.port_cum_1y = np.exp(self.port_daily_1y.cumsum()) - 1
        self.spy_cum_1y = np.exp(self.spy_daily_1y.cumsum()) - 1

    def max_drawdown(ret_series):
            comp = (ret_series + 1).cumprod()
            return float(((comp / comp.expanding(min_periods=1).max()) - 1).min())

        self.port_mdd_1y = max_drawdown(self.port_daily_1y)
        self.spy_mdd_1y = max_drawdown(self.spy_daily_1y)

        self.port_final_return = float(self.port_cum_1y.iloc[-1])
        self.spy_final_return = float(self.spy_cum_1y.iloc[-1])
        self.port_final_value = self.total_budget * (1 + self.port_final_return)
        self.spy_final_value = self.total_budget * (1 + self.spy_final_return)
        
        # 6. HISTORICAL STRESS TESTING (ΑΚΡΑΙΑ ΣΕΝΑΡΙΑ)
        self.port_all_daily = self.all_returns.loc[self.common_dates].dot(self.final_weights)
        self.spy_all_daily = np.log(self.spy_data.loc[self.common_dates] / self.spy_data.shift(1).loc[self.common_dates]).dropna()
        self.mdd_func = max_drawdown # Save for main.py to use

    def run_forward_monte_carlo(self):
        sim_days = config.sim_years * 252
        mu = self.return_after
        sigma = self.risk_after

        daily_mu = (mu - 0.5 * sigma**2) / 252
        daily_sigma = sigma / np.sqrt(252)

        print(f"Ρίχνουμε τα ζάρια για {config.num_sims} μελλοντικά σενάρια αγοράς...")
        Z = np.random.normal(0, 1, (sim_days, config.num_sims))
        daily_sim_returns = daily_mu + daily_sigma * Z

        self.price_paths = np.zeros((sim_days + 1, config.num_sims))
        self.price_paths[0] = self.total_budget

        for t in range(1, sim_days + 1):
            self.price_paths[t] = self.price_paths[t-1] * np.exp(daily_sim_returns[t-1])

        self.final_sim_values = self.price_paths[-1]

    def prepare_risk_and_correlation(self):
        # Correlation Matrices Prep
        self.old_tickers = list(self.old_investments.keys())
        self.old_corr = self.filtered_returns[self.old_tickers].corr()

        self.bought_new_tickers = [t for i, t in enumerate(self.stock_hedge_tickers + self.macro_hedge_tickers) if self.opt_amounts[i] > 1.00]
        self.final_portfolio_tickers = self.old_tickers + self.bought_new_tickers
        self.new_corr = self.filtered_returns[self.final_portfolio_tickers].corr()

        # Risk Contribution Prep
        self.final_cov_matrix = self.filtered_returns[self.final_portfolio_tickers].cov() * 252
        final_weights_filtered = []
        for ticker in self.final_portfolio_tickers:
            if ticker in self.old_investments:
                final_weights_filtered.append(self.old_investments[ticker] / self.total_budget)
            else:
                idx = (self.stock_hedge_tickers + self.macro_hedge_tickers).index(ticker)
                final_weights_filtered.append(self.opt_amounts[idx] / self.total_budget)

        self.final_weights_filtered = np.array(final_weights_filtered)
        portfolio_vol = np.sqrt(np.dot(self.final_weights_filtered.T, np.dot(self.final_cov_matrix, self.final_weights_filtered)))
        marginal_contrib = np.dot(self.final_cov_matrix, self.final_weights_filtered) / portfolio_vol
        self.risk_contribution_pct = (self.final_weights_filtered * marginal_contrib) / portfolio_vol

    def analyze_vix_sharpe(self):
        """
        STRATEGY A: Regime-Switching Sharpe Ratio Analysis.
        Evaluates the risk-adjusted performance of the Base Portfolio vs the Macro Hedge 
        under different market volatility regimes (Normal vs. High Stress/Panic).
        """
        # Fetch exact VIX dates to match our returns index
        vix_data = yf.download('^VIX', start=self.all_returns.index[0], end=self.all_returns.index[-1], progress=False)['Close']
        self.aligned_vix = vix_data.reindex(self.all_returns.index).ffill()
        
        # Define Volatility Regimes
        vix_normal = self.aligned_vix[self.aligned_vix < config.VIX_THRESHOLD]
        vix_stress = self.aligned_vix[self.aligned_vix >= config.VIX_THRESHOLD]
        
    def calc_sharpe(returns_df, tickers):
            """Internal helper to calculate annualized Sharpe Ratio."""
            valid_tickers = [t for t in tickers if t in returns_df.columns]
            if len(returns_df) == 0 or len(valid_tickers) == 0: return 0
            
            # Equal weight assumption for the benchmark test
            w = np.ones(len(valid_tickers)) / len(valid_tickers)
            port_ret = returns_df[valid_tickers].dot(w)
            
            mean_ret = port_ret.mean() * 252
            volatility = port_ret.std() * np.sqrt(252)
            
            if volatility == 0: return 0
            return (mean_ret - config.risk_free_rate) / volatility

        self.vix_metrics = {
            "Metric (Return/Risk Profile)": [
                "1. Standard Sharpe Ratio (All Market Days)", 
                "2. VIX-Adjusted Sharpe (Calm Markets, VIX < 20)", 
                "3. Stress Sharpe (Panic Markets, VIX >= 20)"
            ],
            "Base Portfolio (Equities)": [
                calc_sharpe(self.all_returns, self.old_tickers),
                calc_sharpe(self.all_returns.loc[vix_normal.index], self.old_tickers),
                calc_sharpe(self.all_returns.loc[vix_stress.index], self.old_tickers)
            ],
            "Macro Hedge (Bonds/Cmdty)": [
                calc_sharpe(self.all_returns, self.macro_hedge_tickers),
                calc_sharpe(self.all_returns.loc[vix_normal.index], self.macro_hedge_tickers),
                calc_sharpe(self.all_returns.loc[vix_stress.index], self.macro_hedge_tickers)
            ]
        }
        self.df_vix_metrics = pd.DataFrame(self.vix_metrics)

    def run_gold_monte_carlo(self):
        """
        STRATEGY B: Dynamic Monte Carlo Projection for Gold (GLD).
        Simulates 1,000 future price paths using Geometric Brownian Motion (GBM) 
        to forecast potential price action until the end of 2026.
        """
        # Identify the correct Gold ticker from the user's config
        self.gold_ticker = 'GLD' if 'GLD' in self.all_returns.columns else ('GC=F' if 'GC=F' in self.all_returns.columns else None)
        
        if self.gold_ticker:
            # Fetch last 1-year history for trend baseline
            gld_prices = yf.download(self.gold_ticker, period="1y", progress=False)['Close'].squeeze()
            self.gld_last_price = float(gld_prices.iloc[-1])
            self.gld_hist_dates = gld_prices.index
            self.gld_hist_prices = gld_prices.values
            
            # Trading days remaining until approximately Dec 31, 2026
            self.days_left_2026 = 210 
            
            # Calculate drift (mu) and volatility (sigma) from recent history
            gld_ret = self.all_returns[self.gold_ticker][-252:]
            mu_gld = gld_ret.mean()
            sigma_gld = gld_ret.std()
            
            sims = 1000
            self.gld_paths = np.zeros((self.days_left_2026 + 1, sims))
            self.gld_paths[0] = self.gld_last_price
            
            # GBM Formula Execution
            for t in range(1, self.days_left_2026 + 1):
                Z = np.random.normal(0, 1, sims)
                self.gld_paths[t] = self.gld_paths[t-1] * np.exp(mu_gld + sigma_gld * Z)
                
            # Extract dynamic confidence intervals (Percentiles)
            self.best_path = np.percentile(self.gld_paths, 90, axis=1)
            self.median_path = np.percentile(self.gld_paths, 50, axis=1)
            self.worst_path = np.percentile(self.gld_paths, 10, axis=1)
            self.future_gld_dates = pd.date_range(start=self.gld_hist_dates[-1], periods=self.days_left_2026 + 1, freq='B')

    def calculate_hedge_correlations(self):
        """
        STRATEGY C: Hedging Correlation Analytics.
        Measures how each candidate hedge asset correlates against the 
        Base Portfolio to identify the absolute best defensive assets.
        """
        # Re-calculate Base Portfolio returns using old weights
        old_w = self.old_amounts / np.sum(self.old_amounts)
        old_portfolio_returns = self.filtered_returns[self.old_tickers].dot(old_w)
        
        correlations = {}
        for ticker in self.stock_hedge_tickers + self.macro_hedge_tickers:
            if ticker in self.filtered_returns.columns:
                corr = np.corrcoef(old_portfolio_returns, self.filtered_returns[ticker])[0, 1]
                correlations[ticker] = corr
                
        # Sort from lowest (best hedge/negative) to highest correlation
        self.sorted_correlations = sorted(correlations.items(), key=lambda item: item[1])

    def run_min_variance_optimization(self):
        """
        STRATEGY D: Global Minimum Variance Optimization.
        Calculates the alternative "Absolute Defense" portfolio that strictly 
        minimizes volatility regardless of the return tradeoff.
        """
        # Objective: Minimize Volatility
    def objective_min_vol(new_amounts): 
            return self._get_portfolio_metrics(new_amounts)[1] * 1000
        
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x[:self.num_stock]) - config.stock_budget},
            {'type': 'eq', 'fun': lambda x: np.sum(x[self.num_stock:]) - config.macro_budget}
        )
        bounds = tuple((0, max(config.stock_budget, config.macro_budget)) for _ in range(self.total_new))
        initial_guess = np.array([config.stock_budget / self.num_stock] * self.num_stock + 
                                 [config.macro_budget / self.num_macro] * self.num_macro)
                                 
        opt_res = minimize(objective_min_vol, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        
        self.min_var_amounts = opt_res.x
        self.min_var_return, self.min_var_risk, self.min_var_sharpe = self._get_portfolio_metrics(self.min_var_amounts)

    def calculate_vix_adjusted_returns(self):
        """
        STRATEGY E: VIX-Adjusted Returns Analysis.
        Demonstrates the penalty of market volatility on the unhedged (Old) portfolio
        by scaling returns inversely to the VIX level.
        """
        # 1. Calculate returns only for the old (Base/Tech) portfolio
        old_w = self.old_amounts / np.sum(self.old_amounts)
        self.old_port_returns = self.all_returns[self.old_tickers].dot(old_w)

        # 2. Fetch and align VIX data
        vix_data = yf.download('^VIX', start=self.all_returns.index[0], end=self.all_returns.index[-1], progress=False)['Close'].squeeze()
        vix_aligned = vix_data.reindex(self.old_port_returns.index).ffill()
        self.scale_factor = vix_aligned.mean()

        # 3. Apply the VIX-adjustment formula
        self.adj_old_returns = (self.old_port_returns / vix_aligned) * self.scale_factor

        self.cum_real_old = (np.exp(self.old_port_returns.cumsum()) - 1) * 100
        self.cum_adj_old = (np.exp(self.adj_old_returns.cumsum()) - 1) * 100

        # 4. Statistical Conclusions
        self.real_vol_old = self.old_port_returns.std() * np.sqrt(252)
        # Unscaled vol for true comparison
        self.adj_vol_old = (self.adj_old_returns / self.scale_factor).std() * np.sqrt(252) 
        
        # Save high VIX days for the visualizer
        self.high_vix_days = vix_aligned[vix_aligned > 25].index


    def calculate_rolling_volatility(self, window=60):
        """
        STRATEGY F: Rolling Volatility Analysis.
        Calculates the 60-day rolling volatility of the Base Portfolio
        to visually and mathematically compare true market risk versus 
        VIX-adjusted risk stabilization (Target Volatility).
        """
        # Note: This relies on calculate_vix_adjusted_returns() having run first
        self.real_rolling_vol = self.old_port_returns.rolling(window).std() * np.sqrt(252) * 100
        self.adj_rolling_vol = self.adj_old_returns.rolling(window).std() * np.sqrt(252) * 100

    def run_monte_carlo_comparison(self):
        """
        STRATEGY G: Monte Carlo Comparison (Actual vs VIX-Adjusted).
        Simulates two portfolios starting from a $10k base to demonstrate 
        how VIX-adjustment narrows the uncertainty cone of future outcomes.
        """
        days_left = 210
        sims = 1000
        start_value = 10000 
        
        # Calculate drift and volatility
        mu_real = self.old_port_returns.mean()
        sigma_real = self.old_port_returns.std()
        mu_adj = self.adj_old_returns.mean()
        sigma_adj = self.adj_old_returns.std()
        
        self.real_paths = np.zeros((days_left + 1, sims))
        self.adj_paths = np.zeros((days_left + 1, sims))
        self.real_paths[0] = start_value
        self.adj_paths[0] = start_value
        
        for t in range(1, days_left + 1):
            # Using same random seed logic for 'fair' comparison
            Z = np.random.normal(0, 1, sims)
            self.real_paths[t] = self.real_paths[t-1] * np.exp(mu_real + sigma_real * Z)
            self.adj_paths[t] = self.adj_paths[t-1] * np.exp(mu_adj + sigma_adj * Z)
            
        self.real_percentiles = np.percentile(self.real_paths, [90, 50, 10], axis=1)
        self.adj_percentiles = np.percentile(self.adj_paths, [90, 50, 10], axis=1)
        self.future_dates = pd.date_range(start=self.old_port_returns.index[-1], periods=days_left + 1, freq='B')
    def calculate_technical_indicators(self, ticker='SPY'):
        """
        TECHNICAL ANALYSIS: SMA & EMA Calculation.
        Computes 20 and 50-period Simple and Exponential Moving Averages
        for trend analysis.
        """
        # Fetch data directly if not in historical data
        price_data = yf.download(ticker, period="1y", progress=False)['Close'].dropna().squeeze()[-252:]
        
        # Calculate Indicators
        self.sma_20 = price_data.rolling(window=20).mean()
        self.sma_50 = price_data.rolling(window=50).mean()
        self.ema_20 = price_data.ewm(span=20, adjust=False).mean()
        self.ema_50 = price_data.ewm(span=50, adjust=False).mean()
        
        self.ta_price_data = price_data # Save for visualizer
        self.ta_ticker = ticker
        
    def run_trend_scanner(self):
        """
        STRATEGY I: Market Trend Scanner.
        Scans all assets for Bullish/Bearish status based on 20/50 SMA rules.
        """
        print("🔍 Scanning portfolio trends...")
        # Get data for all tickers (portfolio + hedge candidates)
        scan_data = yf.download(self.all_tickers, period="6mo", progress=False)['Close']
        
        self.bullish, self.bearish, self.consolidation = [], [], []
        
        for ticker in self.all_tickers:
            try:
                prices = scan_data[ticker].dropna()
                if len(prices) < 50: continue
                
                last_price = prices.iloc[-1]
                sma_20 = prices.rolling(window=20).mean().iloc[-1]
                sma_50 = prices.rolling(window=50).mean().iloc[-1]
                
                if last_price > sma_20 and last_price > sma_50:
                    self.bullish.append((ticker, last_price, sma_20, sma_50))
                elif last_price < sma_20 and last_price < sma_50:
                    self.bearish.append((ticker, last_price, sma_20, sma_50))
                else:
                    self.consolidation.append((ticker, last_price, sma_20, sma_50))
            except:
                continue
