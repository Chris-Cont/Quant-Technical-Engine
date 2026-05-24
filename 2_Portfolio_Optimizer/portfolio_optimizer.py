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
        
        self.port_all_daily = self.all_returns.loc[self.common_dates].dot(self.final_weights)
        self.spy_all_daily = np.log(self.spy_data.loc[self.common_dates] / self.spy_data.shift(1).loc[self.common_dates]).dropna()
        self.mdd_func = max_drawdown

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
        self.old_corr = self.filtered_returns[self.old_tickers].corr()
        self.bought_new_tickers = [t for i, t in enumerate(self.stock_hedge_tickers + self.macro_hedge_tickers) if self.opt_amounts[i] > 1.00]
        self.final_portfolio_tickers = self.old_tickers + self.bought_new_tickers
        self.new_corr = self.filtered_returns[self.final_portfolio_tickers].corr()

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
        vix_data = yf.download('^VIX', start=self.all_returns.index[0], end=self.all_returns.index[-1], progress=False)['Close']
        self.aligned_vix = vix_data.reindex(self.all_returns.index).ffill()
        vix_normal = self.aligned_vix[self.aligned_vix < config.VIX_THRESHOLD]
        vix_stress = self.aligned_vix[self.aligned_vix >= config.VIX_THRESHOLD]
        
        def calc_sharpe(returns_df, tickers):
            valid_tickers = [t for t in tickers if t in returns_df.columns]
            if len(returns_df) == 0 or len(valid_tickers) == 0: return 0
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
        self.gold_ticker = 'GLD' if 'GLD' in self.all_returns.columns else ('GC=F' if 'GC=F' in self.all_returns.columns else None)
        if self.gold_ticker:
            gld_prices = yf.download(self.gold_ticker, period="1y", progress=False)['Close'].squeeze()
            self.gld_last_price = float(gld_prices.iloc[-1])
            self.gld_hist_dates = gld_prices.index
            self.gld_hist_prices = gld_prices.values
            self.days_left_2026 = 210 
            gld_ret = self.all_returns[self.gold_ticker][-252:]
            mu_gld = gld_ret.mean()
            sigma_gld = gld_ret.std()
            sims = 1000
            self.gld_paths = np.zeros((self.days_left_2026 + 1, sims))
            self.gld_paths[0] = self.gld_last_price
            for t in range(1, self.days_left_2026 + 1):
                Z = np.random.normal(0, 1, sims)
                self.gld_paths[t] = self.gld_paths[t-1] * np.exp(mu_gld + sigma_gld * Z)
            self.best_path = np.percentile(self.gld_paths, 90, axis=1)
            self.median_path = np.percentile(self.gld_paths, 50, axis=1)
            self.worst_path = np.percentile(self.gld_paths, 10, axis=1)
            self.future_gld_dates = pd.date_range(start=self.gld_hist_dates[-1], periods=self.days_left_2026 + 1, freq='B')

    def calculate_hedge_correlations(self):
        old_w = self.old_amounts / np.sum(self.old_amounts)
        old_portfolio_returns = self.filtered_returns[self.old_tickers].dot(old_w)
        correlations = {}
        for ticker in self.stock_hedge_tickers + self.macro_hedge_tickers:
            if ticker in self.filtered_returns.columns:
                corr = np.corrcoef(old_portfolio_returns, self.filtered_returns[ticker])[0, 1]
                correlations[ticker] = corr
        self.sorted_correlations = sorted(correlations.items(), key=lambda item: item[1])

    def run_min_variance_optimization(self):
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
        old_w = self.old_amounts / np.sum(self.old_amounts)
        self.old_port_returns = self.all_returns[self.old_tickers].dot(old_w)
        vix_data = yf.download('^VIX', start=self.all_returns.index[0], end=self.all_returns.index[-1], progress=False)['Close'].squeeze()
        vix_aligned = vix_data.reindex(self.old_port_returns.index).ffill()
        self.scale_factor = vix_aligned.mean()
        self.adj_old_returns = (self.old_port_returns / vix_aligned) * self.scale_factor
        self.cum_real_old = (np.exp(self.old_port_returns.cumsum()) - 1) * 100
        self.cum_adj_old = (np.exp(self.adj_old_returns.cumsum()) - 1) * 100
        self.real_vol_old = self.old_port_returns.std() * np.sqrt(252)
        self.adj_vol_old = (self.adj_old_returns / self.scale_factor).std() * np.sqrt(252) 
        self.high_vix_days = vix_aligned[vix_aligned > 25].index

    def calculate_rolling_volatility(self, window=60):
        self.real_rolling_vol = self.old_port_returns.rolling(window).std() * np.sqrt(252) * 100
        self.adj_rolling_vol = self.adj_old_returns.rolling(window).std() * np.sqrt(252) * 100

    def run_monte_carlo_comparison(self):
        days_left = 210
        sims = 1000
        start_value = 10000 
        mu_real = self.old_port_returns.mean()
        sigma_real = self.old_port_returns.std()
        mu_adj = self.adj_old_returns.mean()
        sigma_adj = self.adj_old_returns.std()
        self.real_paths = np.zeros((days_left + 1, sims))
        self.adj_paths = np.zeros((days_left + 1, sims))
        self.real_paths[0] = start_value
        self.adj_paths[0] = start_value
        for t in range(1, days_left + 1):
            Z = np.random.normal(0, 1, sims)
            self.real_paths[t] = self.real_paths[t-1] * np.exp(mu_real + sigma_real * Z)
            self.adj_paths[t] = self.adj_paths[t-1] * np.exp(mu_adj + sigma_adj * Z)
        self.real_percentiles = np.percentile(self.real_paths, [90, 50, 10], axis=1)
        self.adj_percentiles = np.percentile(self.adj_paths, [90, 50, 10], axis=1)
        self.future_dates = pd.date_range(start=self.old_port_returns.index[-1], periods=days_left + 1, freq='B')

    def calculate_technical_indicators(self, ticker='SPY'):
        price_data = yf.download(ticker, period="1y", progress=False)['Close'].dropna().squeeze()[-252:]
        self.sma_20 = price_data.rolling(window=20).mean()
        self.sma_50 = price_data.rolling(window=50).mean()
        self.ema_20 = price_data.ewm(span=20, adjust=False).mean()
        self.ema_50 = price_data.ewm(span=50, adjust=False).mean()
        self.ta_price_data = price_data 
        self.ta_ticker = ticker
        
    def run_trend_scanner(self):
        print("🔍 Scanning portfolio trends...")
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
                
        # --- COMPUTE DISTANCES FOR QUADRANT PLOT (TΩΡΑ ΕΙΝΑΙ ΣΩΣΤΑ ΣΤΟΙΧΙΣΜΕΝΟ) ---
        all_scanned_assets = self.bullish + self.bearish + self.consolidation
        plot_data = []
        for t, p, s20, s50 in all_scanned_assets:
            dist_20 = ((p / s20) - 1) * 100 
            dist_50 = ((p / s50) - 1) * 100 
            plot_data.append({'Ticker': t, 'Dist_SMA50': dist_50, 'Dist_SMA20': dist_20})
            
        self.df_quadrants = pd.DataFrame(plot_data)                
                
    def calculate_pro_dashboard_indicators(self, ticker='SPY'):
        """
        STRATEGY K: Pro Trading Dashboard (Momentum & Trend).
        Calculates SMA (20,50,200), RSI (14), Stochastic (14,3,3), and MACD (12,26,9)
        to evaluate the macro trend and micro momentum of a specific asset.
        """
        # Fetch 2 years of data to properly calculate the 200-day SMA
        df = yf.download(ticker, period="2y", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

        # 1. Moving Averages (Trend)
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()

        # 2. RSI 14 (Momentum)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI_14'] = 100 - (100 / (1 + rs))

        # 3. Stochastic 14,3,3 (Reversion)
        low_min = df['Low'].rolling(window=14).min()
        high_max = df['High'].rolling(window=14).max()
        df['Stoch_K_raw'] = 100 * ((df['Close'] - low_min) / (high_max - low_min))
        df['Stoch_K'] = df['Stoch_K_raw'].rolling(window=3).mean()
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()

        # 4. MACD 12,26,9 (Trend/Momentum Crossover)
        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema_12 - ema_26
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['Signal_Line']

        # Save the last 252 trading days (1 year) for the visualizer
        self.pro_dash_data = df.iloc[-252:].copy()
        self.pro_dash_ticker = ticker
    def run_sniper_scanner(self):
        """
        STRATEGY L: The Sniper Scanner (Algorithmic Confluence).
        Scans all portfolio and hedge candidates for the ultimate 'Buy Setup' 
        using RSI, MACD Histogram, and Stochastic Crossover confluence.
        """
        print("🎯 Sniper Scanner active: Searching for optimal confluence setups...")
        scan_data = yf.download(self.all_tickers, period="1y", progress=False)

        # Handle MultiIndex for multiple tickers
        if isinstance(scan_data.columns, pd.MultiIndex):
            close_data = scan_data['Close']
            high_data = scan_data['High']
            low_data = scan_data['Low']
        else:
            close_data = pd.DataFrame({self.all_tickers[0]: scan_data['Close']})
            high_data = pd.DataFrame({self.all_tickers[0]: scan_data['High']})
            low_data = pd.DataFrame({self.all_tickers[0]: scan_data['Low']})

        self.sniper_buy_signals = []
        self.sniper_watch_list = []

        for ticker in self.all_tickers:
            try:
                df = pd.DataFrame({
                    'Close': close_data[ticker],
                    'High': high_data[ticker],
                    'Low': low_data[ticker]
                }).dropna()
                
                if len(df) < 200: continue
                
                df['SMA_200'] = df['Close'].rolling(window=200).mean()
                
                # RSI 14
                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df['RSI'] = 100 - (100 / (1 + rs))
                
                # Stochastic 14,3,3
                low_min = df['Low'].rolling(window=14).min()
                high_max = df['High'].rolling(window=14).max()
                df['Stoch_K_raw'] = 100 * ((df['Close'] - low_min) / (high_max - low_min))
                df['Stoch_K'] = df['Stoch_K_raw'].rolling(window=3).mean()
                df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
                
                # MACD 12,26,9
                ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
                ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
                df['MACD'] = ema_12 - ema_26
                df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
                df['MACD_Hist'] = df['MACD'] - df['Signal']
                
                # Execution Logic (Today vs Yesterday)
                last = df.iloc[-1]
                prev = df.iloc[-2] 
                
                # Confluence Conditions
                cond_rsi = last['RSI'] < 45 
                cond_stoch = (last['Stoch_K'] > last['Stoch_D']) and (prev['Stoch_K'] <= prev['Stoch_D']) and (last['Stoch_K'] < 40)
                cond_macd = (last['MACD_Hist'] > prev['MACD_Hist']) 
                
                if cond_rsi and cond_stoch and cond_macd:
                    status = "STRONG BUY (Dip in Uptrend)" if last['Close'] > last['SMA_200'] else "BOTTOM FISHING (High Risk)"
                    self.sniper_buy_signals.append((ticker, status, last['Close'], last['RSI'], last['Stoch_K']))
                elif cond_rsi and (last['Stoch_K'] < 20):
                    self.sniper_watch_list.append((ticker, "Oversold - Awaiting Stoch Crossover", last['Close'], last['RSI']))
                    
            except Exception:
                continue
                
    def run_macro_monte_carlo(self, ticker='SPY', sim_years=5):
        """
        STRATEGY M: Macro Dashboard & 5-Year Monte Carlo.
        Fetches 10-year historical data for a specific asset and the US 10-Year Treasury Yield (^TNX)
        to run a long-term (5-year) Monte Carlo projection alongside macro interest rate tracking.
        """
        print(f"🌍 Running {sim_years}-Year Macro Monte Carlo & Fed Yield analysis for {ticker}...")
        
        # 1. Download 10 years of data for statistical significance
        macro_data = yf.download([ticker, '^TNX'], period="10y", progress=False)['Close'].dropna()
        
        if ticker not in macro_data.columns or '^TNX' not in macro_data.columns:
            print("⚠️ Warning: Could not fetch data for Macro Monte Carlo.")
            return

        stock_prices = macro_data[ticker]
        self.macro_rates = macro_data['^TNX'] # US 10-Year Yield
        
        # 2. Monte Carlo Math Setup
        log_returns = np.log(stock_prices / stock_prices.shift(1)).dropna()
        mu = log_returns.mean()
        sigma = log_returns.std()
        self.macro_last_price = stock_prices.iloc[-1]
        
        sim_days = sim_years * 252
        sims = 1000
        
        price_paths = np.zeros((sim_days + 1, sims))
        price_paths[0] = self.macro_last_price
        
        Z = np.random.normal(0, 1, (sim_days, sims))
        daily_sim_returns = mu + sigma * Z
        
        for t in range(1, sim_days + 1):
            price_paths[t] = price_paths[t-1] * np.exp(daily_sim_returns[t-1])
            
        # Extract Confidence Intervals
        self.macro_best_path = np.percentile(price_paths, 90, axis=1)
        self.macro_median_path = np.percentile(price_paths, 50, axis=1)
        self.macro_worst_path = np.percentile(price_paths, 10, axis=1)
        
        # Save variables for the visualizer
        self.macro_future_dates = pd.date_range(start=stock_prices.index[-1], periods=sim_days + 1, freq='B')
        self.macro_hist_prices = stock_prices
        self.macro_ticker = ticker
        self.macro_sim_years = sim_years

    def run_holistic_screener(self):
        """
        STRATEGY N: Holistic Technical Screener & Action Matrix.
        Scans all assets, applies RSI, Stochastic, and MACD logic, 
        and assigns a definitive action (BUY/HOLD/SELL) based on confluence.
        """
        print("📟 Initializing Action Matrix Screener...")
        scan_data = yf.download(self.all_tickers, period="6mo", progress=False)

        if isinstance(scan_data.columns, pd.MultiIndex):
            close_data, high_data, low_data = scan_data['Close'], scan_data['High'], scan_data['Low']
        else:
            close_data = pd.DataFrame({self.all_tickers[0]: scan_data['Close']})
            high_data = pd.DataFrame({self.all_tickers[0]: scan_data['High']})
            low_data = pd.DataFrame({self.all_tickers[0]: scan_data['Low']})

        self.screener_results = []
        self.screener_failed = []

        for ticker in self.all_tickers:
            try:
                df = pd.DataFrame({'Close': close_data[ticker], 'High': high_data[ticker], 'Low': low_data[ticker]}).dropna()
                if len(df) < 50: 
                    self.screener_failed.append(ticker)
                    continue
                
                # 1. RSI (14)
                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df['RSI'] = 100 - (100 / (1 + rs))
                
                # 2. Stochastic (14, 3, 3)
                low_min = df['Low'].rolling(window=14).min()
                high_max = df['High'].rolling(window=14).max()
                df['Stoch_K_raw'] = 100 * ((df['Close'] - low_min) / (high_max - low_min))
                df['Stoch_K'] = df['Stoch_K_raw'].rolling(window=3).mean()
                df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
                
                # 3. MACD
                ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
                ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
                df['MACD'] = ema_12 - ema_26
                df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
                df['MACD_Hist'] = df['MACD'] - df['Signal']
                
                last = df.iloc[-1]
                prev = df.iloc[-2]
                
                p = last['Close']
                rsi = last['RSI']
                stoch = last['Stoch_K']
                macd_trend = "Bullish" if last['MACD_Hist'] > 0 else "Bearish"
                macd_mom = "Expanding" if last['MACD_Hist'] > prev['MACD_Hist'] else "Contracting"
                
                # Action Matrix AI Logic
                action = "HOLD 🟡"
                
                if rsi < 35 and stoch < 25 and last['MACD_Hist'] > prev['MACD_Hist']:
                    action = "STRONG BUY 🟢"
                elif rsi < 45 and last['MACD_Hist'] > prev['MACD_Hist']:
                    action = "BUY 📈"
                elif rsi > 75 and stoch > 80 and last['MACD_Hist'] < prev['MACD_Hist']:
                    action = "STRONG SELL 🔴"
                elif rsi > 65 and last['MACD_Hist'] < 0:
                    action = "SELL 📉"
                    
                self.screener_results.append((ticker, p, rsi, stoch, f"{macd_trend} ({macd_mom})", action))
                
            except Exception:
                self.screener_failed.append(ticker)
   
