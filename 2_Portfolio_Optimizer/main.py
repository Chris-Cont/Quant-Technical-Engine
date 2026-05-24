from portfolio_optimizer import PortfolioQuantEngine
import visualizer
import config
import numpy as np

if __name__ == "__main__":
    print("="*65)
    print("🚀 INITIALIZING PORTFOLIO QUANT ENGINE 🚀")
    print("="*65)

    # 1. CORE MATHEMATICAL EXECUTION
    engine = PortfolioQuantEngine()
    engine.fetch_data_and_filter()
    engine.run_optimization()
    engine.calculate_vix_adjusted_returns()
    engine.run_backtests_and_stress()
    engine.run_forward_monte_carlo()
    engine.prepare_risk_and_correlation()
    engine.calculate_rolling_volatility()
    engine.run_monte_carlo_comparison() 
     # TECHNICAL ANALYSIS
    engine.calculate_technical_indicators(ticker='SPY') # <-- you can type ticker you want
    engine.calculate_pro_dashboard_indicators(ticker='SPY') # <-- you can type ticker you want
    engine.run_sniper_scanner()
    engine.run_macro_monte_carlo(ticker='SPY', sim_years=5) 
    engine.run_holistic_screener()
    
    # STRATEGY EXTENSIONS
    engine.analyze_vix_sharpe()
    engine.run_gold_monte_carlo()
    engine.calculate_hedge_correlations()
    engine.run_min_variance_optimization()

    engine.calculate_ultimate_dashboard(ticker='SPY')

    # 2. HISTORICAL STRESS TESTING
    scenarios = {
        "COVID-19 Crash (Feb - Mar 2020)": ("2020-02-19", "2020-03-23"),
        "2022 Bear Market / Rate Hikes (Jan - Oct 2022)": ("2022-01-03", "2022-10-12")
    }
    print("\n" + "="*65)
    print(" 🌪️ HISTORICAL STRESS TESTING (EXTREME SCENARIOS)")
    print("="*65)
    for name, (start, end) in scenarios.items():
        try:
            p_ret = engine.port_all_daily.loc[start:end]
            s_ret = engine.spy_all_daily.loc[start:end]
            p_cum = np.exp(p_ret.cumsum()) - 1
            s_cum = np.exp(s_ret.cumsum()) - 1
            p_dd = engine.mdd_func(p_ret)
            s_dd = engine.mdd_func(s_ret)
            
            print(f"\nSCENARIO: {name}")
            print(f" > Portfolio   : Return {p_cum.iloc[-1]:>7.2%} | Max Drawdown {p_dd:>7.2%}")
            print(f" > S&P 500     : Return {s_cum.iloc[-1]:>7.2%} | Max Drawdown {s_dd:>7.2%}")
            if p_dd > s_dd:
                print("   ✅ Hedge successful! Portfolio drawdown was strictly lower than the market.")
        except Exception:
            print(f"\nSCENARIO: {name} - (Insufficient data for timeframe)")

    # 3. BACKTESTING PERFORMANCE
    print("\n" + "="*65)
    print(" 📊 BACKTESTING CALCULATIONS (TRAILING 12 MONTHS)")
    print("="*65)
    print(f"Initial Capital Invested : ${engine.total_budget:,.2f}")
    print(f"Final Portfolio Value    : ${engine.port_final_value:,.2f} (PnL: ${engine.port_final_value - engine.total_budget:,.2f})")
    print(f"100% SPY Benchmark Value : ${engine.spy_final_value:,.2f} (PnL: ${engine.spy_final_value - engine.total_budget:,.2f})")
    print("-" * 65)
    print(f"Cumulative Portfolio Return : {engine.port_final_return:.2%}")
    print(f"Cumulative SPY Return       : {engine.spy_final_return:.2%}")
    print(f"Portfolio Max Drawdown      : {engine.port_mdd_1y:.2%}")
    print(f"SPY Max Drawdown            : {engine.spy_mdd_1y:.2%}")
    print("=" * 65)

    # 4. TRADE EXECUTION ALLOCATIONS (MAX SHARPE)
    print(f"\n📱 [EQUITIES] ALLOCATION ORDERS (Budget {config.stock_budget}$):")
    for i, ticker in enumerate(config.stock_hedge_tickers):
        if engine.opt_amounts[i] > 1.00: 
            print(f" > {ticker:<7}: ${engine.opt_amounts[i]:.2f}")

    print(f"\n📈 [MACRO/CMDTY] ALLOCATION ORDERS (Budget {config.macro_budget}$):")
    for i, ticker in enumerate(config.macro_hedge_tickers):
        if engine.opt_amounts[engine.num_stock + i] > 1.00: 
            print(f" > {ticker:<4}: ${engine.opt_amounts[engine.num_stock + i]:.2f}")
    print("=" * 65)

    # 5. FORWARD MONTE CARLO
    print("\n" + "="*65)
    print(f" 🔮 FORWARD MONTE CARLO (PORTFOLIO PROJECTION - {config.sim_years} YEARS)")
    print("="*65)
    percentile_10 = np.percentile(engine.final_sim_values, 10)
    percentile_50 = np.percentile(engine.final_sim_values, 50)
    percentile_90 = np.percentile(engine.final_sim_values, 90)

    print(f"\nInitial Capital Today: ${engine.total_budget:,.2f}")
    print(f"\nEstimated Value in {config.sim_years} Years ({config.num_sims} Simulations):")
    print(f" 🟢 Best Case (Top 10%)    : ${percentile_90:,.2f}")
    print(f" 🟡 Base Trend (Median)    : ${percentile_50:,.2f}")
    print(f" 🔴 Worst Case (Bottom 10%): ${percentile_10:,.2f}")
    print("=" * 65)
    
    # 6. STRATEGY A: VIX SHARPE RATIO ANALYSIS
    print("\n" + "="*65)
    print(" 📉 STRATEGY A: SHARPE RATIO ANALYSIS (BASE VS MACRO HEDGE)")
    print("="*65)
    for i, row in engine.df_vix_metrics.iterrows():
        print(f"\n{row['Metric (Return/Risk Profile)']}:")
        print(f" > Base Portfolio : {row['Base Portfolio (Equities)']:.2f}")
        print(f" > Macro Hedge    : {row['Macro Hedge (Bonds/Cmdty)']:.2f}")

    # 7. STRATEGY C: CORRELATION CHEAT SHEET
    print("\n" + "="*65)
    print(" 🔍 HEDGING ANALYSIS: Asset Correlation vs Base Portfolio")
    print(" (Negative/Low numbers = Absolute Hedge against Base Equities)")
    print("="*65)
    for ticker, corr in engine.sorted_correlations:
        print(f" > {ticker:<5} : Correlation {corr:>5.2f}")

    # 8. STRATEGY D: MIN VARIANCE CHEAT SHEET (ABSOLUTE DEFENSE)
    print("\n" + "="*65)
    print(" 🛡️ ALTERNATIVE: GLOBAL MINIMUM VARIANCE (ABSOLUTE DEFENSE)")
    print("="*65)
    print("Expected Return : {:.2%}".format(engine.min_var_return))
    print("Annualized Risk : {:.2%} <-- MATHEMATICAL VOLATILITY FLOOR".format(engine.min_var_risk))
    print("-" * 65)

    print(f"\n📱 [EQUITIES] BUY ORDERS FOR ABSOLUTE DEFENSE:")
    for i, ticker in enumerate(config.stock_hedge_tickers):
        amount = engine.min_var_amounts[i] 
        if amount > 1.00:
            print(f" > {ticker:<4}: ${amount:.2f}")

    print(f"\n📈 [MACRO/CMDTY] BUY ORDERS FOR ABSOLUTE DEFENSE:")
    num_stock = len(config.stock_hedge_tickers)
    for i, ticker in enumerate(config.macro_hedge_tickers):
        amount = engine.min_var_amounts[num_stock + i]
        if amount > 1.00:
            print(f" > {ticker:<4}: ${amount:.2f}")
    print("=" * 65)

    # 8.5 STRATEGY E: VIX-ADJUSTED RETURNS PROOF
    print("\n" + "="*65)
    print(" ⚖️ STRATEGY E: THE VOLATILITY PENALTY ON THE UNHEDGED PORTFOLIO")
    print("="*65)
    print(f"Risk Statistics (Base Tech Portfolio):")
    print(f" > Actual Volatility (Black Line)       : {engine.real_vol_old:.2%}")
    print(f" > VIX-Adjusted Volatility (Red Line)   : {engine.adj_vol_old:.2%}")
    print("-" * 65)

        # 8.6 STRATEGY F: ROLLING VOLATILITY ANALYSIS
    print("\n" + "="*65)
    print(" 🌪️ STRATEGY F: THE ULTIMATE PROOF - ROLLING VOLATILITY")
    print("="*65)
    print("Rolling risk calculations complete. Observe the 'Target Volatility'")
    print("flattening effect in the upcoming rendering engine charts.")
    print("-" * 65)
    
        # 8.7 STRATEGY G: MONTE CARLO PROJECTION
    print("\n" + "="*65)
    print(" 🔮 STRATEGY G: MONTE CARLO PROBABILISTIC OUTCOMES")
    print("="*65)
    print("Simulations complete. Comparing raw uncertainty against volatility-managed paths.")
    print("-" * 65)

    # 8.8 STRATEGY H: TECHNICAL ANALYSIS
    print("\n" + "="*65)
    print(f" 📈 STRATEGY H: TECHNICAL ANALYSIS (SMA/EMA) FOR {engine.ta_ticker}")
    print("="*65)
    print("Moving averages calculated. Visualizing trend signals...")
    print("-" * 65)

        # 9. STRATEGY I: TREND SCANNER
    engine.run_trend_scanner()
    print("\n" + "="*75)
    print(" 📖 TREND SCANNER: MARKET STATUS BASED ON SMA RULES")
    print("="*75)
    
    print("\n🟢 BULLISH (Uptrend):")
    for t, p, s20, s50 in engine.bullish:
        print(f" > {t:<6} | Price: ${p:>7.2f} | SMA20: ${s20:>7.2f} | SMA50: ${s50:>7.2f}")

    print("\n🔴 BEARISH (Downtrend):")
    for t, p, s20, s50 in engine.bearish:
        print(f" > {t:<6} | Price: ${p:>7.2f} | SMA20: ${s20:>7.2f} | SMA50: ${s50:>7.2f}")

    print("\n🟡 CONSOLIDATION (Range/Wavering):")
    for t, p, s20, s50 in engine.consolidation:
        print(f" > {t:<6} | Price: ${p:>7.2f} | SMA20: ${s20:>7.2f} | SMA50: ${s50:>7.2f}")
    print("="*75)

        # 9.5 STRATEGY J: MARKET CYCLE QUADRANTS
    print("\n" + "="*75)
    print(" 📊 STRATEGY J: MARKET CYCLE QUADRANTS SCATTER ANALYSIS")
    print("="*75)
    print("Quadrant coordinate matrix computed. Generating visual representation...")
    print("-" * 75)

        # 9.6 STRATEGY K: PRO TRADING DASHBOARD
    print("\n" + "="*75)
    print(f" 🎛️ STRATEGY K: PRO TRADING DASHBOARD INITIALIZED FOR {engine.pro_dash_ticker}")
    print("="*75)
    print("MACD, RSI, and Stochastic oscillators computed. Matrix queued for rendering.")
    print("-" * 75)

    # 9.7 STRATEGY L: THE SNIPER SCANNER
    print("\n" + "="*75)
    print(" 🎯 STRATEGY L: THE SNIPER SCANNER (ALGORITHMIC CONFLUENCE)")
    print("="*75)
    
    print("\n🟢 PURE BUY SIGNALS (RSI + MACD + Stoch Alignment):")
    print("-" * 75)
    if not engine.sniper_buy_signals: 
        print(" > No absolute setups today. Patience pays.")
    for t, status, p, rsi, stoch in engine.sniper_buy_signals:
        print(f" 🚀 {t:<5} | Price: ${p:>7.2f} | RSI: {rsi:>5.1f} | Stoch: {stoch:>5.1f} | Type: {status}")

    print("\n🟡 RADAR / WATCHLIST (Extreme Panic - Ready to Bounce):")
    print("-" * 75)
    if not engine.sniper_watch_list: 
        print(" > No extreme oversold assets found.")
    for t, msg, p, rsi in engine.sniper_watch_list:
        print(f" 👀 {t:<5} | Price: ${p:>7.2f} | RSI: {rsi:>5.1f} | Status: {msg}")
    print("="*75)

    # 9.8 STRATEGY M: 5-YEAR MACRO MONTE CARLO
    if hasattr(engine, 'macro_ticker'):
        print("\n" + "="*75)
        print(f" 🌍 STRATEGY M: 5-YEAR MACRO PROJECTION FOR {engine.macro_ticker}")
        print("="*75)
        print(f"Starting Price (Today) : ${engine.macro_last_price:.2f}")
        
        median_ret = ((engine.macro_median_path[-1] / engine.macro_last_price) - 1) * 100
        best_ret = ((engine.macro_best_path[-1] / engine.macro_last_price) - 1) * 100
        worst_ret = ((engine.macro_worst_path[-1] / engine.macro_last_price) - 1) * 100
        
        print(f"Median Target          : ${engine.macro_median_path[-1]:>7.2f} (Return: {median_ret:>6.1f}%)")
        print(f"Bull Case (Top 10%)    : ${engine.macro_best_path[-1]:>7.2f} (Return: {best_ret:>6.1f}%)")
        print(f"Bear Case (Bottom 10%) : ${engine.macro_worst_path[-1]:>7.2f} (Return: {worst_ret:>6.1f}%)")
        print("="*75)


            # 9.9 STRATEGY N: HOLISTIC SCREENER & ACTION MATRIX
    print("\n" + "="*85)
    print(" 📟 STRATEGY N: ACTION MATRIX (RSI + MACD + STOCHASTIC)")
    print("="*85)
    
    print(f"{'TICKER':<7} | {'PRICE ($)':<9} | {'RSI (Power)':<12} | {'STOCH (Spring)':<16} | {'MACD (Momentum)':<22} | {'VERDICT':<15}")
    print("-" * 85)

    order = {"STRONG BUY 🟢": 1, "BUY 📈": 2, "HOLD 🟡": 3, "SELL 📉": 4, "STRONG SELL 🔴": 5}
    engine.screener_results.sort(key=lambda x: order.get(x[5], 99))

    for t, p, rsi, stoch, macd, action in engine.screener_results:
        print(f"{t:<7} | ${p:<8.2f} | {rsi:<12.1f} | {stoch:<16.1f} | {macd:<22} | {action}")
    print("="*85)

    if engine.screener_failed:
        print(f"⚠️ WARNING: Ignored tickers (Symbol error or lack of data): {', '.join(engine.screener_failed)}")
        print("="*85)

    # 10. TRADER'S CHEAT SHEET
    print("\n" + "="*85)
    print(" 📖 THE QUANT TRADER'S CHEAT SHEET")
    print("="*85)
    print("""
1. SMA (Moving Averages) - "The Road & The Trend"
   - SMA 200 (The King): Macro trend (1 year). 
     * Price > SMA 200 = Bull Market (Healthy). 
     * Price < SMA 200 = Bear Market (Stay away).
   - SMA 50 (The Lieutenant): Medium-term trend. Acts as dynamic Support/Resistance.

2. RSI (Relative Strength Index) - "The Speedometer"
   - RSI < 30 (Oversold): Panic. Asset is cheap.
   - RSI > 70 (Overbought): Greed. Asset is expensive.
   - RSI at 50 (Neutral): Choppy market.
   * Golden Signal: Drops below 30, then hooks UP (Bullish divergence).

3. Stochastic Oscillator - "The Coiled Spring"
   - Under 20: Spring is fully compressed.
   - Over 80: Spring is stretched and ready to snap.
   * Golden Signal: Both lines < 20, and Fast line crosses Slow line UP.

4. MACD - "The Gas Pedal"
   - Green Bars: Buyers in control (Bullish Momentum).
   - Red Bars: Sellers in control (Bearish Momentum).
   * Golden Signal: MACD Line crosses Signal Line UP (Bullish Crossover).

🎯 THE ULTIMATE BUY SETUP (Confluence):
   [1] The Permission: Price is ABOVE SMA 200 & SMA 50.
   [2] The Opportunity: Price drops (touches SMA) & RSI < 35.
   [3] The Spring: Stochastic < 20 & crosses UP.
   [4] The Gas Pedal: MACD creates Bullish Crossover.
   => RESULT: Buying a premium asset at a discount, exactly when Smart Money enters!
    """)
    print("="*85)
    
    # 11. RENDER VISUALS (ALWAYS AT THE END)
    print("\n🎨 Triggering Rendering Engine for Plots...")
    visualizer.generate_plots(engine)
