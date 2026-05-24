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
    engine.run_backtests_and_stress()
    engine.run_forward_monte_carlo()
    engine.prepare_risk_and_correlation()
    
    # STRATEGY EXTENSIONS
    engine.analyze_vix_sharpe()
    engine.run_gold_monte_carlo()
    engine.calculate_hedge_correlations()
    engine.run_min_variance_optimization()

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
    
    # 9. RENDER VISUALS (ALWAYS AT THE END)
    print("\n🎨 Triggering Rendering Engine for Plots...")
    visualizer.generate_plots(engine)
