from portfolio_optimizer import PortfolioQuantEngine
import visualizer
import config
import numpy as np

if __name__ == "__main__":
    print("="*60)
    print("🚀 ΕΚΚΙΝΗΣΗ PORTFOLIO QUANT ENGINE 🚀")
    print("="*60)

    # 1. Εκτέλεση Μαθηματικών Υπολογισμών
    engine = PortfolioQuantEngine()
    engine.fetch_data_and_filter()
    engine.run_optimization()
    engine.run_backtests_and_stress()
    engine.run_forward_monte_carlo()
    engine.prepare_risk_and_correlation()

    # 2. Historical Stress Testing
    scenarios = {
        "COVID-19 Crash (Φεβ - Μαρ 2020)": ("2020-02-19", "2020-03-23"),
        "2022 Bear Market / Rate Hikes (Ιαν - Οκτ 2022)": ("2022-01-03", "2022-10-12")
    }
    print("\n" + "="*60)
    print(" 🌪️ HISTORICAL STRESS TESTING (ΑΚΡΑΙΑ ΙΣΤΟΡΙΚΑ ΣΕΝΑΡΙΑ)")
    print("="*60)
    for name, (start, end) in scenarios.items():
        try:
            p_ret = engine.port_all_daily.loc[start:end]
            s_ret = engine.spy_all_daily.loc[start:end]
            p_cum = np.exp(p_ret.cumsum()) - 1
            s_cum = np.exp(s_ret.cumsum()) - 1
            p_dd = engine.mdd_func(p_ret)
            s_dd = engine.mdd_func(s_ret)
            
            print(f"\nΣΕΝΑΡΙΟ: {name}")
            print(f" > Το Χαρτοφυλάκιό σου : Απόδοση {p_cum.iloc[-1]:>7.2%} | Max Drawdown {p_dd:>7.2%}")
            print(f" > S&P 500 (SPY)       : Απόδοση {s_cum.iloc[-1]:>7.2%} | Max Drawdown {s_dd:>7.2%}")
            if p_dd > s_dd:
                print("   ✅ Το Hedge δούλεψε! Έχασες λιγότερα από την αγορά.")
        except Exception:
            print(f"\nΣΕΝΑΡΙΟ: {name} - (Μη επαρκή δεδομένα)")

    # 3. Εκτύπωση Backtest
    print("\n" + "="*60)
    print(" 📊 BACKTESTING ΥΠΟΛΟΓΙΣΜΟΙ (ΤΕΛΕΥΤΑΙΟΙ 12 ΜΗΝΕΣ)")
    print("="*60)
    print(f"Αρχικό Κεφάλαιο Επένδυσης : ${engine.total_budget:.2f}")
    print(f"Τελική Αξία Χαρτοφυλακίου : ${engine.port_final_value:.2f} (Κέρδος/Ζημιά: ${engine.port_final_value - engine.total_budget:.2f})")
    print(f"Αν τα έβαζες ΟΛΑ στον SPY : ${engine.spy_final_value:.2f} (Κέρδος/Ζημιά: ${engine.spy_final_value - engine.total_budget:.2f})")
    print("-" * 60)
    print(f"Σωρευτική Απόδοση Χαρτοφυλακίου σου : {engine.port_final_return:.2%}")
    print(f"Σωρευτική Απόδοση S&P 500 (SPY)     : {engine.spy_final_return:.2%}")
    print(f"Max Drawdown Χαρτοφυλακίου σου      : {engine.port_mdd_1y:.2%}")
    print(f"Max Drawdown S&P 500 (SPY)          : {engine.spy_mdd_1y:.2%}")
    print("=" * 60)

    # 4. Εκτύπωση Εντολών Αγοράς
    print(f"\n📱 [REVOLUT] ΑΓΟΡΕΣ ΜΕΤΟΧΩΝ (Budget {config.stock_budget}$):")
    for i, ticker in enumerate(config.stock_hedge_tickers):
        if engine.opt_amounts[i] > 1.00: print(f" > {ticker:<7}: ${engine.opt_amounts[i]:.2f}")

    print(f"\n📈 [eToro] ΑΓΟΡΕΣ MACRO/OIL (Budget {config.macro_budget}$):")
    for i, ticker in enumerate(config.macro_hedge_tickers):
        if engine.opt_amounts[engine.num_stock + i] > 1.00: print(f" > {ticker:<4}: ${engine.opt_amounts[engine.num_stock + i]:.2f}")
    print("=" * 60)

    # 5. Εκτύπωση Forward Monte Carlo
    print("\n" + "="*60)
    print(f" 🔮 ΠΡΟΒΛΕΨΗ ΜΕΛΛΟΝΤΟΣ (FORWARD MONTE CARLO - {config.sim_years} ΕΤΗ)")
    print("="*60)
    percentile_10 = np.percentile(engine.final_sim_values, 10)
    percentile_50 = np.percentile(engine.final_sim_values, 50)
    percentile_90 = np.percentile(engine.final_sim_values, 90)

    print(f"\nΑρχικό Κεφάλαιο Σήμερα: ${engine.total_budget:.2f}")
    print(f"\nΕκτιμώμενη Αξία σε {config.sim_years} Έτη (Βάσει {config.num_sims} Προσομοιώσεων):")
    print(f" 🟢 Αισιόδοξο Σενάριο (Top 10%)  : ${percentile_90:,.2f}")
    print(f" 🟡 Μέσο Σενάριο (Median 50%)    : ${percentile_50:,.2f}")
    print(f" 🔴 Απαισιόδοξο Σενάριο (Low 10%) : ${percentile_10:,.2f}")
    print("=" * 60)

    # 6. Εκτέλεση Visualizer
    print("\n🎨 Παραγωγή Γραφημάτων...")
    visualizer.generate_plots(engine)

    # 5.5 STRATEGY A: VIX SHARPE RATIO ANALYSIS
    print("\n" + "="*65)
    print(" 📊 STRATEGY A: SHARPE RATIO ANALYSIS (BASE VS MACRO HEDGE)")
    print("="*65)
    for i, row in engine.df_vix_metrics.iterrows():
        print(f"\n{row['Metric (Return/Risk Profile)']}:")
        print(f" > Base Portfolio : {row['Base Portfolio (Equities)']:.2f}")
        print(f" > Macro Hedge    : {row['Macro Hedge (Bonds/Cmdty)']:.2f}")
