import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import config
import warnings
warnings.filterwarnings('ignore')

def generate_plots(engine):
    plt.style.use('ggplot')

    # -- Γράφημα 1: Monte Carlo & Efficient Frontier --
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(engine.results[0,:], engine.results[1,:], c=engine.results[2,:], cmap='viridis', marker='o', s=10, alpha=0.3)
    plt.colorbar(scatter, label='VIX-Adjusted Sharpe Ratio')
    if len(engine.frontier_vols) > 0:
        plt.plot(engine.frontier_vols, engine.valid_returns, color='black', linestyle='--', linewidth=2.5, label='Efficient Frontier')
    plt.scatter(engine.risk_before, engine.return_before, marker='X', color='red', s=200, edgecolor='black', label='Old Portfolio (ΠΡΙΝ)')
    plt.scatter(engine.risk_after, engine.return_after, marker='*', color='gold', s=350, edgecolor='black', zorder=5, label='Optimal Hedged Portfolio (ΜΕΤΑ)')
    plt.title('Monte Carlo & Efficient Frontier')
    plt.xlabel('Risk (Volatility)'); plt.ylabel('Expected Return')
    plt.legend(); plt.tight_layout()
    plt.show()

    # -- Γράφημα 2: Πίτα & Μπάρες --
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    labels = [t for i, t in enumerate(engine.all_tickers) if engine.final_weights[i] > 0.01]
    sizes = [w for w in engine.final_weights if w > 0.01]
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, pctdistance=0.85, colors=plt.cm.tab20.colors)
    ax1.add_patch(plt.Circle((0,0),0.70,fc='white'))
    ax1.set_title('Τελική Κατανομή')

    ax2.bar(['Ρίσκο ΠΡΙΝ', 'Ρίσκο ΜΕΤΑ'], [engine.risk_before, engine.risk_after], color=['#e74c3c', '#2ecc71'])
    ax2.set_title('Μείωση Κινδύνου')
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    plt.tight_layout()
    plt.show()

    # -- Γράφημα 3: Backtesting vs S&P 500 --
    plt.figure(figsize=(10, 5))
    plt.plot(engine.port_cum_1y.index, engine.port_cum_1y * 100, label='All-Weather Portfolio', color='#2ecc71', linewidth=2)
    plt.plot(engine.spy_cum_1y.index, engine.spy_cum_1y * 100, label='S&P 500 (SPY)', color='#e74c3c', linestyle='--', linewidth=2)
    plt.fill_between(engine.port_cum_1y.index, engine.port_cum_1y * 100, engine.spy_cum_1y * 100, where=(engine.port_cum_1y > engine.spy_cum_1y), color='#2ecc71', alpha=0.1)
    plt.fill_between(engine.port_cum_1y.index, engine.port_cum_1y * 100, engine.spy_cum_1y * 100, where=(engine.port_cum_1y <= engine.spy_cum_1y), color='#e74c3c', alpha=0.1)
    plt.title('Backtesting: Portfolio vs S&P 500 (Τελευταίο 1 Έτος)')
    plt.ylabel('Cumulative Return (%)')
    plt.legend(); plt.tight_layout()
    plt.show()

    # -- ΟΠΤΙΚΟΠΟΙΗΣΗ: SPAGHETTI CHART --
    plt.figure(figsize=(14, 7))
    plt.plot(engine.price_paths[:, :150], color='#3498db', alpha=0.1)
    p90, p50, p10 = np.percentile(engine.final_sim_values, [90, 50, 10])
    
    plt.axhline(y=p90, color='#2ecc71', linestyle='--', linewidth=2.5, label=f'Top 10% (Αισιόδοξο): ${p90:,.0f}')
    plt.axhline(y=p50, color='#f1c40f', linestyle='-', linewidth=2.5, label=f'Median (Μέσο): ${p50:,.0f}')
    plt.axhline(y=p10, color='#e74c3c', linestyle='--', linewidth=2.5, label=f'Bottom 10% (Απαισιόδοξο): ${p10:,.0f}')
    plt.axhline(y=engine.total_budget, color='black', linestyle='-', linewidth=2, label=f'Σημείο Εκκίνησης (${engine.total_budget:,.0f})')

    plt.title(f'Forward Monte Carlo: Πιθανή Εξέλιξη Χαρτοφυλακίου στα Επόμενα {config.sim_years} Έτη', fontsize=14, fontweight='bold')
    plt.xlabel(f'Ημέρες Διαπραγμάτευσης', fontsize=12)
    plt.ylabel('Αξία Χαρτοφυλακίου ($)', fontsize=12)
    plt.legend(loc='upper left', fontsize=11, framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

    # -- ΔΙΠΛΟ HEATMAP ΣΥΣΧΕΤΙΣΕΩΝ --
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
    sns.heatmap(engine.old_corr, annot=True, cmap='RdYlGn_r', vmin=-1, vmax=1, ax=ax1, linewidths=1, fmt=".2f", square=True, cbar_kws={"shrink": .8}, annot_kws={"size": 11})
    ax1.set_title('ΠΡΙΝ: Συσχετίσεις Αρχικού Χαρτοφυλακίου', fontsize=14, fontweight='bold', pad=20)
    sns.heatmap(engine.new_corr, annot=True, cmap='RdYlGn_r', vmin=-1, vmax=1, ax=ax2, linewidths=1, fmt=".2f", square=True, cbar_kws={"shrink": .8}, annot_kws={"size": 9})
    ax2.set_title('ΜΕΤΑ: Συσχετίσεις Νέου Hedged Χαρτοφυλακίου', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()

    # -- RISK CONTRIBUTION --
    plt.figure(figsize=(12, 6))
    colors = ['#e74c3c' if ticker in engine.old_tickers else '#3498db' for ticker in engine.final_portfolio_tickers]
    plt.bar(engine.final_portfolio_tickers, engine.risk_contribution_pct * 100, color=colors, edgecolor='black')
    plt.axhline(0, color='black', linewidth=1)
    plt.title('Συμμετοχή στον Συνολικό Κίνδυνο (Risk Contribution %)', fontsize=14, fontweight='bold')
    plt.ylabel('% του Συνολικού Κινδύνου', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(handles=[mpatches.Patch(color='#e74c3c', label='Παλιό Χαρτοφυλάκιο'), mpatches.Patch(color='#3498db', label='Νέα Hedges (Max Sharpe)')], loc='upper right')
    plt.tight_layout()
    plt.show()

        # -- STRATEGY E: VIX PENALTY CHART (UNHEDGED PORTFOLIO) --
    if hasattr(engine, 'cum_real_old'):
        fig, ax1 = plt.subplots(figsize=(14, 7))

        # Black line: Actual Return of Base Portfolio
        ax1.plot(engine.cum_real_old.index, engine.cum_real_old, color='black', linewidth=2, label='Base Portfolio - Actual Return')

        # Red dashed line: VIX-Adjusted Return
        ax1.plot(engine.cum_adj_old.index, engine.cum_adj_old, color='#e74c3c', linewidth=2.5, linestyle='--', 
                 label='Base Portfolio - VIX-Adjusted ($r_t / VIX_t$)')

        # Grey panic zones
        for day in engine.high_vix_days:
            ax1.axvline(x=day, color='grey', alpha=0.05, linewidth=2)

        ax1.set_title('The Penalty of Volatility (VIX) on the Unhedged Portfolio', fontsize=15, fontweight='bold')
        ax1.set_ylabel('Cumulative Return (%)', fontsize=12)
        ax1.set_xlabel('Date', fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.5)

        # Custom Legend
        lines, labels = ax1.get_legend_handles_labels()
        vix_patch = mpatches.Patch(color='grey', alpha=0.3, label='High VIX Periods (>25)')
        ax1.legend(handles=lines + [vix_patch], loc='upper left', fontsize=11)

        plt.tight_layout()
        plt.show()
    # -- ROLLING CORRELATIONS & 180-DAY PROJECTION --
    old_w = engine.old_amounts / np.sum(engine.old_amounts)
    base_returns = engine.filtered_returns[engine.old_tickers].dot(old_w)

    plt.figure(figsize=(15, 8))
    m_colors = {'TLT': '#2980b9', 'GLD': '#f1c40f', 'DBC': '#e67e22', 'USO': '#8e44ad'}
    last_date = base_returns.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=config.projection_days, freq='B')

    for ticker in config.macro_hedge_tickers:
        if ticker in engine.filtered_returns.columns:
            roll_corr = base_returns.rolling(window=config.window).corr(engine.filtered_returns[ticker]).dropna()
            if len(roll_corr) == 0: continue
            
            idx = (config.stock_hedge_tickers + config.macro_hedge_tickers).index(ticker)
            is_bought = engine.opt_amounts[idx] > 1.00
            label_suffix = " (Αγοράστηκε)" if is_bought else ""
            line_color = m_colors.get(ticker, '#7f8c8d')
            
            plt.plot(roll_corr.index, roll_corr, label=f'Portfolio vs {ticker}{label_suffix}', color=line_color, linewidth=2.5, alpha=0.9)
            
            hist_mean = roll_corr.mean()
            last_val = roll_corr.iloc[-1]
            decay_factor = 0.03
            projected_corr = []
            current_proj = last_val
            
            for _ in range(config.projection_days):
                current_proj = current_proj + decay_factor * (hist_mean - current_proj)
                projected_corr.append(current_proj)
            
            proj_dates = [last_date] + list(future_dates)
            proj_values = [last_val] + projected_corr
            plt.plot(proj_dates, proj_values, color=line_color, linewidth=2.5, linestyle=':', alpha=0.8)

    plt.axhline(0, color='black', linestyle='-', linewidth=2, zorder=5)
    all_dates = list(roll_corr.index) + list(future_dates)
    plt.fill_between(all_dates, 0, -1, color='#2ecc71', alpha=0.1)
    plt.fill_between(all_dates, 0, 1, color='#e74c3c', alpha=0.1)

    plt.title(f'Δυναμικό Hedging: Ιστορική Συσχέτιση ({config.window} Ημερών) & Πρόβλεψη 180 Ημερών', fontsize=15, fontweight='bold')
    plt.ylim(-1.0, 1.0)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    handles = list(by_label.values())
    handles.extend([mlines.Line2D([], [], color='black', linewidth=2.5, label='Ιστορική Πορεία'), mlines.Line2D([], [], color='black', linewidth=2.5, linestyle=':', label='Πρόβλεψη (Mean Reversion)')])
    
    plt.legend(handles=handles, loc='upper left', fontsize=10, framealpha=0.9, ncol=2)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axvline(x=last_date, color='grey', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # -- STRATEGY B: GOLD DYNAMIC MONTE CARLO PROJECTION CHART --
    if hasattr(engine, 'gold_ticker') and engine.gold_ticker:
        plt.figure(figsize=(12, 7))
        
        # Plot trailing 1-year historical prices
        plt.plot(engine.gld_hist_dates, engine.gld_hist_prices, color='#f1c40f', linewidth=2.5, label=f'Historical Price ({engine.gold_ticker})')
        last_date = engine.gld_hist_dates[-1]
        
        # Plot projected trendlines
        plt.plot(engine.future_gld_dates, engine.best_path, color='#2ecc71', linestyle='--', linewidth=2.5, label=f'Best Case (Top 10%): ${engine.best_path[-1]:.2f}')
        plt.plot(engine.future_gld_dates, engine.median_path, color='#95a5a6', linestyle='-', linewidth=2.5, label=f'Base Trend (Median): ${engine.median_path[-1]:.2f}')
        plt.plot(engine.future_gld_dates, engine.worst_path, color='#e74c3c', linestyle='--', linewidth=2.5, label=f'Worst Case (Bottom 10%): ${engine.worst_path[-1]:.2f}')
        
        # Fill the Monte Carlo "Cone" of probability
        plt.fill_between(engine.future_gld_dates, engine.worst_path, engine.best_path, color='#f1c40f', alpha=0.1)
        
        plt.title(f'Dynamic Monte Carlo: {engine.gold_ticker} Price Projection to Late 2026', fontsize=14, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel(f'{engine.gold_ticker} Price (USD)', fontsize=12)
        plt.legend(loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # 'Today' Marker
        plt.axvline(x=last_date, color='black', linestyle=':', alpha=0.5)
        plt.text(last_date + pd.Timedelta(days=5), engine.gld_last_price, 'Today 👉', color='black', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    else:
        print("\n⚠️ Gold projection plot skipped (Target ticker not found in portfolio).")
