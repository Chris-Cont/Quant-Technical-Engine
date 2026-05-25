import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.ticker import StrMethodFormatter
import numpy as np
import pandas as pd
import config
import warnings
warnings.filterwarnings('ignore')

def generate_plots(engine):
    plt.style.use('ggplot')

    # -- Plot 1: Monte Carlo & Efficient Frontier --
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(engine.results[0,:], engine.results[1,:], c=engine.results[2,:], cmap='viridis', marker='o', s=10, alpha=0.3)
    plt.colorbar(scatter, label='VIX-Adjusted Sharpe Ratio')
    if len(engine.frontier_vols) > 0:
        plt.plot(engine.frontier_vols, engine.valid_returns, color='black', linestyle='--', linewidth=2.5, label='Efficient Frontier')
    plt.scatter(engine.risk_before, engine.return_before, marker='X', color='red', s=200, edgecolor='black', label='Old Portfolio (Unhedged)')
    plt.scatter(engine.risk_after, engine.return_after, marker='*', color='gold', s=350, edgecolor='black', zorder=5, label='Optimal Hedged Portfolio')
    plt.title('Monte Carlo Simulations & Efficient Frontier', fontweight='bold')
    plt.xlabel('Risk (Annualized Volatility)')
    plt.ylabel('Expected Annual Return')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # -- Plot 2: Allocation Pie & Risk Reduction Bars --
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    labels = [t for i, t in enumerate(engine.all_tickers) if engine.final_weights[i] > 0.01]
    sizes = [w for w in engine.final_weights if w > 0.01]
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, pctdistance=0.85, colors=plt.cm.tab20.colors)
    ax1.add_patch(plt.Circle((0,0),0.70,fc='white'))
    ax1.set_title('Final Target Allocation', fontweight='bold')

    ax2.bar(['Risk BEFORE', 'Risk AFTER'], [engine.risk_before, engine.risk_after], color=['#e74c3c', '#2ecc71'])
    ax2.set_title('Volatility Reduction', fontweight='bold')
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    plt.tight_layout()
    plt.show()

    # -- Plot 3: Backtesting vs S&P 500 --
    plt.figure(figsize=(10, 5))
    plt.plot(engine.port_cum_1y.index, engine.port_cum_1y * 100, label='All-Weather Portfolio', color='#2ecc71', linewidth=2)
    plt.plot(engine.spy_cum_1y.index, engine.spy_cum_1y * 100, label='S&P 500 (SPY)', color='#e74c3c', linestyle='--', linewidth=2)
    plt.fill_between(engine.port_cum_1y.index, engine.port_cum_1y * 100, engine.spy_cum_1y * 100, where=(engine.port_cum_1y > engine.spy_cum_1y), color='#2ecc71', alpha=0.1)
    plt.fill_between(engine.port_cum_1y.index, engine.port_cum_1y * 100, engine.spy_cum_1y * 100, where=(engine.port_cum_1y <= engine.spy_cum_1y), color='#e74c3c', alpha=0.1)
    plt.title('1-Year Backtest: Portfolio vs S&P 500', fontweight='bold')
    plt.ylabel('Cumulative Return (%)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # -- Plot 4: Forward Monte Carlo Spaghetti Chart --
    plt.figure(figsize=(14, 7))
    plt.plot(engine.price_paths[:, :150], color='#3498db', alpha=0.1)
    p90, p50, p10 = np.percentile(engine.final_sim_values, [90, 50, 10])
    
    plt.axhline(y=p90, color='#2ecc71', linestyle='--', linewidth=2.5, label=f'Top 10% (Best Case): ${p90:,.0f}')
    plt.axhline(y=p50, color='#f1c40f', linestyle='-', linewidth=2.5, label=f'Median (Base Trend): ${p50:,.0f}')
    plt.axhline(y=p10, color='#e74c3c', linestyle='--', linewidth=2.5, label=f'Bottom 10% (Worst Case): ${p10:,.0f}')
    plt.axhline(y=engine.total_budget, color='black', linestyle='-', linewidth=2, label=f'Starting Capital (${engine.total_budget:,.0f})')

    plt.title(f'Forward Monte Carlo: Portfolio Equity Curve Projection ({config.sim_years} Years)', fontsize=14, fontweight='bold')
    plt.xlabel('Trading Days', fontsize=12)
    plt.ylabel('Portfolio Value (USD)', fontsize=12)
    plt.legend(loc='upper left', fontsize=11, framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

    # -- Plot 5: Dual Correlation Heatmaps --
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
    sns.heatmap(engine.old_corr, annot=True, cmap='RdYlGn_r', vmin=-1, vmax=1, ax=ax1, linewidths=1, fmt=".2f", square=True, cbar_kws={"shrink": .8}, annot_kws={"size": 11})
    ax1.set_title('BEFORE: Base Portfolio Asset Correlation', fontsize=14, fontweight='bold', pad=20)
    sns.heatmap(engine.new_corr, annot=True, cmap='RdYlGn_r', vmin=-1, vmax=1, ax=ax2, linewidths=1, fmt=".2f", square=True, cbar_kws={"shrink": .8}, annot_kws={"size": 9})
    ax2.set_title('AFTER: Hedged Portfolio Asset Correlation', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()

    # -- Plot 6: Risk Contribution --
    plt.figure(figsize=(12, 6))
    colors = ['#e74c3c' if ticker in engine.old_tickers else '#3498db' for ticker in engine.final_portfolio_tickers]
    plt.bar(engine.final_portfolio_tickers, engine.risk_contribution_pct * 100, color=colors, edgecolor='black')
    plt.axhline(0, color='black', linewidth=1)
    plt.title('Marginal Risk Contribution per Asset (%)', fontsize=14, fontweight='bold')
    plt.ylabel('% of Total Portfolio Volatility', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(handles=[mpatches.Patch(color='#e74c3c', label='Base Equities'), mpatches.Patch(color='#3498db', label='Macro Hedges')], loc='upper right')
    plt.tight_layout()
    plt.show()

    # -- STRATEGY E: VIX PENALTY CHART (UNHEDGED PORTFOLIO) --
    if hasattr(engine, 'cum_real_old'):
        fig, ax1 = plt.subplots(figsize=(14, 7))
        ax1.plot(engine.cum_real_old.index, engine.cum_real_old, color='black', linewidth=2, label='Base Portfolio - Actual Return')
        ax1.plot(engine.cum_adj_old.index, engine.cum_adj_old, color='#e74c3c', linewidth=2.5, linestyle='--', label='Base Portfolio - VIX-Adjusted ($r_t / VIX_t$)')
        for day in engine.high_vix_days:
            ax1.axvline(x=day, color='grey', alpha=0.05, linewidth=2)
        ax1.set_title('The Penalty of Volatility (VIX) on the Unhedged Portfolio', fontsize=15, fontweight='bold')
        ax1.set_ylabel('Cumulative Return (%)', fontsize=12)
        ax1.set_xlabel('Date', fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.5)
        lines, labels = ax1.get_legend_handles_labels()
        vix_patch = mpatches.Patch(color='grey', alpha=0.3, label='High VIX Periods (>25)')
        ax1.legend(handles=lines + [vix_patch], loc='upper left', fontsize=11)
        plt.tight_layout()
        plt.show()

    # -- STRATEGY F: ROLLING VOLATILITY CHART --
    if hasattr(engine, 'real_rolling_vol'):
        fig, ax1 = plt.subplots(figsize=(14, 7))
        ax1.plot(engine.real_rolling_vol.index, engine.real_rolling_vol, color='black', linewidth=2, label='Actual Risk (Base Tech Portfolio)')
        ax1.plot(engine.adj_rolling_vol.index, engine.adj_rolling_vol, color='#e74c3c', linewidth=3, linestyle='-', alpha=0.9, label='VIX-Adjusted Risk (Target Volatility)')
        if hasattr(engine, 'high_vix_days'):
            for day in engine.high_vix_days:
                ax1.axvline(x=day, color='grey', alpha=0.05, linewidth=2)
        ax1.set_title('The Truth of Risk: How VIX-Adjustment Flattens Volatility Spikes', fontsize=15, fontweight='bold')
        ax1.set_ylabel('Annualized Volatility (%)', fontsize=12)
        ax1.set_xlabel('Date', fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.5)
        lines, labels = ax1.get_legend_handles_labels()
        vix_patch = mpatches.Patch(color='grey', alpha=0.3, label='High VIX Periods (Panic)')
        ax1.legend(handles=lines + [vix_patch], loc='upper left', fontsize=11)
        plt.tight_layout()
        plt.show()

    # -- Plot 9: ROLLING CORRELATIONS & PROJECTION --
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
            label_suffix = " (Purchased)" if is_bought else ""
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

    plt.title(f'Dynamic Hedging: Historical Correlation ({config.window}-Day) & 180-Day Mean Reversion Projection', fontsize=15, fontweight='bold')
    plt.ylim(-1.0, 1.0)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    handles = list(by_label.values())
    handles.extend([mlines.Line2D([], [], color='black', linewidth=2.5, label='Historical Trajectory'), mlines.Line2D([], [], color='black', linewidth=2.5, linestyle=':', label='Projection (Mean Reversion)')])
    
    plt.legend(handles=handles, loc='upper left', fontsize=10, framealpha=0.9, ncol=2)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axvline(x=last_date, color='grey', linestyle='--', alpha=0.7)
    plt.text(last_date + pd.Timedelta(days=5), 0.9, 'Start Projection\n(180 Days) 👉', color='grey', fontweight='bold')
    plt.tight_layout()
    plt.show()

    # -- STRATEGY B: GOLD DYNAMIC MONTE CARLO PROJECTION CHART --
    if hasattr(engine, 'gold_ticker') and engine.gold_ticker:
        plt.figure(figsize=(12, 7))
        plt.plot(engine.gld_hist_dates, engine.gld_hist_prices, color='#f1c40f', linewidth=2.5, label=f'Historical Price ({engine.gold_ticker})')
        last_date = engine.gld_hist_dates[-1]
        
        plt.plot(engine.future_gld_dates, engine.best_path, color='#2ecc71', linestyle='--', linewidth=2.5, label=f'Best Case (Top 10%): ${engine.best_path[-1]:.2f}')
        plt.plot(engine.future_gld_dates, engine.median_path, color='#95a5a6', linestyle='-', linewidth=2.5, label=f'Base Trend (Median): ${engine.median_path[-1]:.2f}')
        plt.plot(engine.future_gld_dates, engine.worst_path, color='#e74c3c', linestyle='--', linewidth=2.5, label=f'Worst Case (Bottom 10%): ${engine.worst_path[-1]:.2f}')
        
        plt.fill_between(engine.future_gld_dates, engine.worst_path, engine.best_path, color='#f1c40f', alpha=0.1)
        
        plt.title(f'Dynamic Monte Carlo: {engine.gold_ticker} Price Projection to Late 2026', fontsize=14, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel(f'{engine.gold_ticker} Price (USD)', fontsize=12)
        plt.legend(loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.axvline(x=last_date, color='black', linestyle=':', alpha=0.5)
        plt.text(last_date + pd.Timedelta(days=5), engine.gld_last_price, 'Today 👉', color='black', fontweight='bold')
        plt.tight_layout()
        plt.show()
    else:
        print("\n⚠️ Gold projection plot skipped (Target ticker not found in portfolio).")
        
    # -- STRATEGY J: MARKET CYCLE QUADRANTS SCATTER PLOT --
    if hasattr(engine, 'df_quadrants') and not engine.df_quadrants.empty:
        fig, ax = plt.subplots(figsize=(14, 10))
        df_plot = engine.df_quadrants

        # Dynamic boundaries for background shading
        max_x, min_x = df_plot['Dist_SMA50'].max() + 5, df_plot['Dist_SMA50'].min() - 5

        # Draw the 4 Quadrants
        ax.axvspan(0, max_x, ymin=0.5, ymax=1, color='#2ecc71', alpha=0.15)      # Full Bull
        ax.axvspan(min_x, 0, ymin=0, ymax=0.5, color='#e74c3c', alpha=0.15)     # Capitulation
        ax.axvspan(min_x, 0, ymin=0.5, ymax=1, color='#f1c40f', alpha=0.1)      # Bear Market Rally
        ax.axvspan(0, max_x, ymin=0, ymax=0.5, color='#f1c40f', alpha=0.1)      # Pullback

        # Axes
        ax.axhline(0, color='black', linewidth=1.5, linestyle='--')
        ax.axvline(0, color='black', linewidth=1.5, linestyle='--')

        # Scatter points
        ax.scatter(df_plot['Dist_SMA50'], df_plot['Dist_SMA20'], color='#2c3e50', s=80, edgecolors='white', zorder=5)

        # Labels for tickers
        for i, row in df_plot.iterrows():
            ax.text(row['Dist_SMA50'] + 0.3, row['Dist_SMA20'] + 0.3, row['Ticker'], 
                    fontsize=11, fontweight='bold', color='black')

        # Formatting
        ax.set_title('Market Cycle Quadrants (Asset Positioning)', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Macro Trend (% Distance from SMA 50)', fontsize=12)
        ax.set_ylabel('Micro Momentum (% Distance from SMA 20)', fontsize=12)

        # Wall Street Slang Labels
        ax.text(df_plot['Dist_SMA50'].max(), df_plot['Dist_SMA20'].max(), '🟢 FULL BULL\n(Mark-Up Phase)', 
                fontsize=14, color='green', alpha=0.5, ha='right', va='top', fontweight='bold')
                
        ax.text(df_plot['Dist_SMA50'].min(), df_plot['Dist_SMA20'].min(), '🔴 CAPITULATION\n(Mark-Down / Panic)', 
                fontsize=14, color='red', alpha=0.5, ha='left', va='bottom', fontweight='bold')
                
        ax.text(df_plot['Dist_SMA50'].min(), df_plot['Dist_SMA20'].max(), '🟡 BEAR MARKET RALLY\n(Dead Cat Bounce?)', 
                fontsize=12, color='olive', alpha=0.5, ha='left', va='top', fontweight='bold')
                
        ax.text(df_plot['Dist_SMA50'].max(), df_plot['Dist_SMA20'].min(), '🟡 PULLBACK\n(Healthy Correction / Buy the Dip)', 
                fontsize=12, color='olive', alpha=0.5, ha='right', va='bottom', fontweight='bold')

        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.show()
        
    # -- STRATEGY G: COMPARATIVE MONTE CARLO PROJECTION --
    if hasattr(engine, 'real_paths'):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

        # Plot 1: Raw Portfolio
        ax1.plot(engine.future_dates, engine.real_percentiles[0], color='#2ecc71', linestyle='--', label=f'Top 10%: ${engine.real_percentiles[0,-1]:,.0f}')
        ax1.plot(engine.future_dates, engine.real_percentiles[1], color='black', linestyle='-', linewidth=2.5, label=f'Median: ${engine.real_percentiles[1,-1]:,.0f}')
        ax1.plot(engine.future_dates, engine.real_percentiles[2], color='#e74c3c', linestyle='--', label=f'Bottom 10%: ${engine.real_percentiles[2,-1]:,.0f}')
        ax1.fill_between(engine.future_dates, engine.real_percentiles[2], engine.real_percentiles[0], color='black', alpha=0.1)
        ax1.set_title('Unhedged Portfolio (High Uncertainty)', fontweight='bold')
        ax1.set_xlabel('Date'); ax1.set_ylabel('Portfolio Value ($)'); ax1.grid(True, linestyle='--', alpha=0.5); ax1.legend(loc='upper left')

        # Plot 2: VIX-Adjusted Portfolio
        ax2.plot(engine.future_dates, engine.adj_percentiles[0], color='#2ecc71', linestyle='--', label=f'Top 10%: ${engine.adj_percentiles[0,-1]:,.0f}')
        ax2.plot(engine.future_dates, engine.adj_percentiles[1], color='#c0392b', linestyle='-', linewidth=2.5, label=f'Median: ${engine.adj_percentiles[1,-1]:,.0f}')
        ax2.plot(engine.future_dates, engine.adj_percentiles[2], color='#e74c3c', linestyle='--', label=f'Bottom 10%: ${engine.adj_percentiles[2,-1]:,.0f}')
        ax2.fill_between(engine.future_dates, engine.adj_percentiles[2], engine.adj_percentiles[0], color='#e74c3c', alpha=0.15)
        ax2.set_title('VIX-Adjusted Portfolio (Volatility Control)', fontweight='bold')
        ax2.set_xlabel('Date'); ax2.grid(True, linestyle='--', alpha=0.5); ax2.legend(loc='upper left')

        plt.suptitle('Projection to End of 2026: The Power of Risk Stabilization', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    # -- STRATEGY H: TECHNICAL ANALYSIS (SMA & EMA) --
    if hasattr(engine, 'ta_price_data'):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        # Plot SMA
        ax1.plot(engine.ta_price_data.index, engine.ta_price_data, color='black', linewidth=1.5, label=f'{engine.ta_ticker} Price', alpha=0.5)
        ax1.plot(engine.sma_20.index, engine.sma_20, color='#3498db', linewidth=2, label='SMA (20)')
        ax1.plot(engine.sma_50.index, engine.sma_50, color='#2980b9', linewidth=2, linestyle='--', label='SMA (50)')
        ax1.set_title(f'Simple Moving Averages (SMA) for {engine.ta_ticker}', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price (USD)'); ax1.legend(); ax1.grid(True, linestyle='--', alpha=0.5)

        # Plot EMA
        ax2.plot(engine.ta_price_data.index, engine.ta_price_data, color='black', linewidth=1.5, label=f'{engine.ta_ticker} Price', alpha=0.5)
        ax2.plot(engine.ema_20.index, engine.ema_20, color='#e67e22', linewidth=2, label='EMA (20) - Fast')
        ax2.plot(engine.ema_50.index, engine.ema_50, color='#d35400', linewidth=2, linestyle='--', label='EMA (50) - Slow')
        ax2.set_title(f'Exponential Moving Averages (EMA) for {engine.ta_ticker}', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Price (USD)'); ax2.set_xlabel('Date'); ax2.legend(); ax2.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.show()
        
    # -- STRATEGY K: PRO TRADING DASHBOARD --
    if hasattr(engine, 'pro_dash_data') and not engine.pro_dash_data.empty:
        df_plot = engine.pro_dash_data
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(16, 18), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1, 1.5]})
        fig.suptitle(f'Institutional Technical Analysis: {engine.pro_dash_ticker} (Trend & Momentum)', fontsize=18, fontweight='bold', y=0.92)

        # Panel 1: Price & MAs
        ax1.plot(df_plot.index, df_plot['Close'], color='black', linewidth=1.5, label='Close Price')
        ax1.plot(df_plot.index, df_plot['SMA_20'], color='#3498db', linewidth=2, label='SMA 20 (Short-Term)')
        ax1.plot(df_plot.index, df_plot['SMA_50'], color='#e67e22', linewidth=2, label='SMA 50 (Medium-Term)')
        ax1.plot(df_plot.index, df_plot['SMA_200'], color='#e74c3c', linewidth=2.5, label='SMA 200 (Macro Trend)')
        ax1.set_title('1. Price Action & Moving Averages', fontsize=12, loc='left', fontweight='bold')
        ax1.set_ylabel('Price (USD)'); ax1.legend(loc='upper left'); ax1.grid(True, linestyle='--', alpha=0.5)

        # Panel 2: RSI
        ax2.plot(df_plot.index, df_plot['RSI_14'], color='#8e44ad', linewidth=2, label='RSI (14)')
        ax2.axhline(70, color='red', linestyle='--', alpha=0.5)
        ax2.axhline(30, color='green', linestyle='--', alpha=0.5)
        ax2.fill_between(df_plot.index, 70, 100, color='red', alpha=0.1)
        ax2.fill_between(df_plot.index, 0, 30, color='green', alpha=0.1)
        ax2.set_title('2. Relative Strength Index (Overbought > 70 | Oversold < 30)', fontsize=12, loc='left', fontweight='bold')
        ax2.set_ylabel('RSI'); ax2.set_ylim(0, 100); ax2.grid(True, linestyle='--', alpha=0.5)

        # Panel 3: Stochastic
        ax3.plot(df_plot.index, df_plot['Stoch_K'], color='#2980b9', linewidth=2, label='%K (Fast)')
        ax3.plot(df_plot.index, df_plot['Stoch_D'], color='#e74c3c', linewidth=2, linestyle='--', label='%D (Slow)')
        ax3.axhline(80, color='red', linestyle='--', alpha=0.5)
        ax3.axhline(20, color='green', linestyle='--', alpha=0.5)
        ax3.fill_between(df_plot.index, 80, 100, color='red', alpha=0.1)
        ax3.fill_between(df_plot.index, 0, 20, color='green', alpha=0.1)
        ax3.set_title('3. Stochastic Oscillator (Watch Extreme Crossovers)', fontsize=12, loc='left', fontweight='bold')
        ax3.set_ylabel('Stoch'); ax3.set_ylim(0, 100); ax3.legend(loc='upper left'); ax3.grid(True, linestyle='--', alpha=0.5)

        # Panel 4: MACD
        colors = ['#2ecc71' if val >= 0 else '#e74c3c' for val in df_plot['MACD_Hist']]
        ax4.bar(df_plot.index, df_plot['MACD_Hist'], color=colors, alpha=0.5, label='MACD Histogram')
        ax4.plot(df_plot.index, df_plot['MACD'], color='black', linewidth=2, label='MACD Line (12, 26)')
        ax4.plot(df_plot.index, df_plot['Signal_Line'], color='blue', linewidth=2, linestyle='--', label='Signal Line (9)')
        ax4.set_title('4. MACD (Momentum Crossovers & Divergence)', fontsize=12, loc='left', fontweight='bold')
        ax4.set_ylabel('MACD'); ax4.set_xlabel('Date'); ax4.legend(loc='upper left'); ax4.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.show()
        
            # -- STRATEGY M: MACRO DASHBOARD & 5-YEAR MONTE CARLO --
    if hasattr(engine, 'macro_ticker'):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [1, 2]})

        # --- Top Panel: US Treasury Yields ---
        ax1.plot(engine.macro_rates.index, engine.macro_rates, color='#c0392b', linewidth=2)
        ax1.fill_between(engine.macro_rates.index, engine.macro_rates, color='#c0392b', alpha=0.1)
        ax1.set_title('Macroeconomic Environment: US 10-Year Treasury Yield (%)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Yield (%)', fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.yaxis.set_major_formatter(StrMethodFormatter('{x:.1f}'))

        current_rate = engine.macro_rates.iloc[-1]
        ax1.text(engine.macro_rates.index[-1], current_rate + 0.5, f"Current Yield:\n{current_rate:.2f}%", 
                 fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))

        # --- Bottom Panel: 5-Year Monte Carlo ---
        hist_dates = engine.macro_hist_prices.index[-750:]
        hist_prices = engine.macro_hist_prices[-750:]

        ax2.plot(hist_dates, hist_prices, color='black', linewidth=2, label='Historical Price')
        ax2.plot(engine.macro_future_dates, engine.macro_best_path, color='#2ecc71', linestyle='--', linewidth=2, label=f'Best Case (Top 10%): ${engine.macro_best_path[-1]:,.1f}')
        ax2.plot(engine.macro_future_dates, engine.macro_median_path, color='#3498db', linestyle='-', linewidth=2.5, label=f'Median Projection: ${engine.macro_median_path[-1]:,.1f}')
        ax2.plot(engine.macro_future_dates, engine.macro_worst_path, color='#e74c3c', linestyle='--', linewidth=2, label=f'Worst Case (Bottom 10%): ${engine.macro_worst_path[-1]:,.1f}')
        ax2.fill_between(engine.macro_future_dates, engine.macro_worst_path, engine.macro_best_path, color='#3498db', alpha=0.1)

        ax2.set_title(f'Long-Term Horizon: {engine.macro_sim_years}-Year Monte Carlo Projection for {engine.macro_ticker}', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Stock Price (USD)', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.legend(loc='upper left', fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:.1f}'))

        ax2.axvline(x=engine.macro_hist_prices.index[-1], color='grey', linestyle=':', linewidth=2)
        ax2.text(engine.macro_hist_prices.index[-1] + pd.Timedelta(days=30), engine.macro_hist_prices.iloc[-1], 'TODAY 🚀', fontweight='bold')

        plt.tight_layout()
        plt.show()

    # -- STRATEGY O: ULTIMATE CANDLESTICK DASHBOARD --
    if hasattr(engine, 'ultimate_dash_df'):
        from mplfinance.original_flavor import candlestick_ohlc
        import matplotlib.dates as mdates

        df_plot = engine.ultimate_dash_df.reset_index()
        df_plot['Date_mpl'] = df_plot['Date'].apply(lambda x: mdates.date2num(x))
        ohlc = df_plot[['Date_mpl', 'Open', 'High', 'Low', 'Close']].values
        buy_points = df_plot[df_plot['Buy_Signal']]
        sell_points = df_plot[df_plot['Sell_Signal']]

        plt.style.use('dark_background')
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(16, 20), sharex=True, 
                                                 gridspec_kw={'height_ratios': [3.5, 1, 1, 1.5]})
        fig.suptitle(f'Advanced Technical Analysis: {engine.ultimate_ticker}', fontsize=20, fontweight='bold', color='white', y=0.92)

        # Panel 1: Price, SMAs & Signals
        candlestick_ohlc(ax1, ohlc, width=0.6, colorup='#2ecc71', colordown='#e74c3c')
        ax1.plot(df_plot['Date_mpl'], df_plot['SMA_50'], color='#f39c12', linewidth=2.5, label='SMA 50')
        ax1.plot(df_plot['Date_mpl'], df_plot['SMA_200'], color='#3498db', linewidth=3.5, label='SMA 200')
        ax1.scatter(buy_points['Date_mpl'], buy_points['Low'] * 0.98, marker='^', color='#2ecc71', s=200, zorder=5, label='Buy Signal')
        ax1.scatter(sell_points['Date_mpl'], sell_points['High'] * 1.02, marker='v', color='#e74c3c', s=200, zorder=5, label='Sell Signal')
        ax1.set_title('1. Price Action, Trend & Signals', color='#bdc3c7', fontsize=13, loc='left')
        ax1.set_ylabel('Price (USD)')
        ax1.grid(True, linestyle='--', alpha=0.3)
        ax1.legend(loc='upper left')

        # Panel 2: RSI 14
        ax2.plot(df_plot['Date_mpl'], df_plot['RSI_14'], color='#3498db', linewidth=2)
        ax2.axhline(70, color='#e74c3c', linestyle='--', alpha=0.5)
        ax2.axhline(30, color='#2ecc71', linestyle='--', alpha=0.5)
        ax2.set_title('2. RSI (14) - Momentum Strength', color='#bdc3c7', fontsize=12, loc='left')
        ax2.set_ylim(0, 100)
        ax2.set_ylabel('RSI')
        ax2.grid(True, linestyle='--', alpha=0.3)

        # Panel 3: Stochastic
        ax3.plot(df_plot['Date_mpl'], df_plot['Stoch_K'], color='#2980b9', linewidth=2, label='%K')
        ax3.plot(df_plot['Date_mpl'], df_plot['Stoch_D'], color='#e74c3c', linewidth=2, linestyle='--', label='%D')
        ax3.set_title('3. Stochastic Oscillator', color='#bdc3c7', fontsize=12, loc='left')
        ax3.set_ylabel('Stoch')
        ax3.set_ylim(0, 100)
        ax3.legend(loc='upper left')
        ax3.grid(True, linestyle='--', alpha=0.3)

        # Panel 4: MACD
        colors = ['#2ecc71' if val >= 0 else '#e74c3c' for val in df_plot['MACD_Hist']]
        ax4.bar(df_plot['Date_mpl'], df_plot['MACD_Hist'], color=colors, alpha=0.5, label='MACD Histogram')
        ax4.plot(df_plot['Date_mpl'], df_plot['MACD'], color='#3498db', linewidth=2, label='MACD Line')
        ax4.plot(df_plot['Date_mpl'], df_plot['Signal_Line'], color='#e67e22', linewidth=2, linestyle='--', label='Signal Line')
        ax4.set_title('4. MACD Momentum', color='#bdc3c7', fontsize=12, loc='left')
        ax4.set_ylabel('MACD')
        ax4.set_xlabel('Date')
        ax4.legend(loc='upper left')
        ax4.grid(True, linestyle='--', alpha=0.3)

        plt.tight_layout()
        plt.show()
        plt.style.use('ggplot')
