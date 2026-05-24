# =====================================================================
# ⚙️ MASTER CONFIGURATION CENTER (USER INPUTS & BUDGETS)
# =====================================================================

old_investments = {
    'AMD': 100.38, 'AMZN': 100.00, 'CAT': 63.9, 'GS': 48.8,
    'GLD': 121.38, 'MXL': 2.45, 'AAPL': 10.00
}

stock_hedge_tickers = ['MNST', 'BKNG', 'AZO', 'SHW', 'JPM', 'ORCL', 'V', "ABEO", "AGEN", "XOM"]
macro_hedge_tickers = ['TLT', 'DBC', 'USO'] # eToro Macro Hedge

stock_budget = 470.00
macro_budget = 450.00

VIX_THRESHOLD = 20.0
risk_free_rate = 0.04

# Προβολές & Προσομοιώσεις
sim_years = 5
num_sims = 100000
window = 60
projection_days = 180
