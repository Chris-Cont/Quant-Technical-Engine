# =====================================================================
# ⚙️ MASTER CONTROL CENTER (TICKER MANAGEMENT)
# =====================================================================
import os

print("Updating Master Ticker List in system configuration...")

# 1. Core Portfolio (Tech & Growth)
my_portfolio = ['AMZN', 'AAPL', 'AMD', 'GS', 'CAT', 'MXL', 'GLD']

# 2. Defensive Portfolio (Blue Chips & Hedges)
my_defensive = [
    'VIST', 'MNST', 'BKNG', 'AZO', 'SHW', 'JPM', 'V', 'ORCL', 'NVDA', 'TSLA', 'META',
    'BP', 'SHEL', 'BARC.L', 'AXP', 'BAC', 'KHC', 'KR', 'MCO', 'GOOGL', 'UNH', 'SATS', 
    'TWLO', 'GDS', 'PFGC', 'CVNA', 'DKS', 'WBD', 'ADMA', 'SHC', 'CVS', 'LYV', 'HUT', 
    'FUN', 'KVUE', 'REZI', 'CXM', 'RSP', 'TER', 'CRCL', 'NUE', 'MU', 'SHOP', 'COIN', 
    'HOOD', 'TEM', 'RBLX', 'ACHR', 'KTOS', 'BMNR', 'TSM', 'DE', 'TXG', 'BEAM', 'CVX', 
    'XOM', 'STM'
]

# Remove any accidental duplicates (e.g., 'MU', 'RBLX' were duplicated in the raw list)
my_defensive = list(dict.fromkeys(my_defensive))

# 3. Commodities & Energy
my_commodities = ['GLD', 'DBC', 'USO']

# 4. US Bonds
my_bonds = ['TLT', 'IEF', 'BND']

# 5. Additional Tech Companies
my_tech = []

# --- CENTRAL MASTER LIST ACCESSED BY ALL SCRIPTS ---
MASTER_TICKERS = my_portfolio + my_defensive + my_commodities + my_bonds + my_tech

# --- MONTE CARLO SIMULATION SETTINGS ---
# Asset focus for long-term macroeconomic simulation
MACRO_FOCUS = 'SPY' 

if __name__ == "__main__":
    print(f"✅ Successfully loaded {len(MASTER_TICKERS)} unique Tickers into the configuration module!")
    print(f"✅ Macro Focus Asset designated for Monte Carlo: {MACRO_FOCUS}")
    print("=" * 70)
