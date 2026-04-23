
import pandas as pd
import numpy as np
import glob

def print_metrics(file_path):
    try:
        df = pd.read_csv(file_path)
        
        # Filter NoBet
        bets = df[df['Action'] != 'NoBet']
        n_bets = len(bets)
        
        if n_bets == 0:
            print(f"File: {file_path} | No Bets placed.")
            return

        profit = bets['Profit'].sum()
        roi = (profit / n_bets) * 100
        
        # Hit Rate (Profit > 0)
        wins = bets[bets['Profit'] > 0]
        hit_rate = (len(wins) / n_bets) * 100
        
        balance = df['Balance'].values
        peak = np.maximum.accumulate(balance)
        drawdown = (peak - balance) / peak
        max_dd = drawdown.max() * 100
        
        print(f"File: {file_path}")
        print(f"  Bets: {n_bets}/{len(df)} ({n_bets/len(df)*100:.1f}%)")
        print(f"  Profit: {profit:.2f} units")
        print(f"  ROI: {roi:.2f}%")
        print(f"  Hit Rate: {hit_rate:.2f}%")
        print(f"  Max Drawdown: {max_dd:.2f}%")
        print("-" * 30)
        
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

print("=== FINAL RESULTS ===")
# Global
print_metrics('results/predictions_global_all.csv')

# Leagues
for f in sorted(glob.glob('results/predictions_league_*.csv')):
    print_metrics(f)
