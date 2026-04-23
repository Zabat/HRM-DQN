
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

def plot_combined_balance():
    plt.figure(figsize=(12, 8))
    
    # Define styles for distinction
    styles = {
        'Global': {'color': 'black', 'linewidth': 2.5, 'linestyle': '--'},
        'D1': {'color': 'green', 'linewidth': 2, 'label': 'Bundesliga (D1)'},   # Profitable
        'E0': {'color': 'blue', 'linewidth': 1.5, 'label': 'Premier League (E0)'},
        'F1': {'color': 'red', 'linewidth': 1.5, 'label': 'Ligue 1 (F1)'},
        'I1': {'color': 'purple', 'linewidth': 1.5, 'label': 'Serie A (I1)'},
        'SP1': {'color': 'orange', 'linewidth': 1.5, 'label': 'La Liga (SP1)'}
    }
    
    # 1. Global
    if os.path.exists('results/predictions_global_all.csv'):
        df = pd.read_csv('results/predictions_global_all.csv')
        # Re-center to start at 0 profit for clearer comparison? 
        # Or just raw bankroll. Request said "bankroll evolution".
        # Let's use Profit Accumulation (Starting at 0) to compare apples to apples if bankrolls differed, 
        # though they all start at 100. Let's strictly plot Bankroll (Current Balance).
        plt.plot(df['Balance'], label='Global Model', **styles['Global'])
        
    # 2. Leagues
    league_files = sorted(glob.glob('results/predictions_league_*.csv'))
    for f in league_files:
        # Extract league code from filename "predictions_league_XX.csv"
        code = f.split('_')[-1].replace('.csv', '')
        
        df = pd.read_csv(f)
        style = styles.get(code, {})
        label = style.get('label', code)
        color = style.get('color', None)
        lw = style.get('linewidth', 1.5)
        
        plt.plot(df['Balance'], label=label, color=color, linewidth=lw)

    plt.title('Bankroll Evolution Comparison (100 Epochs)', fontsize=16)
    plt.xlabel('Number of Bets', fontsize=12)
    plt.ylabel('Bankroll (Units)', fontsize=12)
    plt.axhline(y=100, color='gray', linestyle=':', alpha=0.5, label='Break-even (100)')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    output_path = 'results/combined_balance_evolution.png'
    plt.savefig(output_path, dpi=300)
    print(f"Saved combined plot to {output_path}")

if __name__ == "__main__":
    plot_combined_balance()
