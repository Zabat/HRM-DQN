
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")
plt.rcParams['font.family'] = 'serif'

def plot_benchmark_evolution():
    # Define files and labels
    model_map = {
        'results/predictions_league_D1.csv': 'HRM-DQN (Ours)',
        'results/predictions_lstm_dueling_double_league_D1.csv': 'LSTM-DQN',
        'results/predictions_mlp_league_D1.csv': 'Standard DQN',
        'results/predictions_mlp_double_league_D1.csv': 'Double DQN',
        'results/predictions_mlp_dueling_double_league_D1.csv': 'Dueling DQN'
    }
    
    # Colors
    colors = {
        'HRM-DQN (Ours)': 'green',
        'LSTM-DQN': 'blue',
        'Standard DQN': 'orange',
        'Double DQN': 'gray',
        'Dueling DQN': 'red'
    }
    
    # Linestyles
    styles = {
        'HRM-DQN (Ours)': '-',
        'LSTM-DQN': '--',
        'Standard DQN': '-.',
        'Double DQN': ':',
        'Dueling DQN': ':'
    }

    plt.figure(figsize=(12, 8))
    
    for f, label in model_map.items():
        if os.path.exists(f):
            df = pd.read_csv(f)
            # Ensure chronological order
            # (Matches are already sorted by date in evaluate.py)
            
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            
            # We want to plot Cumulative Balance OVER TIME.
            # To handle multiple matches on same day or missing days, we can resample or just plot points.
            # Plotting every bet date is fine.
            
            dates = df['Date'].tolist()
            balances = df['Balance'].tolist()
            
            # Start from initial
            if len(dates) > 0:
                start_date = dates[0] - pd.Timedelta(days=1)
                dates = [start_date] + dates
                balances = [100.0] + balances
            
            plt.plot(dates, balances, label=label, color=colors.get(label, 'black'), 
                     linestyle= styles.get(label, '-'), linewidth=2.5 if 'Ours' in label or 'LSTM' in label else 1.5)
            
    plt.title('Bankroll Evolution: HRM-DQN vs Baselines (Bundesliga)', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Bankroll (Units)', fontsize=14)
    plt.axhline(100, color='black', linestyle='-', linewidth=1, alpha=0.5)
    plt.legend(loc='upper left', frameon=True, framealpha=0.9)
    # Rotate date labels
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/benchmark_bankroll_evolution.png', dpi=300)
    plt.savefig('results/pdf/benchmark_bankroll_evolution.pdf', dpi=300)
    print("Saved benchmark evolution plot to results/ and results/pdf/")

if __name__ == "__main__":
    os.makedirs('results/pdf', exist_ok=True)
    plot_benchmark_evolution()
