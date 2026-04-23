
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")
plt.rcParams['font.family'] = 'serif'

def plot_benchmark_volatility():
    # Define files and labels
    model_map = {
        'results/predictions_league_D1.csv': 'HRM-DQN (Ours)',
        'results/predictions_lstm_dueling_double_league_D1.csv': 'LSTM-DQN',
        'results/predictions_mlp_league_D1.csv': 'Standard DQN',
        'results/predictions_mlp_double_league_D1.csv': 'Double DQN',
        'results/predictions_mlp_dueling_double_league_D1.csv': 'Dueling DQN'
    }
    
    colors = {
        'HRM-DQN (Ours)': 'green',
        'LSTM-DQN': 'blue',
        'Standard DQN': 'orange',
        'Double DQN': 'gray',
        'Dueling DQN': 'red'
    }
    
    # Use rolling window
    WINDOW = 20

    plt.figure(figsize=(12, 6))
    
    for f, label in model_map.items():
        if os.path.exists(f):
            df = pd.read_csv(f)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            
            # Filter No Bet if necessary, though Profit is 0 so var is affected. 
            # Usually volatility of ACTIVE bets is what matters.
            # But "No Bet" is a valid action reducing portfolio volatility.
            # Let's keep all rows to show portfolio volatility.
            
            # Calculate Rolling Std Dev of Profit
            # Profit is per unit stake (mostly).
            rolling_vol = df['Profit'].rolling(window=WINDOW).std()
            
            plt.plot(df['Date'], rolling_vol, label=label, color=colors.get(label, 'black'), 
                     linewidth=2 if 'Ours' in label or 'LSTM' in label else 1, alpha=0.8)
            
    plt.title(f'Rolling Volatility (Std Dev of Profit, Window={WINDOW} Bets)', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Volatility (Std Dev)', fontsize=14)
    plt.legend(loc='upper right', frameon=True, framealpha=0.9)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/benchmark_volatility.png', dpi=300)
    plt.savefig('results/pdf/benchmark_volatility.pdf', dpi=300)
    print("Saved benchmark volatility plot.")

if __name__ == "__main__":
    os.makedirs('results/pdf', exist_ok=True)
    plot_benchmark_volatility()
