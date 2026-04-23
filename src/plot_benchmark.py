
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")

def plot_benchmark():
    results = []
    
    # Define mapping of filenames to readable names
    # Pattern: predictions_{model}_{league}.csv
    # e.g. predictions_mlp_league_D1.csv -> Standard DQN
    # predictions_mlp_double_league_D1.csv -> Double DQN
    # predictions_mlp_dueling_double_league_D1.csv -> Dueling DQN
    # predictions_lstm_dueling_double_league_D1.csv -> LSTM DQN
    # predictions_league_D1.csv -> HRM DQN (Legacy name)
    
    # We look for files in results/
    files = glob.glob('results/predictions_*.csv')
    
    for f in files:
        if 'global' in f and 'D1' not in f: continue # Skip global if we benchmarking D1
        if 'D1' not in f: continue # Strict D1 benchmark
        
        df = pd.read_csv(f)
        total_bets = len(df[df['Action'] != 'NoBet'])
        if total_bets == 0: continue
        
        profit = df['Profit'].sum()
        roi = (profit / total_bets) * 100
        
        # Determine Name
        name = "Unknown"
        if "mlp_league" in f and "double" not in f: name = "Standard DQN"
        elif "mlp_double" in f and "dueling" not in f: name = "Double DQN"
        elif "mlp_dueling_double" in f: name = "Dueling DQN"
        elif "lstm_dueling_double" in f: name = "LSTM DQN"
        elif "league_D1" in f and "mlp" not in f and "lstm" not in f: name = "HRM-DQN (Ours)"
        
        if name == "Unknown": continue
        
        results.append({'Model': name, 'ROI': roi})
        
    df_res = pd.DataFrame(results)
    
    # Sort for visual consistency
    order = ["Standard DQN", "Double DQN", "Dueling DQN", "LSTM DQN", "HRM-DQN (Ours)"]
    # Filter to only those present
    order = [o for o in order if o in df_res['Model'].values]
    
    plt.figure(figsize=(10, 6))
    colors = ['gray' if 'Ours' not in m else 'green' for m in order]
    ax = sns.barplot(x='Model', y='ROI', data=df_res, order=order, palette=colors)
    
    plt.title('Benchmark: HRM-DQN vs Baselines (Bundesliga)', fontweight='bold')
    plt.ylabel('ROI (%)')
    plt.xlabel('Model Architecture')
    plt.axhline(0, color='black', linewidth=1)
    
    # Add labels
    for i, p in enumerate(ax.patches):
        val = p.get_height()
        ax.annotate(f'{val:.2f}%', (p.get_x() + p.get_width() / 2., val),
                    ha='center', va='bottom' if val > 0 else 'top', 
                    fontsize=12, fontweight='bold', color='black')
        
    plt.tight_layout()
    plt.savefig('results/benchmark_comparison.png', dpi=300)
    plt.savefig('results/pdf/benchmark_comparison.pdf', dpi=300)
    print("Saved benchmark plot to results/")

if __name__ == "__main__":
    plot_benchmark()
