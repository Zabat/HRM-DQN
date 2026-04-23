
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.4)

def load_data(league='D1'):
    path = f'results/predictions_league_{league}.csv'
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def plot_threshold_curve(df, title, output_path):
    """
    Plot Profit and ROI as a function of Confidence Threshold.
    Filters bets where Max(Q) > Threshold.
    """
    df['Confidence'] = df['Q_Values'].apply(lambda x: max(eval(str(x))))
    
    thresholds = np.linspace(df['Confidence'].min(), df['Confidence'].max(), 50)
    results = []
    
    for t in thresholds:
        subset = df[df['Confidence'] >= t]
        if len(subset) == 0: continue
        
        profit = subset['Profit'].sum()
        roi = (profit / len(subset)) * 100
        n_bets = len(subset)
        
        results.append({'Threshold': t, 'Profit': profit, 'ROI': roi, 'Bets': n_bets})
        
    res_df = pd.DataFrame(results)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Minimum Confidence Threshold (Q-Value)')
    ax1.set_ylabel('Total Profit (Units)', color=color)
    ax1.plot(res_df['Threshold'], res_df['Profit'], color=color, linewidth=3, label='Profit')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(False)
    
    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('ROI (%)', color=color)
    ax2.plot(res_df['Threshold'], res_df['ROI'], color=color, linestyle='--', linewidth=2, label='ROI')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.grid(True, alpha=0.3)
    
    plt.title(f'{title}: Optimization by Thresholding', fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_odds_performance(df, title, output_path):
    """
    Heatmap of ROI by Action and Odds Bucket.
    """
    def get_odds(row):
        act = row['Action']
        if act == 'Home': return row['OddsHome']
        if act == 'Draw': return row['OddsDraw']
        if act == 'Away': return row['OddsAway']
        return np.nan

    df['SelectedOdds'] = df.apply(get_odds, axis=1)
    df = df.dropna(subset=['SelectedOdds']).copy()
    
    # Simple Buckets
    bins = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 10.0]
    labels = ['<1.5', '1.5-2.0', '2.0-2.5', '2.5-3.0', '3.0-4.0', '>4.0']
    df['OddsRange'] = pd.cut(df['SelectedOdds'], bins=bins, labels=labels)
    
    # Calculate ROI per group
    grouped = df.groupby(['Action', 'OddsRange'], observed=True).apply(
        lambda x: (x['Profit'].sum() / len(x)) * 100 if len(x) > 10 else 0 # Threshold 10 bets to avoid noise
    ).unstack()
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(grouped, annot=True, center=0, cmap='RdYlGn', fmt='.1f')
    plt.title(f'{title}: ROI Heatmap (Action vs Odds)', fontweight='bold')
    plt.ylabel('Bet Type')
    plt.xlabel('Odds Range')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def main():
    d1 = load_data('D1')
    if d1 is None: return

    print("Generating Extra Plots...")
    plot_threshold_curve(d1, 'Bundesliga (D1)', 'results/manuscript_6_threshold_optimization.png')
    plot_odds_performance(d1, 'Bundesliga (D1)', 'results/manuscript_7_odds_heatmap.png')
    print("Done.")

if __name__ == "__main__":
    main()
