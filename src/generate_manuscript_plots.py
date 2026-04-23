
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set style for professional publication
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.4)
colors = sns.color_palette("deep")

def load_data(league='D1'):
    path = f'results/predictions_league_{league}.csv'
    if not os.path.exists(path):
        # Fallback for global
        if league == 'Global':
            path = 'results/predictions_global_all.csv'
    
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def plot_roi_by_confidence(df, title, output_path):
    """Plot 1: ROI by Q-Value Confidence Interval"""
    # Max Q-value as proxy for confidence
    df['Confidence'] = df['Q_Values'].apply(lambda x: max(eval(str(x))))
    
    # Binning
    bins = np.percentile(df['Confidence'], [0, 25, 50, 75, 100])
    df['Conf_Bin'] = pd.cut(df['Confidence'], bins, labels=['Low', 'Medium', 'High', 'Very High'])
    
    roi_per_bin = df.groupby('Conf_Bin', observed=True).apply(
        lambda x: (x['Profit'].sum() / len(x)) * 100 if len(x) > 0 else 0
    ).reset_index(name='ROI')

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Conf_Bin', y='ROI', data=roi_per_bin, palette='RdYlGn')
    plt.title(f'{title}: ROI by Model Confidence', fontweight='bold')
    plt.axhline(0, color='black', linewidth=1)
    plt.xlabel('Confidence Level (Q-Value Quartiles)')
    plt.ylabel('ROI (%)')
    
    # Add values
    for i, p in enumerate(ax.patches):
        color = 'black'
        val = p.get_height()
        ax.annotate(f'{val:.1f}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom' if val > 0 else 'top', color=color, fontsize=12, fontweight='bold')
            
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_calibration(df, title, output_path):
    """Plot 2: Odds Calibration (Implied vs Actual Win Rate)"""
    # Focusing on Home/Draw/Away bets for simplicity in odds extraction
    # Map Action to the correct Odds column
    def get_odds(row):
        act = row['Action']
        if act == 'Home': return row['OddsHome']
        if act == 'Draw': return row['OddsDraw']
        if act == 'Away': return row['OddsAway']
        return np.nan

    df['SelectedOdds'] = df.apply(get_odds, axis=1)
    df_bets = df.dropna(subset=['SelectedOdds']).copy()
    
    # Implied Prob
    df_bets['ImpliedProb'] = 1 / df_bets['SelectedOdds']
    
    # Bins
    df_bets['ProbBin'] = pd.cut(df_bets['ImpliedProb'], bins=np.arange(0, 1.1, 0.1))
    
    calibration = df_bets.groupby('ProbBin', observed=True).apply(
        lambda x: pd.Series({
            'Predicted': x['ImpliedProb'].mean(),
            'Actual': (x['Profit'] > 0).mean()
        })
    ).reset_index()

    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    sns.lineplot(x='Predicted', y='Actual', data=calibration, marker='o', markersize=10, linewidth=2.5, color='darkblue')
    
    plt.title(f'{title}: Market Odds vs Actual Win Rate', fontweight='bold')
    plt.xlabel('Implied Probability (1/Odds)')
    plt.ylabel('Actual Win Rate')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_profit_by_action(df, title, output_path):
    """Plot 3: Cumulative Profit by Action Type"""
    plt.figure(figsize=(12, 7))
    
    actions = df['Action'].unique()
    for action in actions:
        subset = df[df['Action'] == action].copy()
        if len(subset) == 0: continue
        
        subset['CumProfit'] = subset['Profit'].cumsum()
        # Align index to total bets for x-axis
        plt.plot(subset.index, subset['CumProfit'], label=f"{action} (n={len(subset)})", linewidth=2)
        
    plt.title(f'{title}: Cumulative Profit by Bet Type', fontweight='bold')
    plt.xlabel('Bet Number (Chronological)')
    plt.ylabel('Profit (Units)')
    plt.axhline(0, color='black', linewidth=1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_rolling_metrics(df, title, output_path):
    """Plot 4: Rolling 50-Match ROI and Hit Rate"""
    window = 50
    df['RollingROI'] = df['Profit'].rolling(window).mean() * 100 # Profit per bet * 100 = ROI% roughly per bet
    # Actually strictly: (Sum Profit / Sum Stake) * 100. Assuming Stake=1.
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(df['RollingROI'], color='teal', label=f'{window}-Bet Rolling ROI', linewidth=2)
    plt.fill_between(df.index, df['RollingROI'], 0, where=(df['RollingROI']>=0), color='green', alpha=0.1)
    plt.fill_between(df.index, df['RollingROI'], 0, where=(df['RollingROI']<0), color='red', alpha=0.1)
    
    plt.title(f'{title}: Stability Analysis (Rolling {window}-Bet ROI)', fontweight='bold')
    plt.xlabel('Bet Number')
    plt.ylabel('ROI (%)')
    plt.axhline(0, color='black', linewidth=1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_action_comparison(df1, label1, df2, label2, output_path):
    """Plot 5: Action Distribution Comparison"""
    c1 = df1['Action'].value_counts(normalize=True).reset_index()
    c1.columns = ['Action', 'Proportion']
    c1['Model'] = label1
    
    c2 = df2['Action'].value_counts(normalize=True).reset_index()
    c2.columns = ['Action', 'Proportion']
    c2['Model'] = label2
    
    combined = pd.concat([c1, c2])
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Action', y='Proportion', hue='Model', data=combined, palette='muted')
    plt.title('Strategy Comparison: Action Distribution', fontweight='bold')
    plt.ylabel('Frequency (%)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def main():
    os.makedirs('results', exist_ok=True)
    
    # Load Data
    d1 = load_data('D1') # Profitable
    global_df = load_data('Global') # Baseline
    
    if d1 is None or global_df is None:
        print("Error: Could not load datafiles.")
        return

    print("Generating Plots...")
    
    # 1. ROI by Confidence (D1)
    plot_roi_by_confidence(d1, 'Bundesliga (D1)', 'results/manuscript_1_roi_confidence.png')
    
    # 2. Calibration (D1)
    plot_calibration(d1, 'Bundesliga (D1)', 'results/manuscript_2_calibration.png')
    
    # 3. Accumulated Profit by Action (D1)
    plot_profit_by_action(d1, 'Bundesliga (D1)', 'results/manuscript_3_profit_breakdown.png')
    
    # 4. Stability (Rolling ROI)
    plot_rolling_metrics(d1, 'Bundesliga (D1)', 'results/manuscript_4_rolling_stability.png')
    
    # 5. Comparison
    plot_action_comparison(d1, 'Bundesliga (Pros)', global_df, 'Global (Baseline)', 'results/manuscript_5_strategy_comparison.png')
    
    print("Done. Saved 5 plots to results/")

if __name__ == "__main__":
    main()
