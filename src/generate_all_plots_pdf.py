
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import glob
import math

# Set style for professional publication
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.4)
colors = sns.color_palette("deep")

OUTPUT_DIR = 'results/pdf'

def ensure_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data(league='D1'):
    path = f'results/predictions_league_{league}.csv'
    if not os.path.exists(path):
        if league == 'Global':
            path = 'results/predictions_global_all.csv'
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

# ==========================================
# 1. Combined Balance (from plot_combined.py)
# ==========================================
def plot_combined_balance_pdf():
    plt.figure(figsize=(12, 8))
    
    styles = {
        'Global': {'color': 'black', 'linewidth': 2.5, 'linestyle': '--'},
        'D1': {'color': 'green', 'linewidth': 2, 'label': 'Bundesliga (D1)'},
        'E0': {'color': 'blue', 'linewidth': 1.5, 'label': 'Premier League (E0)'},
        'F1': {'color': 'red', 'linewidth': 1.5, 'label': 'Ligue 1 (F1)'},
        'I1': {'color': 'purple', 'linewidth': 1.5, 'label': 'Serie A (I1)'},
        'SP1': {'color': 'orange', 'linewidth': 1.5, 'label': 'La Liga (SP1)'}
    }
    
    # Global
    if os.path.exists('results/predictions_global_all.csv'):
        df = pd.read_csv('results/predictions_global_all.csv')
        plt.plot(df['Balance'], label='Global Model', **styles['Global'])
        
    # Leagues
    league_files = sorted(glob.glob('results/predictions_league_*.csv'))
    for f in league_files:
        code = f.split('_')[-1].replace('.csv', '')
        df = pd.read_csv(f)
        style = styles.get(code, {})
        label = style.get('label', code)
        color = style.get('color', None)
        lw = style.get('linewidth', 1.5)
        plt.plot(df['Balance'], label=label, color=color, linewidth=lw)

    plt.title('Bankroll Evolution Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Bets', fontsize=12)
    plt.ylabel('Bankroll (Units)', fontsize=12)
    plt.axhline(y=100, color='gray', linestyle=':', alpha=0.5, label='Break-even (100)')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    output_path = f'{OUTPUT_DIR}/combined_balance_evolution.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")

# ==========================================
# 2. Manuscript Plots (from generate_manuscript_plots.py)
# ==========================================
def plot_roi_by_confidence_pdf(df, title):
    df['Confidence'] = df['Q_Values'].apply(lambda x: max(eval(str(x))))
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
    
    for i, p in enumerate(ax.patches):
        color = 'black'
        val = p.get_height()
        ax.annotate(f'{val:.1f}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom' if val > 0 else 'top', color=color, fontsize=12, fontweight='bold')
            
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/manuscript_1_roi_confidence.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def plot_calibration_pdf(df, title):
    def get_odds(row):
        act = row['Action']
        if act == 'Home': return row['OddsHome']
        if act == 'Draw': return row['OddsDraw']
        if act == 'Away': return row['OddsAway']
        return np.nan

    df['SelectedOdds'] = df.apply(get_odds, axis=1)
    df_bets = df.dropna(subset=['SelectedOdds']).copy()
    df_bets['ImpliedProb'] = 1 / df_bets['SelectedOdds']
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
    plt.savefig(f'{OUTPUT_DIR}/manuscript_2_calibration.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def plot_profit_by_action_pdf(df, title):
    plt.figure(figsize=(12, 7))
    actions = df['Action'].unique()
    for action in actions:
        subset = df[df['Action'] == action].copy()
        if len(subset) == 0: continue
        subset['CumProfit'] = subset['Profit'].cumsum()
        plt.plot(subset.index, subset['CumProfit'], label=f"{action} (n={len(subset)})", linewidth=2)
        
    plt.title(f'{title}: Cumulative Profit by Bet Type', fontweight='bold')
    plt.xlabel('Bet Number (Chronological)')
    plt.ylabel('Profit (Units)')
    plt.axhline(0, color='black', linewidth=1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/manuscript_3_profit_breakdown.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def plot_rolling_metrics_pdf(df, title):
    window = 50
    df['RollingROI'] = df['Profit'].rolling(window).mean() * 100 
    
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
    plt.savefig(f'{OUTPUT_DIR}/manuscript_4_rolling_stability.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def plot_action_comparison_pdf(df1, label1, df2, label2):
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
    plt.savefig(f'{OUTPUT_DIR}/manuscript_5_strategy_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# ==========================================
# 3. Extra Plots (from generate_extra_plots.py)
# ==========================================
def plot_threshold_curve_pdf(df, title):
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
    plt.savefig(f'{OUTPUT_DIR}/manuscript_6_threshold_optimization.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def plot_odds_performance_pdf(df, title):
    def get_odds(row):
        act = row['Action']
        if act == 'Home': return row['OddsHome']
        if act == 'Draw': return row['OddsDraw']
        if act == 'Away': return row['OddsAway']
        return np.nan

    df['SelectedOdds'] = df.apply(get_odds, axis=1)
    df = df.dropna(subset=['SelectedOdds']).copy()
    
    bins = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 10.0]
    labels = ['<1.5', '1.5-2.0', '2.0-2.5', '2.5-3.0', '3.0-4.0', '>4.0']
    df['OddsRange'] = pd.cut(df['SelectedOdds'], bins=bins, labels=labels)
    
    grouped = df.groupby(['Action', 'OddsRange'], observed=True).apply(
        lambda x: (x['Profit'].sum() / len(x)) * 100 if len(x) > 10 else 0
    ).unstack()
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(grouped, annot=True, center=0, cmap='RdYlGn', fmt='.1f')
    plt.title(f'{title}: ROI Heatmap (Action vs Odds)', fontweight='bold')
    plt.ylabel('Bet Type')
    plt.xlabel('Odds Range')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/manuscript_7_odds_heatmap.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    ensure_dir()
    
    # Load Main Data
    d1 = load_data('D1')
    global_df = load_data('Global')
    
    if d1 is None or global_df is None:
        print("Error: Could not load data.")
        return

    print("Generating PDF Plots...")
    
    plot_combined_balance_pdf()
    
    plot_roi_by_confidence_pdf(d1, 'Bundesliga (D1)')
    plot_calibration_pdf(d1, 'Bundesliga (D1)')
    plot_profit_by_action_pdf(d1, 'Bundesliga (D1)')
    plot_rolling_metrics_pdf(d1, 'Bundesliga (D1)')
    plot_action_comparison_pdf(d1, 'Bundesliga (Pros)', global_df, 'Global (Baseline)')
    
    plot_threshold_curve_pdf(d1, 'Bundesliga (D1)')
    plot_odds_performance_pdf(d1, 'Bundesliga (D1)')
    
    print(f"All 8 PDF plots saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
