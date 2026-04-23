
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import os

class FootballDataset:
    def __init__(self, csv_path, seq_len=10):
        self.csv_path = csv_path
        self.seq_len = seq_len  # Length of history for LSTM (if we were using raw sequences)
        # But HRM usually takes a feature vector constructed from history.
        # We will compute rolling averages as "historical features".
        
        self.data = self.load_data()
        self.preprocess()
        
    def load_data(self):
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"File not found: {self.csv_path}")
        return pd.read_csv(self.csv_path)

    def preprocess(self):
        # 1. Sort by Date
        self.data['MatchDate'] = pd.to_datetime(self.data['MatchDate'])
        self.data = self.data.sort_values('MatchDate').reset_index(drop=True)
        
        # 2. Calculate Derived Odds (Double Chance)
        self._calculate_double_chance_odds()
        
        # 3. Calculate Rolling Stats (History)
        self._calculate_rolling_stats()
        
        # 4. Encode Categorical Metadata
        self._encode_metadata()
        
        # 5. Normalize Numerical Features
        self._normalize_features()
        
        print(f"Data preprocessed. Shape: {self.data.shape}")
        
    def _calculate_double_chance_odds(self):
        # 1X = 1 / (1/H + 1/D)
        # 12 = 1 / (1/H + 1/A)
        # X2 = 1 / (1/D + 1/A)
        
        # Use simple margin-free approximation or market margin handling?
        # Simple probability summation is standard for approximation.
        
        # Safety for zero division (though odds shouldn't be 0)
        odds_h = self.data['OddHome']
        odds_d = self.data['OddDraw']
        odds_a = self.data['OddAway']
        
        self.data['Odd1X'] = 1 / ((1/odds_h) + (1/odds_d))
        self.data['Odd12'] = 1 / ((1/odds_h) + (1/odds_a))
        self.data['OddX2'] = 1 / ((1/odds_d) + (1/odds_a))
        
        # Fill NaNs if any odds were missing
        self.data['Odd1X'] = self.data['Odd1X'].fillna(1.0) # Default to 1.0 (no profit)
        self.data['Odd12'] = self.data['Odd12'].fillna(1.0)
        self.data['OddX2'] = self.data['OddX2'].fillna(1.0)

    def _calculate_rolling_stats(self, window=5):
        # We need to construct a robust team-level history
        # Create a long-format DataFrame: Date, Team, GF, GA, Shots, ShotsTarget, Corners
        
        home_df = self.data[['MatchDate', 'HomeTeam', 'FTHome', 'FTAway', 'HomeShots', 'HomeTarget', 'HomeCorners']].copy()
        home_df.columns = ['Date', 'Team', 'GF', 'GA', 'Shots', 'SoT', 'Corners']
        
        away_df = self.data[['MatchDate', 'AwayTeam', 'FTAway', 'FTHome', 'AwayShots', 'AwayTarget', 'AwayCorners']].copy()
        away_df.columns = ['Date', 'Team', 'GF', 'GA', 'Shots', 'SoT', 'Corners']
        
        # Concatenate and sort
        team_stats = pd.concat([home_df, away_df]).sort_values('Date')
        
        # Calculate Rolling Means per Team
        cols_to_roll = ['GF', 'GA', 'Shots', 'SoT', 'Corners']
        
        # Shift 1 to avoid data leakage (current match stats shouldn't be known)
        rolling_stats = team_stats.groupby('Team')[cols_to_roll].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        
        rolling_stats.columns = [f'Avg{c}_Last{window}' for c in cols_to_roll]
        team_stats = pd.concat([team_stats[['Date', 'Team']], rolling_stats], axis=1)
        
        # Merge back to Main Data
        # We need to merge specific 'Avg' columns for HomeTeam and AwayTeam
        
        # Prepare helper for merging
        team_stats['join_key'] = team_stats['Date'].astype(str) + "_" + team_stats['Team']
        
        # Add keys to main data
        self.data['join_key_home'] = self.data['MatchDate'].astype(str) + "_" + self.data['HomeTeam']
        self.data['join_key_away'] = self.data['MatchDate'].astype(str) + "_" + self.data['AwayTeam']
        
        # Map values
        stat_cols = [f'Avg{c}_Last{window}' for c in cols_to_roll]
        
        # Create a dictionary for fast lookup or use merge
        # Using merge is safer
        stats_map = team_stats.set_index('join_key')[stat_cols]
        
        # Join for Home
        self.data = self.data.join(stats_map.add_suffix('_Home'), on='join_key_home')
        # Join for Away
        self.data = self.data.join(stats_map.add_suffix('_Away'), on='join_key_away')
        
        # Fill NaNs (first matches of team) with 0 or league avg. 0 is safe for NN.
        self.data.fillna(0, inplace=True)
        
        # Cleanup
        self.data.drop(['join_key_home', 'join_key_away'], axis=1, inplace=True)
        
    def _encode_metadata(self):
        self.le_division = LabelEncoder()
        self.data['Division_Code'] = self.le_division.fit_transform(self.data['Division'])
        
        self.le_team = LabelEncoder()
        all_teams = pd.concat([self.data['HomeTeam'], self.data['AwayTeam']]).unique()
        self.le_team.fit(all_teams)
        self.data['HomeTeam_Code'] = self.le_team.transform(self.data['HomeTeam'])
        self.data['AwayTeam_Code'] = self.le_team.transform(self.data['AwayTeam'])

    def _normalize_features(self):
        # Identify columns to scale
        # Match Features: Elo, Form, Rolling Stats
        self.match_features_cols = [
            'HomeElo', 'AwayElo', 
            'Form3Home', 'Form5Home', 'Form3Away', 'Form5Away',
            'AvgGF_Last5_Home', 'AvgGA_Last5_Home', 'AvgShots_Last5_Home', 'AvgSoT_Last5_Home', 'AvgCorners_Last5_Home',
            'AvgGF_Last5_Away', 'AvgGA_Last5_Away', 'AvgShots_Last5_Away', 'AvgSoT_Last5_Away', 'AvgCorners_Last5_Away'
        ]
        
        # Odds Features: 
        self.odds_cols = [
            'OddHome', 'OddDraw', 'OddAway', 
            'Odd1X', 'Odd12', 'OddX2',
            'Over25', 'Under25'
        ]
        
        # Metadata Features: (Already Encoded or Raw)
        self.meta_cols = ['Division_Code', 'HomeTeam_Code', 'AwayTeam_Code']
        
        # SAVE RAW ODDS for Reward Calculation
        for col in self.odds_cols:
            self.data[f'Raw{col}'] = self.data[col]
        
        # Scale Numerical
        scaler = MinMaxScaler()
        self.data[self.match_features_cols + self.odds_cols] = scaler.fit_transform(self.data[self.match_features_cols + self.odds_cols])
        
    def get_train_test(self):
        # Training: 2022-2023, 2023-2024
        # Testing: 2024-2025
        
        train_df = self.data[self.data['Season'].isin(['2022-2023', '2023-2024'])].copy()
        test_df = self.data[self.data['Season'] == '2024-2025'].copy()
        
        return train_df, test_df

    def get_features(self, df):
        # Return S_t and Metadata
        # Concatenate everything for the model input
        # Note: In HRM, we might want separate inputs.
        # Here we return a dictionary or raw arrays.
        
        match_feats = df[self.match_features_cols].values
        odds_feats = df[self.odds_cols].values
        meta_feats = df[self.meta_cols].values
        
        # Full state vector (concat)
        state = np.hstack([match_feats, odds_feats, meta_feats])
        return state, df  # Return df for reward calculation (needs raw odds)
