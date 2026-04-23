
from src.data_loader import FootballDataset
import pandas as pd

try:
    ds = FootballDataset('matchs_3_dernieres_saisons.csv')
    train, test = ds.get_train_test()
    
    print("Train Shape:", train.shape)
    print("Test Shape:", test.shape)
    print("Columns:", train.columns.tolist())
    
    # Check if Rolling Stats exist
    print("Checking rolling stats sample:")
    print(train[['HomeTeam', 'MatchDate', 'AvgGF_Last5_Home']].head(10))
    
    # Check Odds
    print("Checking Odds sample:")
    print(train[['OddHome', 'OddDraw', 'OddAway', 'Odd1X', 'Odd12', 'OddX2']].head())
    
    print("Success!")
except Exception as e:
    print("Error:", e)
    import traceback
    traceback.print_exc()
