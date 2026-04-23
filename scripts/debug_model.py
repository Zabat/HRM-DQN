
import torch
from src.data_loader import FootballDataset
from src.models import HRM_DQN

try:
    ds = FootballDataset('matchs_3_dernieres_saisons.csv')
    
    # Get Max Indices for Embeddings
    n_teams = len(ds.le_team.classes_)
    n_leagues = len(ds.le_division.classes_)
    print(f"Teams: {n_teams}, Leagues: {n_leagues}")
    
    # Instantiate Model
    model = HRM_DQN(match_dim=16, odds_dim=8, n_teams=n_teams, n_leagues=n_leagues)
    
    # Get a batch
    train_df, _ = ds.get_train_test()
    features, _ = ds.get_features(train_df.head(10))
    
    # Convert to Tensor
    input_tensor = torch.FloatTensor(features)
    print("Input Tensor Shape:", input_tensor.shape)
    
    # Forward Pass
    q_vals = model(input_tensor)
    print("Q Values Shape:", q_vals.shape)
    print("Q Values Sample:", q_vals[0].detach().numpy())
    
    print("Success!")
except Exception as e:
    print("Error:", e)
    import traceback
    traceback.print_exc()
