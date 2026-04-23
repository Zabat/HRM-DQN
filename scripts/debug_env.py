
from src.data_loader import FootballDataset
from src.environment import BettingEnvironment
import numpy as np

try:
    ds = FootballDataset('matchs_3_dernieres_saisons.csv')
    env = BettingEnvironment(ds, usage='train')
    
    state = env.reset()
    print("Initial State Shape:", state.shape)
    
    # Test random actions
    print("Testing 10 random steps...")
    total_reward = 0
    for _ in range(10):
        action = np.random.randint(0, 7) # 7 actions
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        print(f"Action: {action}, Reward: {reward:.2f}, Done: {done}")
        if done:
            break
            
    print("Total Reward:", total_reward)
    print("Raw Odds Check:", env.raw_df[['RawOddHome', 'OddHome']].head())
    print("Success!")
except Exception as e:
    print("Error:", e)
    import traceback
    traceback.print_exc()
