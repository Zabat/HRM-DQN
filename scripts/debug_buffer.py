
from src.replay_buffer import PrioritizedReplayBuffer
import numpy as np

try:
    buffer = PrioritizedReplayBuffer(100)
    
    # Push synthetic data
    for i in range(50):
        buffer.push(np.random.randn(10), 1, 1.0, np.random.randn(10), False)
        
    print(f"Filled buffer with {len(buffer)} items")
    
    # Sample
    states, actions, rewards, next_states, dones, idxs, weights = buffer.sample(10)
    
    print("Sampled Batch sizes:")
    print("States:", states.shape)
    print("Weights:", weights.shape)
    
    # Update priorities
    errors = np.abs(np.random.randn(10))
    buffer.update_priority(idxs, errors)
    print("Priorities updated.")
    
    print("Success!")
except Exception as e:
    print("Error:", e)
    import traceback
    traceback.print_exc()
