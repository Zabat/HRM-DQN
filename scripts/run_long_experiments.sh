
#!/bin/bash

# Long Experiment: 100 Epochs per model to ensure convergence

# 1. Train Global Model
echo "Training Global Model (100 Epochs)..."
python3 -m src.train --mode global --epochs 100

# 2. Train League Models
echo "Training League Models (100 Epochs)..."
for league in D1 E0 F1 SP1 I1
do
    echo "Training $league..."
    python3 -m src.train --mode league --league $league --epochs 100
done

# 3. Evaluate Global Model on All Data
echo "Evaluating Global Model..."
python3 -m src.evaluate --mode global --model_path checkpoints/hrm_dqn_global.pth

# 4. Evaluate League Models on their Leagues
for league in D1 E0 F1 SP1 I1
do
    echo "Evaluating $league Model on $league..."
    python3 -m src.evaluate --mode league --league $league --model_path checkpoints/hrm_dqn_league_${league}.pth
done

echo "Long Experiments Complete. Results in results/ folder."
