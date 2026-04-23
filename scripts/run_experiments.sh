
#!/bin/bash

# 1. Train Global Model
echo "Training Global Model..."
python3 -m src.train --mode global --epochs 20

# 2. Train League Models
echo "Training League Models..."
for league in D1 E0 F1 SP1 I1
do
    echo "Training $league..."
    python3 -m src.train --mode league --league $league --epochs 20
done

# 3. Evaluate Global Model on All Data
echo "Evaluating Global Model..."
python3 -m src.evaluate --mode global --model_path checkpoints/hrm_dqn_global.pth

# 4. Evaluate Global Model on Each League (Robustness)
for league in D1 E0 F1 SP1 I1
do
    echo "Evaluating Global Model on $league..."
    python3 -m src.evaluate --mode league --league $league --model_path checkpoints/hrm_dqn_global.pth
done

# 5. Evaluate League Models on their Leagues
for league in D1 E0 F1 SP1 I1
do
    echo "Evaluating $league Model on $league..."
    python3 -m src.evaluate --mode league --league $league --model_path checkpoints/hrm_dqn_${league}.pth
done

echo "Experiments Complete. Results in results/ folder."
