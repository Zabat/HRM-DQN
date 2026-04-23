#!/bin/bash

# Configuration
LEAGUE="D1"
EPOCHS=100

echo "=== STARTING BENCHMARK ON LEAGUE ${LEAGUE} ==="
mkdir -p results checkpoints

# 1. Standard DQN (MLP, No Dueling, No Double)
echo "[1/4] Training Standard DQN..."
python3 -m src.train --mode league --league $LEAGUE --epochs $EPOCHS \
    --model mlp

echo "Evaluating Standard DQN..."
python3 -m src.evaluate --mode league --league $LEAGUE \
    --model mlp

# 2. Double DQN (MLP, No Dueling, YES Double)
echo "[2/4] Training Double DQN..."
python3 -m src.train --mode league --league $LEAGUE --epochs $EPOCHS \
    --model mlp --double

echo "Evaluating Double DQN..."
python3 -m src.evaluate --mode league --league $LEAGUE \
    --model mlp --double

# 3. Dueling DQN (MLP, YES Dueling, YES Double)
echo "[3/4] Training Dueling DQN..."
python3 -m src.train --mode league --league $LEAGUE --epochs $EPOCHS \
    --model mlp --dueling --double

echo "Evaluating Dueling DQN..."
python3 -m src.evaluate --mode league --league $LEAGUE \
    --model mlp --dueling --double

# 4. LSTM DQN (LSTM, YES Dueling, YES Double)
echo "[4/4] Training LSTM DQN..."
python3 -m src.train --mode league --league $LEAGUE --epochs $EPOCHS \
    --model lstm --dueling --double

echo "Evaluating LSTM DQN..."
python3 -m src.evaluate --mode league --league $LEAGUE \
    --model lstm --dueling --double

echo "=== BENCHMARK COMPLETE ==="
echo "Results are saved in results/"
