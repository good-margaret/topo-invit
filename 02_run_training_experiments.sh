!/bin/bash

PROBLEM="tsp"
NODES=50       # Тренируемся всегда на 50, тестируем на 100+
EPOCHS=100     # Для статьи обычно 400+, для теста хватит 100
GPU_ID=0

# Сетка гиперпараметров (Grid Search)
# Format: "BETA LAMBDA_GAP"
CONFIGS=(
  "0.0 0.0"   # 1. Pure Baseline (No Topology)
  "0.0 0.05"  # 2. Reward Shaping Only
  "0.1 0.0"   # 3. Encoder Regularization Only (RTD-Lite)
  "0.1 0.05"  # 4. Full Topo-INVIT v2
)

echo "=================================================="
echo "[*] ЭТАП 2: Серия экспериментов по обучению"
echo "=================================================="

for CONFIG in "${CONFIGS[@]}"
do
    # Читаем параметры из строки
    set -- $CONFIG
    BETA=$1
    LAMBDA=$2

    echo "[*] Запуск: Beta=${BETA}, Lambda=${LAMBDA}"
    
    # Запуск обучения
    python train.py \
        --problem "${PROBLEM}" \
        --nb_nodes "${NODES}" \
        --nb_epochs "${EPOCHS}" \
        --beta "${BETA}" \
        --lambda_gap "${LAMBDA}" \
        --gpu_id "${GPU_ID}" \
        --checkpoint_model "n" \
        --data_path "./" 
        
    echo "[+] Обучение конфигурации завершено."
    echo "--------------------------------------------------"
done