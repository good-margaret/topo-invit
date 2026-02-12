!/bin/bash

PROBLEM="tsp"
TRAIN_SIZE=50
TEST_SIZES=(50 100 200) # Проверка Generalization (Zero-shot)
DISTS=("uniform" "clustered")

# Те же конфигурации, что и при обучении, чтобы найти нужные папки
CONFIGS=(
  "0.0 0.0"
  "0.0 0.05"
  "0.1 0.0"
  "0.1 0.05"
)

echo "=================================================="
echo "[*] ЭТАП 3: Финальное тестирование и сравнение"
echo "=================================================="

for CONFIG in "${CONFIGS[@]}"
do
    set -- $CONFIG
    BETA=$1
    LAMBDA=$2
    
    # Формируем путь к модели (нужно убедиться, что train.py сохраняет именно так)
    # Обычно это "checkpoint_timestamp.pkl". 
    # Для автоматизации лучше в train.py сохранять "latest.pkl" или искать последний файл.
    MODEL_DIR="ckpt/${PROBLEM}/train/model"
    
    # ХАК: берем последний измененный файл в папке (самый свежий чекпоинт)
    # В реальном эксперименте лучше указывать точное имя
    CKPT=$(ls -t $MODEL_DIR/*.pkl | head -1)
    
    echo "[*] Оценка модели: Beta=${BETA}, Lambda=${LAMBDA}"
    echo "[*] Чекпоинт: $CKPT"

    for SIZE in "${TEST_SIZES[@]}"
    do
        for DIST in "${DISTS[@]}"
        do
            DATA_PATH="data/data_farm/${PROBLEM}/${PROBLEM}${SIZE}/${PROBLEM}${SIZE}_${DIST}.txt"
            
            echo "   -> Тест на ${SIZE} узлах (${DIST})"
            
            # Запускаем test.py (предполагается, что он у вас есть и выводит Gap)
            python test.py \
                --problem "${PROBLEM}" \
                --nb_nodes "${SIZE}" \
                --checkpoint_model "${CKPT}" \
                --test_data "${DATA_PATH}" \
                --gpu_id 0 \
                --choice_deterministic "True"
        done
    done
    echo "--------------------------------------------------"
done