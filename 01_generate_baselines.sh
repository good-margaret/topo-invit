!/bin/bash

# Настройки
PROBLEM="tsp" # или "cvrp"
RUNS=50       # Количество повторов LKH (для CVRP лучше 10-20, для TSP 50-100)
SIZES=(50 100 200)
DISTS=("uniform" "clustered")
NUM_SAMPLES=1000 # Размер тестового сета (стандарт 1000 или 10000)

echo "=================================================="
echo "[*] ЭТАП 1: Генерация данных и Baselines (LKH3)"
echo "=================================================="

for SIZE in "${SIZES[@]}"
do
    for DIST in "${DISTS[@]}"
    do
        echo "[>] Генерация: ${PROBLEM} | N=${SIZE} | ${DIST}"
        
        # 1. Генерируем датасет
        python generator/data_farm.py \
            --problem-type "${PROBLEM}" \
            --size "${SIZE}" \
            --distribution "${DIST}" \
            --num "${NUM_SAMPLES}"
        
        # Путь к сгенерированному файлу (должен совпадать с логикой data_farm.py)
        DATA_FILE="data/data_farm/${PROBLEM}/${PROBLEM}${SIZE}/${PROBLEM}${SIZE}_${DIST}.txt"
        
        # 2. Решаем через LKH3
        echo "[>] Решение LKH3 для: ${DATA_FILE}"
        python baselines/solve_by_lkh.py \
            --problem-type "${PROBLEM}" \
            --path "${DATA_FILE}" \
            --runs "${RUNS}" \
            --overwrite # Перезаписать, если уже есть старое
            
        echo "--------------------------------------------------"

        echo "[>] Решение Gurobi для: ${DATA_FILE}"
        python baselines/solve_by_Gurobi.py \
            --problem-type "${PROBLEM}" \
            --path "${DATA_FILE}" \
            --runs "${RUNS}" \
            --overwrite # Перезаписать, если уже есть старое
            
        echo "--------------------------------------------------"
    done
done

echo "[!] Базовые линии готовы. Можно приступать к обучению."