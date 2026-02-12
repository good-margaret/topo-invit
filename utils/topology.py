
import torch
import torch.nn as nn

def batch_mst_weight(dist_matrix):
    """
    Вычисляет суммарный вес MST для батча матриц расстояний.
    Сложность: O(B * N^2), где B - размер батча.
    Реализация алгоритма Прима, оптимизированная для PyTorch.
    """
    B, N, _ = dist_matrix.shape
    device = dist_matrix.device
    
    # Инициализация
    min_dists = dist_matrix[:, 0, :] # Дистанции от первого узла до всех остальных
    visited = torch.zeros((B, N), dtype=torch.bool, device=device)
    visited[:, 0] = True
    
    mst_weight = torch.zeros(B, device=device)
    
    for _ in range(N - 1):
        # Маскируем уже посещенные узлы
        masked_dists = min_dists.clone()
        masked_dists[visited] = float('inf')
        
        # Находим минимальное ребро для каждого графа в батче
        values, indices = torch.min(masked_dists, dim=1)
        
        mst_weight += values
        
        # Обновляем маску посещенных узлов
        # (используем scatter для батчевой обработки)
        visited.scatter_(1, indices.unsqueeze(1), True)
        
        # Обновляем минимальные дистанции до нового остова
        # Берем дистанции от только что добавленных узлов
        new_dists = dist_matrix.gather(1, indices.view(B, 1, 1).expand(B, 1, N)).squeeze(1)
        min_dists = torch.min(min_dists, new_dists)
        
    return mst_weight

def get_rtd_lite_loss(coords, embeddings):
    """
    Обновленная версия: Точная реализация Algorithm 2 из статьи.
    """
    # 1. Матрицы расстояний
    d_x = torch.cdist(coords, coords, p=2)
    d_z = torch.cdist(embeddings, embeddings, p=2)
    
    # 2. НОРМАЛИЗАЦИЯ ПО КВАНТИЛЮ (как в эталоне)
    # Это делает лосс стабильным независимо от масштаба координат
    # Вычисляем квантиль по всему батчу для скорости
    q_x = torch.quantile(d_x, 0.9) + 1e-6
    q_z = torch.quantile(d_z, 0.9) + 1e-6
    
    d_x_norm = d_x / q_x
    d_z_norm = d_z / q_z
    
    # 3. Совместная матрица
    d_joint = torch.minimum(d_x_norm, d_z_norm)
    
    # 4. MST веса (используем твой быстрый батчевый Prim)
    mst_x = batch_mst_weight(d_x_norm)
    mst_z = batch_mst_weight(d_z_norm)
    mst_joint = batch_mst_weight(d_joint)
    
    # 5. ФОРМУЛА С КОЭФФИЦИЕНТОМ 0.5
    # RTD(X, Z) = 0.5 * (Sum(MST_x) + Sum(MST_z) - 2 * Sum(MST_joint))
    rtd_loss = 0.5 * (mst_x + mst_z - 2 * mst_joint)
    
    return rtd_loss.mean()

def compute_tour_topological_gaps(tour, coords):
    """
    Вычисляет нормализованные Edge-wise Topological Divergence Gaps.
    Масштабирует расстояния, чтобы штраф был стабильным.
    """
    B, N, _ = coords.shape
    device = coords.device
    
    # 1. Матрица расстояний
    dist_matrix = torch.cdist(coords, coords)
    
    # 2. НОРМАЛИЗАЦИЯ (Scale Invariance)
    # Делим на 0.9-квантиль, чтобы привести расстояния к единому масштабу
    q = torch.quantile(dist_matrix, 0.9) + 1e-6
    dist_matrix_norm = dist_matrix / q
    
    # 3. Вычисляем Bottleneck Matrix (на основе нормализованных данных)
    # Она показывает максимально допустимое "здоровое" расстояние между узлами по MST
    bottleneck_matrix = compute_bottleneck_matrix(dist_matrix_norm)
    
    # 4. Подготовка индексов тура (замыкаем цикл)
    tour_extended = torch.cat([tour, tour[:, :1]], dim=1)
    u = tour_extended[:, :-1] # Откуда
    v = tour_extended[:, 1:]  # Куда
    
    # 5. Извлекаем нормализованные веса ребер тура
    # Собираем значения из dist_matrix_norm по индексам (u, v)
    edge_weights = dist_matrix_norm.gather(2, v.unsqueeze(-1)).gather(1, u.unsqueeze(-1)).squeeze(-1)
    
    # 6. Извлекаем bottleneck веса для тех же пар
    b_weights = bottleneck_matrix.gather(2, v.unsqueeze(-1)).gather(1, u.unsqueeze(-1)).squeeze(-1)
    
    # 7. Расчет Gap (Разрыва)
    # Если ребро тура длиннее, чем путь в MST, это топологический разрыв
    gaps = torch.clamp(edge_weights - b_weights, min=0.0)
    
    # Возвращаем сумму разрывов для каждого графа в батче
    return gaps.sum(dim=1)

def compute_bottleneck_matrix(dist_matrix):
    """
    Оптимизированное вычисление матрицы узких мест (bottleneck matrix).
    Сложность: O(B * N^2).
    """
    B, N, _ = dist_matrix.shape
    device = dist_matrix.device
    b_matrix = torch.zeros((B, N, N), device=device)

    # 1. Получаем ребра MST через алгоритм Прима (O(N^2))
    # Возвращает индексы ребер [B, N-1, 2] и их веса [B, N-1]
    mst_edges, mst_weights = _batch_prim_mst(dist_matrix)

    # 2. Сортируем только ребра MST (N-1 ребер вместо N^2)
    sorted_idx = torch.argsort(mst_weights, dim=1)

    # 3. Заполняем матрицу (Merge-based update)
    # К сожалению, DSU сложно векторизовать по батчу, если порядок ребер разный.
    # Но цикл по B и N итераций с векторными масками внутри — это очень быстро.
    for b in range(B):
        # comp_ids хранит ID компонента для каждого узла
        comp_ids = torch.arange(N, device=device)
        # Матрица принадлежности для быстрой фильтрации (N x N)
        # Используем список списков или маски. Маски в PyTorch эффективнее.
        
        batch_mst_e = mst_edges[b]
        batch_mst_w = mst_weights[b]
        curr_sorted_idx = sorted_idx[b]

        for idx in curr_sorted_idx:
            u, v = batch_mst_e[idx]
            w = batch_mst_w[idx]
            
            id_u, id_v = comp_ids[u], comp_ids[v]
            if id_u == id_v: continue
            
            mask_u = (comp_ids == id_u)
            mask_v = (comp_ids == id_v)

            # Ключевой момент: все пары между компонентом U и компонентом V
            # получают текущий вес w (т.к. это кратчайшее связующее ребро)
            # Используем внешнее произведение масок для заполнения блока
            b_matrix[b].masked_fill_(mask_u.unsqueeze(1) & mask_v.unsqueeze(0), w)
            b_matrix[b].masked_fill_(mask_v.unsqueeze(1) & mask_u.unsqueeze(0), w)

            # Объединяем компоненты
            comp_ids[mask_v] = id_u

    return b_matrix

def _batch_prim_mst(dist_matrix):
    """Векторизованный алгоритм Прима для поиска MST в батче."""
    B, N, _ = dist_matrix.shape
    device = dist_matrix.device
    
    adj = dist_matrix.clone()
    # Чтобы не выбирать петли (i, i)
    adj.diagonal(dim1=-2, dim2=-1).fill_(float('inf'))
    
    min_dists = adj[:, 0, :]
    parents = torch.zeros((B, N), dtype=torch.long, device=device)
    visited = torch.zeros((B, N), dtype=torch.bool, device=device)
    visited[:, 0] = True
    
    mst_edges = torch.zeros((B, N - 1, 2), dtype=torch.long, device=device)
    mst_weights = torch.zeros((B, N - 1), device=device)
    
    for i in range(N - 1):
        # Выбираем минимальное ребро до не посещенных узлов
        masked_dists = min_dists.clone()
        masked_dists[visited] = float('inf')
        
        weights, next_nodes = torch.min(masked_dists, dim=1)
        
        # Записываем ребро
        mst_edges[:, i, 0] = parents.gather(1, next_nodes.unsqueeze(1)).squeeze()
        mst_edges[:, i, 1] = next_nodes
        mst_weights[:, i] = weights
        
        visited.scatter_(1, next_nodes.unsqueeze(1), True)
        
        # Обновляем дистанции: min(старые, дистанции от нового узла)
        new_dists = adj.gather(1, next_nodes.view(B, 1, 1).expand(B, 1, N)).squeeze(1)
        
        # Если новая дистанция меньше, обновляем min_dist и родителя
        update_mask = new_dists < min_dists
        min_dists[update_mask] = new_dists[update_mask]
        parents[update_mask] = next_nodes.view(B, 1).expand(B, N)[update_mask]
        
    return mst_edges, mst_weights