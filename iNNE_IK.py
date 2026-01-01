import numpy as np
from random import sample
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix

class iNNE_IK:
    """
    Isolation Kernel (Nearest Neighbor based implementation).
    
    Parameters
    ----------
    psi : int
        Subsampling size (number of centroids per estimator).
    t : int
        Number of estimators (ensemble size).
    """
    def __init__(self, psi, t):
        self.psi = psi
        self.t = t
        # 初始化实例变量，避免类变量导致的数据污染
        self.centroids_data = []    # 存储具体的中心点坐标
        self.centroids_radius = []  # 存储每个中心点的半径
        self.is_fitted = False

    def fit(self, data):
        """
        Fit the model by selecting centroids and calculating their adaptive radii.
        """
        self.centroids_data = []
        self.centroids_radius = []
        
        sn = data.shape[0] # 样本总数
        
        for i in range(self.t):
            # 1. 随机采样 psi 个索引
            subIndex = sample(range(sn), self.psi)
            
            # 2. 获取中心点数据 (tdata)
            # 优化：直接存储中心点坐标，而不是索引，这样 transform 时不需要原始 data
            tdata = data[subIndex, :]
            self.centroids_data.append(tdata)
            
            # 3. 计算中心点内部距离 (tt_dis)
            tt_dis = cdist(tdata, tdata)
            
            # 4. 计算每个中心点的半径 (radius)
            # 半径定义为：到最近邻中心点的距离
            radius = []
            # 添加大数值到对角线，避免自己和自己距离为0被选中为最小值
            np.fill_diagonal(tt_dis, np.inf)
            
            # axis=1 求每一行的最小值，即离该点最近的那个点的距离
            radius = np.min(tt_dis, axis=1)
            
            self.centroids_radius.append(radius)
            
        self.is_fitted = True
        return self

    def transform(self, newdata):
        """
        Transform new data into the kernel space using the fitted centroids.
        """
        if not self.is_fitted:
            raise RuntimeError("Invoke fit() first!")
            
        n, d = newdata.shape
        IDX = [] # Column indices for sparse matrix
        V = []   # Values for sparse matrix
        
        # 预先生成行索引，因为每一轮 t 对所有 n 个样本都是一样的逻辑
        # 但为了构建稀疏矩阵 (V, (row, col))，我们需要扁平化的索引
        # 这里的处理方式稍微不同，我们按 block 收集
        
        all_row_indices = []
        all_col_indices = []
        all_values = []

        for i in range(self.t):
            # 获取当前 estimator 的中心点和半径
            tdata = self.centroids_data[i]
            radius = self.centroids_radius[i]
            
            # 1. 计算新数据与中心点的距离
            dis = cdist(tdata, newdata) # shape: (psi, n) 注意 cdist 默认是 (XA, XB)
            
            # 2. 找到每个样本距离最近的中心点索引
            # axis=0 表示沿着 psi 维度找最小，返回 shape (n,)
            centerIdx = np.argmin(dis, axis=0) 
            
            # 3. 向量化计算 Value
            # 获取每个样本对应的最近中心点的距离
            # dis[centerIdx, range(n)] 这种写法在 numpy 中不直接支持多维花式索引
            # 正确写法是：dis[centerIdx, np.arange(n)]
            min_dists = dis[centerIdx, np.arange(n)]
            
            # 获取对应的半径阈值
            thresholds = radius[centerIdx]
            
            # 判断是否在半径内 (0 或 1)
            v_batch = (min_dists <= thresholds).astype(int)
            
            # 4. 收集数据用于构建稀疏矩阵
            # 只有值为 1 的地方才需要存储（虽然 logic 上 0 也是结果，但稀疏矩阵不存 0）
            # 原代码逻辑似乎保留了 0 值？
            # 重新阅读原代码：V.append(...) 存入了 0 或 1。
            # 为了保持原代码行为，我们将 0 和 1 都存入，或者利用 CSR 的特性
            
            # 列索引 = 当前中心点在当前树内的索引 + 树的偏移量
            col_idx_batch = centerIdx + i * self.psi
            
            all_row_indices.append(np.arange(n))
            all_col_indices.append(col_idx_batch)
            all_values.append(v_batch)

        # 拼接所有批次的数据
        IDR = np.concatenate(all_row_indices) # Row index
        IDX = np.concatenate(all_col_indices) # Column index
        V = np.concatenate(all_values)        # Values
        
        # 构建稀疏矩阵
        # shape = (样本数, 树数量 * 每棵树的中心点数)
        ndata = csr_matrix((V, (IDR, IDX)), shape=(n, self.t * self.psi))
        return ndata

    def fit_transform(self, data):
        """
        Fit to data, then transform it.
        """
        self.fit(data)
        return self.transform(data)


if __name__ == '__main__':
    import time
    from sklearn.datasets import make_blobs
    from sklearn.metrics.pairwise import cosine_similarity

    # 1. 生成模拟数据
    # 生成 1000 个样本，20 个特征，3 个聚类中心
    X, y = make_blobs(n_samples=1000, n_features=20, centers=3, random_state=42)

    print(f"原始数据形状: {X.shape}")

    # 2. 初始化模型
    psi = 16  # 每个 estimator 采样 16 个点
    t = 100   # 100 个 estimator
    inn_ik = iNNE_IK(psi=psi, t=t)

    # 3. 运行 fit_transform 并计时
    start_time = time.time()
    X_new = inn_ik.fit_transform(X)
    end_time = time.time()

    print(f"转换耗时: {end_time - start_time:.4f} 秒")

    # 4. 验证输出
    print(f"转换后特征形状 (Sparse): {X_new.shape}")
    # 预期形状: (1000, 100 * 16) = (1000, 1600)

    # 检查稀疏性
    sparsity = 1.0 - (X_new.count_nonzero() / float(X_new.shape[0] * X_new.shape[1]))
    print(f"稀疏度: {sparsity:.4f}")

    # 5. 简单应用：计算转换后数据的相似度
    # 在 IK 空间中，属于同一簇的点应该有更高的相似度
    sim_matrix = cosine_similarity(X_new)
    print(f"相似度矩阵形状: {sim_matrix.shape}")
    print("前5个样本的相似度示例:")
    print(sim_matrix[:5, :5])

    # 6. 单独测试 transform 方法
    X_test, _ = make_blobs(n_samples=5, n_features=20, centers=3, random_state=99)
    X_test_trans = inn_ik.transform(X_test)
    print(f"新数据转换形状: {X_test_trans.shape}")