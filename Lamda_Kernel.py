import numpy as np
from scipy.spatial.distance import cdist
from random import sample
from tqdm import tqdm  # 用于显示进度条

class LamdaKernel:
    """
    Scalable implementation of Continuous Lambda Kernel (Isolation Kernel).
    
    优化说明:
    1. 利用 Voronoi 划分的稀疏性质，推导出解析公式替代原始的张量广播计算。
       时间复杂度从 O(N * psi^2) 降低到 O(N * psi)。
       空间复杂度大幅降低，消除了中间的大型三维矩阵。
    2. 引入 batch_size 处理，支持大规模数据的特征提取。
    
    Parameters
    ----------
    psi : int
        采样数 (Subsampling size)，即每个 Estimator 中的中心点数量。
    t : int
        集成大小 (Ensemble size)，即树的数量。
    eta : float
        衰减系数 (Decay parameter)。
    """
    def __init__(self, psi=16, t=100, eta=1.0):
        self.psi = psi
        self.t = t
        self.eta = eta
        self.centroids = []
        self.is_fitted = False

    def fit(self, X):
        """
        训练阶段：随机采样构建 t 组 Voronoi 中心点。
        """
        n_samples = X.shape[0]
        self.centroids = []
        
        # 简单检查 psi 是否大于样本数
        actual_psi = min(self.psi, n_samples)
        
        # 使用随机种子确保可复现性（可选）
        # np.random.seed(42) 
        
        for _ in range(self.t):
            # 随机采样 psi 个索引
            # 使用 numpy 的 choice 比 random.sample 在大数据下略快且易于管理
            sub_index = np.random.choice(n_samples, actual_psi, replace=False)
            self.centroids.append(X[sub_index, :])
            
        self.is_fitted = True
        return self

    def transform(self, X, batch_size=1024):
        """
        将数据转换为连续 Lambda 特征，支持大规模数据分批处理。
        
        Parameters
        ----------
        X : np.ndarray
            输入数据，形状 (n_samples, n_features)
        batch_size : int
            批处理大小，防止内存溢出。
            
        Returns
        -------
        final_features : np.ndarray
            形状 (n_samples, t * psi) 的稠密特征矩阵。
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling transform!")
            
        n_samples = X.shape[0]
        # 预分配最终的特征矩阵，避免频繁内存申请
        final_features = np.zeros((n_samples, self.t * self.psi), dtype=np.float32)
        
        # 分批处理数据
        for start_idx in tqdm(range(0, n_samples, batch_size), desc="Extracting Features"):
            end_idx = min(start_idx + batch_size, n_samples)
            X_batch = X[start_idx:end_idx]
            n_batch = X_batch.shape[0]
            
            # 对每一棵树（Estimator）进行计算
            for i in range(self.t):
                current_centroids = self.centroids[i] # (psi, d)
                
                # 1. 计算距离矩阵 (n_batch, psi)
                # metric='sqeuclidean' 通常比 'euclidean' 快，如果不想开方可以用 sqeuclidean 配合调整 eta
                # 这里为了保持和原论文一致，使用 euclidean
                dists = cdist(X_batch, current_centroids, metric='euclidean')
                
                # 2. 找到最近邻 (Voronoi 划分)
                nearest_idx = np.argmin(dists, axis=1) # (n_batch,)
                
                # 3. 获取最近邻距离 d
                # fancy indexing
                min_dists = dists[np.arange(n_batch), nearest_idx] # (n_batch,)
                
                # 4. 利用解析解直接计算特征值 (Optimization)
                # 原始逻辑：normalization_partition 计算 exp(-2*eta*(vi - vj)) 的归一化
                
                # 公式 A: Active Centroid (也就是最近的那个中心点) 的值
                # val = 1 / sqrt(1 + (psi-1) * exp(-2 * eta * d))
                # 避免溢出：如果 d 很大，exp(-2*eta*d) -> 0, val -> 1
                term_active = (self.psi - 1) * np.exp(-2 * self.eta * min_dists)
                val_active = 1.0 / np.sqrt(1.0 + term_active)
                
                # 公式 B: Inactive Centroids (其他 psi-1 个中心点) 的值
                # val = 1 / sqrt(exp(2 * eta * d) + psi - 1)
                # 避免溢出：如果 d 很大，exp(2*eta*d) 极大，val -> 0
                term_inactive = np.exp(2 * self.eta * min_dists) + (self.psi - 1)
                val_inactive = 1.0 / np.sqrt(term_inactive)
                
                # 5. 填充特征矩阵
                # 这一块在 final_features 中的列范围
                col_start = i * self.psi
                col_end = (i + 1) * self.psi
                
                # 先用 inactive 值填充整个块
                # feature_block shape: (n_batch, psi)
                # 利用广播机制填充
                feature_block = val_inactive[:, np.newaxis].repeat(self.psi, axis=1)
                
                # 再修正 active (最近邻) 的位置
                feature_block[np.arange(n_batch), nearest_idx] = val_active
                
                # 写入大矩阵
                final_features[start_idx:end_idx, col_start:col_end] = feature_block

        # 全局归一化 (对应 Lambda_feature.py 中的 / np.sqrt(t))
        final_features /= np.sqrt(self.t)
        
        return final_features

# ==========================================
# 性能与正确性测试代码
# ==========================================
if __name__ == "__main__":
    import time
    
    # 1. 生成大规模模拟数据 (例如 10万样本)
    N = 10000 
    D = 10
    print(f"Generating {N} samples with {D} dimensions...")
    X_train = np.random.rand(N, D)
    X_test = np.random.rand(1000, D)
    
    # 2. 初始化模型
    psi = 16
    t = 100
    eta = 0.5
    model = LamdaKernel(psi=psi, t=t, eta=eta)
    
    # 3. 拟合
    print("Fitting model...")
    start_time = time.time()
    model.fit(X_train)
    print(f"Fit time: {time.time() - start_time:.4f}s")
    
    # 4. 转换 (提速版)
    print(f"Transforming {N} samples (Scalable Implementation)...")
    start_time = time.time()
    features = model.transform(X_train, batch_size=2048)
    end_time = time.time()
    
    print(f"Transform time: {end_time - start_time:.4f}s")
    print(f"Output shape: {features.shape}")
    print(f"Features mean: {np.mean(features):.6f}")
    
    # 5. 验证数学一致性 (使用小样本对比)
    # 我们手动用原来的慢速公式算一个样本，看看结果是否一样
    print("\nVerifying mathematical correctness with a single sample...")
    
    # 取第一个 estimator 的中心点
    centroids = model.centroids[0]
    x_sample = X_train[0:1]
    
    # 手动计算
    dists = cdist(x_sample, centroids)
    nearest_idx = np.argmin(dists)
    d = dists[0, nearest_idx]
    
    # 原始论文逻辑模拟
    # 构造稀疏向量 [0, ..., d, ..., 0]
    sparse_vec = np.zeros(psi)
    sparse_vec[nearest_idx] = d
    
    # 原始 normalization_partition 逻辑
    a = sparse_vec.reshape(psi, 1) # (psi, 1)
    # M[i, j] = a[j] - a[i] (注意原代码维度变换逻辑，这里简化表达)
    # 原代码: M = a_exp - tmp_2. a_exp[i,j] = a[j], tmp_2[i,j] = a[i]
    # M[i, j] = val_j - val_i
    M = a.T - a # (1, psi) - (psi, 1) -> (psi, psi) via broadcasting
    # Sum over axis=0 (i)
    denom = np.sqrt(np.sum(np.exp(-2 * eta * M), axis=0))
    expected_block = (1.0 / denom).reshape(1, -1)
    
    # 我们的快速计算结果
    calculated_block = features[0, 0:psi] * np.sqrt(t) # 撤销最后的全局归一化以便对比
    
    print("Expected (First 5):  ", expected_block[0, :5])
    print("Calculated (First 5):", calculated_block[:5])
    
    diff = np.abs(expected_block - calculated_block)
    if np.max(diff) < 1e-5:
        print("\n✅ Verification PASSED! The optimization is mathematically equivalent.")
    else:
        print("\n❌ Verification FAILED!")