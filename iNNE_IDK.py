import numpy as np
from iNNE_IK import iNNE_IK  # 假设你保存了之前的类为 iNNE_IK.py


def compute_distribution_embeddings(bag_list, psi, t=100):
    """
    Level 1: 将分布（点集）列表映射为核均值嵌入矩阵 (Kernel Mean Embeddings)。
    
    对应原函数: idk_kernel_map
    
    Parameters
    ----------
    bag_list : list of np.ndarray
        输入数据，每个元素是一个数据包（Distribution），形状为 (n_samples, n_features)。
    psi : int
        Level 1 采样大小。
    t : int
        Level 1 树的数量。
        
    Returns
    -------
    embedding_matrix : np.ndarray
        形状为 (n_bags, t * psi)，每一行代表一个分布的特征向量。
    """
    n_bags = len(bag_list)
    
    # 1. 数据展平 (Flattening)
    # 记录每个包的大小，以便后续切分
    bag_sizes = [len(bag) for bag in bag_list]
    # 使用 vstack 快速合并所有数据
    alldata = np.vstack(bag_list)
    
    # 计算切分索引点 [0, size1, size1+size2, ...]
    split_indices = np.concatenate(([0], np.cumsum(bag_sizes)))
    
    # 2. Level 1 IK 转换
    # 这里将所有点视为混合在一起的大池子进行特征提取
    inne_ik = iNNE_IK(psi, t)
    # fit_transform 返回的是 sparse matrix，转为 dense 方便后续求平均
    # 注意：如果数据量极大，这里转 dense 可能会爆内存，但在 Distribution 层面通常还好
    all_point_features = inne_ik.fit_transform(alldata).toarray()
    
    # 3. 计算均值嵌入 (Mean Embeddings)
    # 对每个包内的点的特征向量求平均
    embedding_list = []
    for i in range(n_bags):
        start_idx = split_indices[i]
        end_idx = split_indices[i+1]
        
        # 提取当前包的所有点特征
        bag_features = all_point_features[start_idx : end_idx]
        
        # 求平均，得到该分布的代表向量
        # axis=0 表示沿着点的维度压缩
        mean_embedding = np.sum(bag_features, axis=0) / bag_sizes[i]
        embedding_list.append(mean_embedding)
        
    return np.array(embedding_list)

def detect_distribution_anomalies(bag_list, psi1, psi2, t1=100, t2=100):
    """
    Level 2: IDK^2 (IDK Square). 检测分布层面的异常。
    
    对应原函数: idk_square
    """
    # Step 1: 获取每个分布的 Embedding (Level 1)
    # 输入: list of [N_points x D], 输出: [N_distributions x (t1*psi1)]
    dist_embeddings = compute_distribution_embeddings(bag_list, psi1, t1)
    
    # Step 2: 在 Embedding 空间再次应用 IK (Level 2)
    # 将每个分布看作一个样本
    inne_ik_l2 = iNNE_IK(psi2, t2)
    
    # 获取 Level 2 特征 (0/1 矩阵)
    l2_features = inne_ik_l2.fit_transform(dist_embeddings).toarray()
    
    # Step 3: 计算异常分数
    # 计算 Level 2 空间的中心点 (Centroid of distributions)
    # 这里的 mean 代表了"正常分布"应该长什么样
    centroid = np.mean(l2_features, axis=0) 
    
    # 计算每个分布与中心的相似度
    # 相似度 = Dot Product (特征, 中心)
    # 我们通常希望归一化到 [0, 1] 区间。
    # IK 特征中，每一行大约有 t2 个 1。
    # 所以 dot product 的最大值接近 t2 (如果完全重合)。
    # 原代码中除以了 t1，这可能是笔误，逻辑上除以当前层的 t2 才是归一化。
    
    similarity_scores = np.dot(l2_features, centroid.T) / t2
    
    # 注意：IK 中，相似度越高越正常，相似度越低越异常。
    return similarity_scores

def point_anomaly_detector(data, psi, t=100):
    """
    标准的点级异常检测 (Point Anomaly Detection)
    
    对应原函数: idk_anomalyDetector
    """
    inne_ik = iNNE_IK(psi, t)
    features = inne_ik.fit_transform(data).toarray()
    
    # 计算全局中心
    centroid = np.mean(features, axis=0)
    
    # 计算每个点到中心的相似度
    # 归一化：除以 t
    scores = np.dot(features, centroid.T) / t
    
    return scores


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score
    def test_point_anomaly():
        # ===========================
        # 1. 生成模拟数据
        # ===========================
        np.random.seed(42)
        n_normal = 300
        n_outliers = 20

        # 生成正常数据 (聚集在中心)
        X_normal = np.random.normal(loc=0.0, scale=0.5, size=(n_normal, 2))

        # 生成异常数据 (均匀分布在四周的噪声)
        X_outliers = np.random.uniform(low=-4, high=4, size=(n_outliers, 2))

        # 过滤掉那些随机生成却意外落在中心区域的点，保证异常点真的很"异常"
        mask = np.linalg.norm(X_outliers, axis=1) > 2
        X_outliers = X_outliers[mask]
        n_outliers = len(X_outliers) # 更新实际异常点数量

        # 合并数据
        X = np.concatenate([X_normal, X_outliers])

        # 创建标签: 0=正常, 1=异常
        y_true = np.concatenate([np.zeros(n_normal), np.ones(n_outliers)])

        print(f"数据生成完毕: 正常样本 {n_normal} 个, 异常样本 {n_outliers} 个")

        # ===========================
        # 2. 运行异常检测
        # ===========================
        psi = 32   # 采样大小
        t = 100    # 树的数量

        scores = point_anomaly_detector(X, psi, t)
        anomaly_scores = 1 - scores

        # ===========================
        # 3. 评估结果 (AUC)
        # ===========================
        auc = roc_auc_score(y_true, anomaly_scores)
        print(f"ROC-AUC Score: {auc:.4f} (1.0 表示完美区分)")

        # ===========================
        # 4. 可视化
        # ===========================
        plt.figure(figsize=(12, 5))

        # 子图1: 原始数据标签
        plt.subplot(1, 2, 1)
        plt.title("Ground Truth (Blue=Normal, Red=Anomaly)")
        plt.scatter(X_normal[:, 0], X_normal[:, 1], c='blue', s=20, label='Normal')
        plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red', s=40, marker='x', label='Anomaly')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 子图2: 算法预测的热力图
        plt.subplot(1, 2, 2)
        plt.title(f"IK Anomaly Score (AUC={auc:.2f})\nDarker Red = More Anomalous")
        sc = plt.scatter(X[:, 0], X[:, 1], c=scores, cmap='Reds', s=30, edgecolor='k', linewidth=0.5)
        plt.colorbar(sc, label='Anomaly Score')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
    test_point_anomaly()