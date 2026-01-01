import torch


class LamdaKernel:
    """
    Extracts distribution-based features by sampling points and computing
    normalized exponential distances.
    """

    def __init__(self, psi, t, eta=1.0):
        """
        Initialize the feature extractor.

        Args:
            psi (int): Number of points to sample each time.
            t (int): Number of times to repeat the sampling.
            eta (float): Parameter for exponential decay. Defaults to 1.0.
        """
        self.psi = psi
        self.t = t
        self.eta = eta
        self.sampled_indices = None

    def fit(self, X_train):
        """
        Sample points from the training input data and store their indices.
        Note: This method now takes X_train explicitly.

        Args:
            X_train (torch.Tensor): Training input data matrix of shape (n_train, d).
        """
        n_train = X_train.shape[0]
        if self.psi > n_train:
             raise ValueError(f"psi ({self.psi}) cannot be larger than the number of training samples ({n_train}).")
        # Store all sampled indices for each iteration
        self.sampled_indices = []
        for _ in range(self.t):
            indices = torch.randperm(n_train)[:self.psi]
            self.sampled_indices.append(indices)
        return self

    def transform(self, X, X_train):
        """
        Transform the input data using the sampled points from X_train.

        Args:
            X (torch.Tensor): Input data matrix of shape (n, d) to transform.
            X_train (torch.Tensor): Training data matrix of shape (n_train, d)
                                    used for the sampling reference.

        Returns:
            torch.Tensor: Feature matrix of shape (n, psi*t).
        """
        if self.sampled_indices is None:
            raise ValueError("Model has not been fitted yet. Call fit(X_train) first.")

        device = X.device
        n = X.shape[0]
        all_features = []

        for indices in self.sampled_indices:
            # Sample points from the *training* data
            sampled_points = X_train[indices] # Shape: (psi, d)

            # Compute distances between all points in X and sampled points
            # X.unsqueeze(1): (n, 1, d)
            # sampled_points.unsqueeze(0): (1, psi, d)
            diff = X.unsqueeze(1) - sampled_points.unsqueeze(0)  # shape: (n, psi, d)
            distances = torch.norm(diff, dim=2)  # shape: (n, psi)

            # Compute exponential term with stability epsilon
            exp_term = torch.exp(-self.eta * distances) / (distances + 1e-10)

            # Normalize features for each sample point by the sum across sampled points
            # Add epsilon to sum for stability
            sum_exp = exp_term.sum(dim=1, keepdim=True) + 1e-10 # Shape: (n, 1)
            normalized_exp_term = exp_term / sum_exp # Shape: (n, psi)

            all_features.append(normalized_exp_term)

        # Concatenate all features from t iterations
        final_features = torch.cat(all_features, dim=1)  # shape: (n, psi*t)

        return final_features

    def fit_transform(self, X):
        """
        Fit the model using X as training data and transform X.

        Args:
            X (torch.Tensor): Input data matrix of shape (n, d).

        Returns:
            torch.Tensor: Feature matrix of shape (n, psi*t).
        """
        return self.fit(X).transform(X, X) # Use X for both fitting and transforming


if __name__ == '__main__':
    import unittest
    import torch
    import numpy as np

    class TestDistributionFeatureExtractor(unittest.TestCase):

        def setUp(self):
            # 固定随机种子，保证测试结果可复现
            torch.manual_seed(42)

            # 构造一些模拟数据
            self.n_samples = 50   # 样本数量
            self.d_features = 10  # 原始特征维度
            self.psi = 5          # 每次采样的点数
            self.t = 3            # 迭代次数
            self.eta = 1.0        # 衰减系数

            self.X_train = torch.randn(self.n_samples, self.d_features)
            self.extractor = LamdaKernel(psi=self.psi, t=self.t, eta=self.eta)

        def test_output_shape(self):
            """测试 1: 输出形状是否正确 (N, psi * t)"""
            print("\n[测试 1] 验证输出特征矩阵的形状...")

            features = self.extractor.fit_transform(self.X_train)

            expected_shape = (self.n_samples, self.psi * self.t)
            self.assertEqual(features.shape, expected_shape, 
                             f"形状错误: 期望 {expected_shape}, 实际得到 {features.shape}")
            print(" -> 形状验证通过。")

        def test_normalization_property(self):
            """测试 2: 验证每一组 psi 特征的和是否为 1 (归一化逻辑)"""
            print("\n[测试 2] 验证分组归一化性质 (Sum = 1)...")

            features = self.extractor.fit_transform(self.X_train)

            # 特征矩阵的列数是 psi * t。
            # 逻辑上，每一次迭代 t 生成的 psi 个特征，其和应该为 1。
            # 我们检查第一个样本 (row 0) 的第一组特征 (前 psi 列)

            first_group_sum = features[0, :self.psi].sum().item()

            # 使用 assertAlmostEqual 处理浮点数精度问题
            self.assertAlmostEqual(first_group_sum, 1.0, places=5, 
                                   msg=f"归一化失败: 第一组特征之和应为 1.0, 实际为 {first_group_sum}")

            # 检查第二组特征
            second_group_sum = features[0, self.psi : 2*self.psi].sum().item()
            self.assertAlmostEqual(second_group_sum, 1.0, places=5)

            print(" -> 归一化逻辑验证通过。")

        def test_transform_on_new_data(self):
            """测试 3: 模拟在新数据(测试集)上的 Transform"""
            print("\n[测试 3] 验证在新数据上的 Transform...")

            # 拟合训练集
            self.extractor.fit(self.X_train)

            # 创建新的测试数据 (例如 5 个样本)
            X_test = torch.randn(5, self.d_features)

            # 注意: 根据你的代码逻辑，transform 时必须再次传入 X_train 作为参照
            features_test = self.extractor.transform(X_test, self.X_train)

            expected_test_shape = (5, self.psi * self.t)
            self.assertEqual(features_test.shape, expected_test_shape)
            print(" -> 测试集转换验证通过。")

        def test_psi_size_error(self):
            """测试 4: 当 psi > 样本数时，是否抛出 ValueError"""
            print("\n[测试 4] 验证非法参数捕获...")

            # 设置 psi 比样本数大
            large_psi = self.n_samples + 10
            bad_extractor = LamdaKernel(psi=large_psi, t=1)

            with self.assertRaises(ValueError):
                bad_extractor.fit(self.X_train)

            print(" -> 异常捕获验证通过。")

        def test_transform_without_fit(self):
            """测试 5: 未调用 fit 直接调用 transform 是否报错"""
            print("\n[测试 5] 验证未拟合直接转换的错误处理...")

            new_extractor = LamdaKernel(psi=5, t=1)
            with self.assertRaises(ValueError):
                new_extractor.transform(self.X_train, self.X_train)

            print(" -> 流程控制验证通过。")

    unittest.main(argv=['first-arg-is-ignored'], exit=False)