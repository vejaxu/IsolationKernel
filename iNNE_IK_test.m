% Clear workspace
clc; clear; close all;

%% 1. 数据生成 (Data Generation)
% 设置随机种子以保证结果可复现
rng(42);

% --- 生成正常数据 (Normal Data) ---
% 生成 300 个点，聚集在 (0,0) 附近，标准差 0.5
n_normal = 300;
X_normal = randn(n_normal, 2) * 0.5;

% --- 生成异常数据 (Outliers) ---
% 生成 20 个点，均匀分布在 [-4, 4] 的区域
n_outlier = 20;
X_outlier = (rand(n_outlier, 2) - 0.5) * 8;

% 过滤掉意外落在中心的点，确保它们离得够远
dist_from_center = sqrt(sum(X_outlier.^2, 2));
X_outlier(dist_from_center < 2, :) = []; % 删除靠近中心的
n_outlier = size(X_outlier, 1);

% 合并数据
X = [X_normal; X_outlier];
labels = [zeros(n_normal, 1); ones(n_outlier, 1)]; % 0:正常, 1:异常

fprintf('数据生成完毕: 正常样本 %d 个, 异常样本 %d 个\n', n_normal, n_outlier);

%% 2. 运行 iNNE-IK 模型 (Run Model)
psi = 16;   % 采样大小
t = 100;    % 树的数量 (Ensemble size)

fprintf('正在运行 iNNE-IK 特征提取 (psi=%d, t=%d)...\n', psi, t);
tic;
% 注意：这里用 X 当作训练集，也用 X 当作测试集
feature_map = iNNE_IK(X, X, psi, t); 
time_taken = toc;
fprintf('特征提取完成，耗时 %.4f 秒\n', time_taken);

% 检查输出维度
[n_samples, n_feats] = size(feature_map);
fprintf('输出特征矩阵维度: [%d x %d]\n', n_samples, n_feats);

%% 3. 计算异常分数 (Compute Anomaly Score)
% 逻辑：计算每个点与"全局平均特征"的相似度
% IK 特征是稀疏的 0/1 向量。
% Mean Map (Center) = sum(feature_map) / n_samples
% Similarity = feature_map * Mean_Map' 

% 转换为全矩阵计算均值 (如果数据量巨大，需用稀疏矩阵算法，这里数据小直接 full)
F_full = full(feature_map); 
Center = mean(F_full, 1); 

% 计算相似度 (Similarity Score)
% 归一化：因为做了 t 次采样，最大重叠次数是 t
similarity_scores = (F_full * Center') / t * psi; 
% 注意：这里的归一化系数取决于具体定义，通常 /t 即可将值缩放到 [0,1] 附近
% 更简单的 IK 异常分数定义：
similarity_scores = F_full * mean(F_full, 1)'; 

% 转换为异常分数 (Anomaly Score): 越不相似越异常
anomaly_scores = 1 - (similarity_scores / max(similarity_scores));

%% 4. 评估与可视化 (Evaluation & Visualization)

% --- 计算 AUC (Area Under Curve) ---
% MATLAB 自带 perfcurve 函数计算 ROC
[X_roc, Y_roc, ~, AUC] = perfcurve(labels, anomaly_scores, 1);
fprintf('检测准确度 (AUC): %.4f\n', AUC);

% --- 绘图 ---
figure('Position', [100, 100, 1000, 400]);

% 子图 1: 真实标签
subplot(1, 2, 1);
gscatter(X(:,1), X(:,2), labels, 'br', 'xo');
title('Ground Truth (Blue=Normal, Red=Anomaly)');
legend('Normal', 'Anomaly', 'Location', 'best');
grid on;
axis equal;

% 子图 2: 预测热力图
subplot(1, 2, 2);
scatter(X(:,1), X(:,2), 30, anomaly_scores, 'filled');
colorbar;
colormap(jet); % 使用 jet 颜色映射 (蓝=低分/正常, 红=高分/异常)
title(['Anomaly Score Heatmap (AUC = ' num2str(AUC, '%.2f') ')']);
xlabel('X1'); ylabel('X2');
grid on;
axis equal;

sgtitle('iNNE-IK Anomaly Detection Test');