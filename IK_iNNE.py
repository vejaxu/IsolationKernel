import numpy as np
from random import sample
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix


class IsolationKernel_INNE:
    data = None
    centroid = []

    def __init__(self, t, psi):
        self.t = t
        self.psi = psi

    def fit_transform(self, data):
        self.data = data
        self.centroid = []
        self.centroids_radius = []
        n, d = self.data.shape

        IDX = np.array([])

        boolValues = [] # 判断每次采样时，点是否在最近采样点的超球区域内，因此是布尔值

        for i in range(self.t):
            print(f"epoch {i}")
            sampled_points_indices = sample(range(n), self.psi)
            print("sample points indices")
            print(sampled_points_indices)
            self.centroid.append(sampled_points_indices)
            sampled_data = self.data[sampled_points_indices, :]

            sampled_sampled_distance = cdist(sampled_data, sampled_data)
            print("sampled_sampled_distance")
            print(sampled_sampled_distance)
            
            # to be optimized
            radius = []
            for index in range(self.psi):
                r = sampled_sampled_distance[index]
                r[r < 0] = 0
                r = np.delete(r, index)
                radius.append(np.min(r))
            print("sampled points radius")
            print(radius)
            self.centroids_radius.append(radius)
            
            sampled_data_distance = cdist(sampled_data, self.data)
            # print(f"sampled_data_distance shape: {sampled_data_distance.shape}")
            print("sampled_data_distance")
            print(sampled_data_distance)

            center_index = np.argmin(sampled_data_distance, axis=0) # 求解每个点的最短距离中心点
            print("center_index")
            print(center_index)

            for j in range(n):
                boolValues.append(int(sampled_data_distance[center_index[j], j] < radius[center_index[j]]))
            print("boolValues")
            print(boolValues)

            # 非零元素列索引
            IDX = np.concatenate((IDX, center_index + i * self.psi), axis=0) # 所有的潜在位置信息 加上类似位置编码的信息
            print("IDX")
            print(IDX)

            print("\n")
        IDR = np.tile(range(n), self.t) # 非0元素行索引
        print("IDR")
        print(IDR)
        ndata = csr_matrix((boolValues, (IDR, IDX)), shape=(n, self.t * self.psi))
        print(ndata)
        ik_feature_map = ndata.toarray()
        return ik_feature_map
    

    def fit(self, data):
        self.data = data
        self.centroid = []
        self.centroids_radius = []
        sn = self.data.shape[0]
        for i in range(self.t):
            subIndex = sample(range(sn), self.psi)
            self.centroid.append(subIndex)
            tdata = self.data[subIndex, :]
            tt_dis = cdist(tdata, tdata)
            radius = [] #restore centroids' radius
            for r_idx in range(self.psi):
                r = tt_dis[r_idx]
                r[r<0] = 0
                r = np.delete(r,r_idx)
                radius.append(np.min(r))
            self.centroids_radius.append(radius)


    def transform(self, newdata):
        assert self.centroid != None, "invoke fit() first!"
        n, d = newdata.shape
        IDX = np.array([])
        V = []
        for i in range(self.t):
            subIndex = self.centroid[i]
            radius = self.centroids_radius[i]
            tdata = self.data[subIndex, :]
            dis = cdist(tdata, newdata)
            centerIdx = np.argmin(dis, axis=0)
            for j in range(n):
                V.append(int(dis[centerIdx[j], j] <= radius[centerIdx[j]]))
            IDX = np.concatenate((IDX, centerIdx + i * self.psi), axis=0)
        IDR = np.tile(range(n), self.t)
        ndata = csr_matrix((V, (IDR, IDX)), shape=(n, self.t * self.psi))
        return ndata
    

if __name__ == '__main__':
    t = 5
    psi = 2
    data = np.array(([1, 2], 
                    [2, 3], 
                    [3, 4], 
                    [10, 10]))
    ik_inne = IsolationKernel_INNE(t, psi)
    ndata = ik_inne.fit_transform(data)
    print(ndata)