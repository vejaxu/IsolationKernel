import torch
import numpy as np
import warnings


class IK_iNNE_GPU():
    def __init__(self, t, psi):
        if not torch.cuda.is_available():
            raise Exception("CUDA is not available")
        else:
            self.device = torch.device("cuda")
        self._t = t
        self._psi = psi
        self._center_index_list = None
        self._radius_list = None
        self.X: np.ndarray = None

    def fit(self, X: np.ndarray) -> None:
        self.X = X

        if self._psi > X.shape[0]:
            self._psi = X.shape[0]
            warnings.warn(f"psi is set to {X.shape[0]} as it is greater than the number of data points.")
        
        self._center_index_list = np.array([np.random.permutation(X.shape[0])[: self._psi] for _ in range(self._t)]) # shape = (t, psi)
        self._center_list = np.zeros((self._psi * self._t), X.shape[1])
        self._radius_list = torch.zeros((self._t, self._psi), dtype=torch.float32, device=self.device)

        for i in range(self._t):
            sample = self.X[self._center_index_list[i]]
            self._center_list[i * self._psi: (i + 1) * self._psi] = sample
            s2s = torch.cdist(torch.tensor(sample, dtype=torch.float32, device=self.device), torch.tensor(sample, dtype=torch.float32, device=self.device), p=2)
            s2s.fill_diagonal_(float('inf'))
            self._radius_list = torch.min(s2s, dim=0).values

        return 