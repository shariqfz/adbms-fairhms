import numpy as np
from sklearn.neighbors import KDTree

class ANN:
    def __init__(self):
        self.tree = None
        self.data_array = None
        self.data_points = []
        self.data_size = 0
        self.dim = 0
        self.bucket_size = 1

    def clean(self):
        self.tree = None
        self.data_array = None
        self.data_points = []
        self.data_size = 0
        self.dim = 0
        self.bucket_size = 1

    def insert_pts(self, dataP):
        self.clean()
        self.data_points = dataP
        self.data_size = len(dataP)
        if self.data_size == 0:
            return
        self.dim = self.data_points[0].get_dimension()
        self.data_array = np.array([[p.get_coordinate(d) for d in range(self.dim)] for p in self.data_points])
        self.bucket_size = 1 + int(np.ceil(self.data_size / 10.0))
        self.tree = KDTree(self.data_array, leaf_size=self.bucket_size, metric='euclidean')

    def get_anns(self, queryP, epsilon):
        if self.tree is None:
            raise RuntimeError("No points in ANN data structure")
        if not queryP:
            return []
        assert self.dim == queryP[0].get_dimension()
        query_array = np.array([[p.get_coordinate(d) for d in range(self.dim)] for p in queryP])
        dists, indices = self.tree.query(query_array, k=1)
        return indices.flatten().tolist()

    def get_ann_ks(self, queryP, epsilon, r):
        if self.tree is None:
            raise RuntimeError("No points in ANN data structure")
        if not queryP:
            return []
        query_array = np.array([[p.get_coordinate(d) for d in range(self.dim)] for p in queryP])
        dists, indices = self.tree.query(query_array, k=r)
        return indices.flatten().tolist()

    def get_k_anns(self, queryP, epsilon, r):
        if self.tree is None:
            raise RuntimeError("No points in ANN data structure")
        if not queryP:
            return [], []
        query_array = np.array([[p.get_coordinate(d) for d in range(self.dim)] for p in queryP])
        dists, indices = self.tree.query(query_array, k=r)
        return indices.tolist(), dists.tolist()

    def get_single_ann(self, query_point, epsilon):
        if self.tree is None:
            raise RuntimeError("No points in ANN data structure")
        query_array = np.array([[query_point.get_coordinate(d) for d in range(self.dim)]])
        dists, indices = self.tree.query(query_array, k=1)
        return indices[0][0], dists[0][0]

    def get_single_k_ann(self, query_point, epsilon, k):
        if self.tree is None:
            raise RuntimeError("No points in ANN data structure")
        query_array = np.array([[query_point.get_coordinate(d) for d in range(self.dim)]])
        dists, indices = self.tree.query(query_array, k=k)
        return indices[0][k-1], dists[0][k-1]

    def get_r_anns(self, Utils, query_point, epsilon, k):
        if self.tree is None:
            raise RuntimeError("No points in ANN data structure")
        query_id = query_point.getId()
        idxs = []
        for util_point in Utils:
            query_array = np.array([[util_point.get_coordinate(d) for d in range(self.dim)]])
            dists, indices = self.tree.query(query_array, k=k)
            for idx in indices[0]:
                if self.data_points[idx].getId() == query_id:
                    idxs.append(util_point.getId())
                    break
        return idxs