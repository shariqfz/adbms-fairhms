import numpy as np
from scipy.linalg import cholesky, lu, lu_solve
from typing import List
from point import Point  # Assuming Point class exists

class MVEUtil:
    @staticmethod
    def get_normalized_mve(dataP: List[Point], epsilon: float) -> tuple:
        """
        Compute Minimum Volume Enclosing Ellipsoid (MVE) and normalize points
        Returns: (normalized_points, outer_radius, inner_radius)
        """
        if not dataP:
            return [], 0.0, 0.0

        d = dataP[0].dim
        n = len(dataP)
        
        # Convert points to numpy array (d x n matrix)
        A = np.array([p.coord for p in dataP]).T

        # Run Khachiyan's algorithm
        Q, c = MVEUtil.khachiyan_algo(A, epsilon, max_iter=1000)
        
        # Calculate maximum (x-c)^T Q (x-c)
        max_val = max(np.dot((p.coord - c).T, np.dot(Q, (p.coord - c))) for p in dataP)
        outer_rad = np.sqrt(max_val + 0.01)

        # Cholesky decomposition
        try:
            L = cholesky(Q, lower=True)
        except np.linalg.LinAlgError:
            raise RuntimeError("Cholesky decomposition failed")

        # Compute transformation matrix and center
        L_tr = L.T
        transformation = np.linalg.inv(L_tr)
        center = c

        # Transform points
        normalizedP = [
            Point(d, coord=np.dot(L_tr, (p.coord - center)) )
            for p in dataP
        ]

        # Compute inner radius
        inner_rad = 1 / ((1 + epsilon) * d)

        return normalizedP, outer_rad, inner_rad

    @staticmethod
    def khachiyan_algo(A: np.ndarray, eps: float, max_iter: int) -> tuple:
        """Khachiyan's algorithm for MVE computation"""
        d, m = A.shape
        Ap = np.vstack([A, np.ones((1, m))])  # Lift matrix
        p = np.ones(m) / m  # Initial weights

        for _ in range(max_iter):
            # Compute Lambda(p) = sum(p_i * Ap[:,i] @ Ap[:,i].T)
            Lambda = np.sum(p[i] * np.outer(Ap[:,i], Ap[:,i]) for i in range(m))
            
            try:
                inv_Lambda = np.linalg.inv(Lambda)
            except np.linalg.LinAlgError:
                break

            # Find maximum diagonal element of Ap.T @ inv_Lambda @ Ap
            M = np.einsum('ij,jk,ki->i', Ap.T, inv_Lambda, Ap)
            max_idx = np.argmax(M)
            max_val = M[max_idx]

            if max_val - d - 1 < eps:
                break

            # Update weights
            step_size = (max_val - d - 1) / ((d + 1) * (max_val - 1))
            p = p * (1 - step_size)
            p[max_idx] += step_size

        # Compute final Q and center
        Q, c = MVEUtil.ka_invert_dual(A, p)
        return Q, c

    @staticmethod
    def ka_invert_dual(A: np.ndarray, p: np.ndarray) -> tuple:
        """Compute dual inversion for final ellipsoid parameters"""
        d, m = A.shape
        PN = A * p  # Equivalent to diag(p) @ A.T
        PN = A @ PN.T

        M2 = A @ p
        M3 = np.outer(M2, M2)

        try:
            inv_matrix = np.linalg.inv(PN - M3)
        except np.linalg.LinAlgError:
            inv_matrix = np.linalg.pinv(PN - M3)

        Q = inv_matrix / d
        c = A @ p
        return Q, c