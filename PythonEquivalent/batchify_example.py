import numpy as np


class LinearTransformation:
    def __init__(self, in_dim: int, out_dim: int):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.W = np.random.rand(self.in_dim, self.out_dim)  # [in_dim, out_dim]
        self.b = np.random.rand(self.out_dim)       # [out_dim]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Forward function of linear transformation.
        @param: x: Input vector, expected shape: [in_dim]
        """
        # [in_dim] -(expand_dims)-> [1, in_dim]
        # [1, in_dim] @ [in_dim, out_dim] + [out_dim ] -> [1, out_dim] -(squeeze)-> [out_dim]
        return np.squeeze(np.expand_dims(x, 0) @ self.W + self.b, 0)


class BatchedLinearTransformation:
    def __init__(self, in_dim: int, out_dim: int):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.W = np.random.rand(self.in_dim, self.out_dim)  # [in_dim, out_dim]
        self.b = np.random.rand(self.out_dim)       # [out_dim]

    def __call__(self, x: np.ndarray):
        """
        Forward function of batched linear transformation.
        :param x: Input vector, expected shape: [batch_size, in_dim]
        """

        # [batch_size, in_dim] @ [in_dim, out_dim] + [out_dim] -> [batch_size, out_dim]
        return x @ self.W + self.b