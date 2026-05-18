import numpy as np

class ImageWrap:
    def __init__(self, x_rev:bool=False, y_rev:bool=False, transpose:bool=False) -> None:
        self.x_rev = x_rev
        self.y_rev = y_rev
        self.transpose = transpose

    def deliver(self, np3d: np.ndarray) -> np.ndarray:
        np3d = np3d.copy()
        if len(np3d.shape) != 3 or np3d.shape[2] != 3:
            raise ValueError("np3d should be of length (H, W, 3)")
        if self.x_rev:
            np3d = np3d[::-1, :, :]
        if self.y_rev:
            np3d = np3d[:, ::-1, :]
        if self.transpose:
            np3d = np3d.transpose(1, 0, 2)
        return np3d
