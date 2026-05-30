
from core.filter_butterworth import FilterButterworth


class TrainModelButterworth:
    
    def __init__(self, cutoff_freq: float, order: int):
        self.cutoff_freq = cutoff_freq
        self.order = order
    
    def train(self, signal: np.ndarray, sr: int) -> FilterButterworth:

        return FilterButterworth(order=self.order)