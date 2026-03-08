from dataset.movingmnist import MovingMNIST
from dataset.taxibj import Taxibj
from dataset.weather import WeatherBench
SETTINGS = [
    "movingmnist",
    "taxibj",
    "weather",
]

DATASETS = {
    "moving_mnist": MovingMNIST,
    "taxibj": Taxibj,
   "weather": WeatherBench,
}

METRIC_CHECKPOINT_INFO = {
    "moving_mnist": {
        "monitor": "MSE",
        "filename": "{epoch:02d}-{MSE:.4f}",
        "mode": "min",
    },
    "taxibj": {
        "monitor": "MSE",
        "filename": "{epoch:02d}-{MSE:.4f}",
        "mode": "min",
    },
    "weather": {
        "monitor": "MSE",
        "filename": "{epoch:02d}-{MSE:.4f}",
        "mode": "min",
    },
}


