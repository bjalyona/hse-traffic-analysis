"""
Пакет для обучения и использования моделей регрессии
"""

from .trainer import ModelTrainer
from .predictor import SalaryPredictor

__version__ = "1.0.0"
__all__ = ["ModelTrainer", "SalaryPredictor"]