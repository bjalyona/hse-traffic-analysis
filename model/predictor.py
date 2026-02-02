from pathlib import Path

import joblib
import numpy as np


class SalaryPredictor:
    """Класс для предсказания зарплат с использованием обученной модели."""

    def __init__(self, models_dir: str = "resources") -> None:
        """Инициализация предсказателя с загрузкой модели."""
        models_dir_path = Path(models_dir)
        self.model = joblib.load(models_dir_path / "best_model.joblib")
        self.scaler = joblib.load(models_dir_path / "scaler.joblib")

    def predict_from_file(self, file_path: str) -> np.ndarray:
        """Предсказание зарплат на основе данных из файла."""
        X = np.load(file_path)
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)

        # Обеспечение неотрицательных предсказаний
        return np.maximum(predictions, 0).astype(float)