from typing import Tuple

import numpy as np

from .handlers import LoadHandler, FeatureExtractionHandler, PrepareHandler


class Pipeline:
    """Конвейер обработки данных."""

    def __init__(self) -> None:
        """Инициализация конвейера с цепочкой обработчиков."""
        load_handler = LoadHandler()
        extract_handler = FeatureExtractionHandler()
        prepare_handler = PrepareHandler()

        load_handler.set_next(extract_handler).set_next(prepare_handler)
        self._first_handler = load_handler
        self._context = {}

    def run(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Запуск конвейера обработки данных."""
        self._context = {"file_path": file_path}
        df = self._first_handler.handle(None, self._context)

        # Извлечение матрицы признаков и целевой переменной
        features = self._context["features"]
        target = self._context["target"]

        X = df[features].values.astype(np.float32)
        y = df[target].values.astype(np.float32)

        return X, y

    @staticmethod
    def save(X: np.ndarray, y: np.ndarray, path: str) -> None:
        """Сохранение обработанных данных."""
        np.save(f"{path}/x_data.npy", X)
        np.save(f"{path}/y_data.npy", y)