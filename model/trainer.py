from pathlib import Path
from typing import Dict

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler


class ModelTrainer:
    """Класс для обучения и оценки моделей машинного обучения."""

    def __init__(self, models_dir: str = "resources") -> None:
        """Инициализация тренера моделей."""
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)

        self.scaler = StandardScaler()
        self.best_model = None
        self.best_score = -np.inf
        self.best_name = None

    def train_from_files(self, x_path: str, y_path: str) -> None:
        """Обучение моделей на данных из файлов."""
        # Загрузка данных
        X = np.load(x_path)
        y = np.load(y_path)

        # Разделение на train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Масштабирование признаков
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        # Определение моделей для сравнения
        models: Dict[str, object] = {
            "Ridge": Ridge(alpha=10.0),
            "GradientBoosting": GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                random_state=42
            ),
            "RandomForest": RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        }

        print("Метрики моделей:")
        for name, model in models.items():
            # Кросс-валидация
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            cv_r2 = cv_scores.mean()

            # Обучение модели
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Расчет метрик
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            print(f"{name}: R²={r2:.4f}, CV_R²={cv_r2:.4f}, MAE={mae:.0f} руб")

            # Выбор лучшей модели
            if r2 > self.best_score:
                self.best_score = r2
                self.best_model = model
                self.best_name = name

        # Сохранение лучшей модели и скейлера
        joblib.dump(self.best_model, self.models_dir / "best_model.joblib")
        joblib.dump(self.scaler, self.models_dir / "scaler.joblib")

        print(f"Лучшая модель: {self.best_name} (R²={self.best_score:.4f})")