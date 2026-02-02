from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd

from .utils import DataCleaner


class Handler(ABC):
    """Абстрактный обработчик для цепочки ответственности."""

    def __init__(self) -> None:
        """Инициализация обработчика."""
        self._next_handler: Optional['Handler'] = None

    def set_next(self, handler: 'Handler') -> 'Handler':
        """Установка следующего обработчика в цепочке."""
        self._next_handler = handler
        return handler

    @abstractmethod
    def handle(self, df: Optional[pd.DataFrame], ctx: dict) -> pd.DataFrame:
        """Обработка данных."""
        pass

    def _call_next(self, df: pd.DataFrame, ctx: dict) -> pd.DataFrame:
        """Вызов следующего обработчика, если он существует."""
        if self._next_handler:
            return self._next_handler.handle(df, ctx)
        return df


class LoadHandler(Handler):
    """Обработчик загрузки данных из файла."""

    def handle(self, df: Optional[pd.DataFrame], ctx: dict) -> pd.DataFrame:
        """Загрузка CSV файла."""
        df = pd.read_csv(
            ctx["file_path"],
            encoding="utf-8",
            sep=",",
            quotechar='"',
            engine="python"
        )
        df.columns = df.columns.str.strip()
        return self._call_next(df, ctx)


class FeatureExtractionHandler(Handler):
    """Обработчик извлечения признаков."""

    def handle(self, df: pd.DataFrame, ctx: dict) -> pd.DataFrame:
        """Извлечение признаков из сырых данных."""
        df = df.copy()

        # Очистка зарплаты
        df["salary"] = df["ЗП"].apply(DataCleaner.clean_salary)

        # Демографические признаки
        df["age"] = df["Пол, возраст"].apply(DataCleaner.extract_age)
        df["is_male"] = df["Пол, возраст"].str.contains("Мужчина", na=False).astype(int)

        # Опыт работы
        experience_col = "Опыт (двойное нажатие для полной версии)"
        df["experience_years"] = df[experience_col].apply(DataCleaner.extract_experience_years)

        # Бинарные признаки
        df["has_car"] = df["Авто"].str.contains("Имеется", na=False).astype(int)
        df["full_time"] = df["Занятость"].str.contains("полная", na=False).astype(int)
        df["remote_allowed"] = df["Город"].str.contains("удален", na=False).astype(int)

        # Образование
        education_col = "Образование и ВУЗ"
        df["higher_education"] = df[education_col].str.contains(
            "высшее", na=False, case=False
        ).astype(int)

        return self._call_next(df, ctx)


class PrepareHandler(Handler):
    """Обработчик подготовки данных для модели."""

    def handle(self, df: pd.DataFrame, ctx: dict) -> pd.DataFrame:
        """Подготовка и очистка данных."""
        features = [
            "age",
            "is_male",
            "experience_years",
            "has_car",
            "full_time",
            "remote_allowed",
            "higher_education",
        ]

        # Удаление пропусков
        df = df.dropna(subset=features + ["salary"])

        # Фильтрация некорректных зарплат
        df = df[df["salary"] > 0]

        # Удаление выбросов по зарплате (5%-95%)
        salary_q1 = df["salary"].quantile(0.05)
        salary_q3 = df["salary"].quantile(0.95)
        df = df[(df["salary"] >= salary_q1) & (df["salary"] <= salary_q3)]

        # Фильтрация по возрасту
        df = df[(df["age"] >= 18) & (df["age"] <= 70)]

        # Фильтрация по опыту
        df = df[(df["experience_years"] >= 0) & (df["experience_years"] <= 50)]

        # Сохранение информации о признаках
        ctx["features"] = features
        ctx["target"] = "salary"

        return df