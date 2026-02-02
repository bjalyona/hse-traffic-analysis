import re
from typing import Optional

import pandas as pd


class DataCleaner:
    """Класс для очистки и извлечения данных из текстовых полей."""

    @staticmethod
    def clean_salary(text: str) -> Optional[float]:
        """Очистка зарплаты из строки, преобразование в рубли."""
        if not isinstance(text, str):
            return None

        # Удаляем все нецифровые символы
        cleaned = re.sub(r'[^\d]', '', text)

        if not cleaned:
            return None

        salary = float(cleaned)

        # Если число маленькое, но есть указание на рубли - умножаем на 1000
        if (salary < 100 and
                any(word in text.lower() for word in ['руб', 'р.', 'зарплат', 'зп'])):
            salary *= 1000

        return salary

    @staticmethod
    def extract_age(text: str) -> Optional[int]:
        """Извлечение возраста из текста."""
        if not isinstance(text, str):
            return None

        match = re.search(r"(\d+)\s*(год|года|лет)", text)
        return int(match.group(1)) if match else None

    @staticmethod
    def extract_experience_years(text: str) -> Optional[float]:
        """Извлечение опыта работы в годах."""
        if not isinstance(text, str):
            return None

        years_match = re.search(r"(\d+)\s*лет", text)
        months_match = re.search(r"(\d+)\s*месяц", text)

        years = int(years_match.group(1)) if years_match else 0
        months = int(months_match.group(1)) if months_match else 0

        total = years + months / 12
        return total if total > 0 else None

    @staticmethod
    def clean_salary_series(series: pd.Series) -> pd.Series:
        """Очистка зарплат из pandas Series."""
        return (
            series.astype(str)
            .str.replace(r"[^\d]", "", regex=True)
            .replace("", pd.NA)
            .replace("nan", pd.NA)
            .astype("Int64")
            .astype(float)
        )