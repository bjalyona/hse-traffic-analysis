import re
import numpy as np
import pandas as pd
from typing import Optional, List


class DataCleaner:
    """Утилиты для очистки данных"""
    
    @staticmethod
    def clean_salary(salary_str: str) -> Optional[float]:
        """Очищает зарплату: '27 000 руб.' -> 27000.0"""
        if pd.isna(salary_str):
            return None
            
        if isinstance(salary_str, (int, float)):
            return float(salary_str)
            
        if not isinstance(salary_str, str):
            return None
        
        # Удаляем всё кроме цифр и пробелов
        cleaned = re.sub(r'[^\d\s]', '', salary_str)
        try:
            return float(cleaned.replace(' ', ''))
        except:
            return None
    
    @staticmethod
    def extract_age(text: str) -> Optional[int]:
        """Извлекает возраст: 'Мужчина, 42 года' -> 42"""
        if pd.isna(text) or not isinstance(text, str):
            return None
            
        match = re.search(r'(\d+)\s*год', text)
        return int(match.group(1)) if match else None
    
    @staticmethod
    def extract_gender(text: str) -> str:
        """Извлекает пол: 'Мужчина' -> 'male'"""
        if pd.isna(text) or not isinstance(text, str):
            return 'unknown'
            
        if 'Мужчина' in text:
            return 'male'
        elif 'Женщина' in text:
            return 'female'
        return 'unknown'
    
    @staticmethod
    def extract_city(text: str) -> Optional[str]:
        """Извлекает город: 'Москва, готов к переезду' -> 'Москва'"""
        if pd.isna(text) or not isinstance(text, str):
            return None
            
        parts = text.split(',')
        return parts[0].strip() if parts else None


class FeatureEncoder:
    """Утилиты для кодирования признаков"""
    
    @staticmethod
    def normalize(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """Нормализация: (x - min) / (max - min)"""
        for col in cols:
            if col in df.columns:
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val > min_val:
                    df[f'{col}_norm'] = (df[col] - min_val) / (max_val - min_val)
                else:
                    df[f'{col}_norm'] = 0
        return df