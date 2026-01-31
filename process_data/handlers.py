from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from .utils import DataCleaner, FeatureEncoder


class Handler(ABC):
    """Базовый обработчик"""
    
    def __init__(self):
        self.next = None
    
    def set_next(self, handler):
        self.next = handler
        return handler
    
    @abstractmethod
    def handle(self, df: pd.DataFrame, ctx: dict) -> pd.DataFrame:
        pass
    
    def _next(self, df: pd.DataFrame, ctx: dict) -> pd.DataFrame:
        return self.next.handle(df, ctx) if self.next else df


class LoadHandler(Handler):
    """Загрузка CSV"""
    
    def handle(self, df: pd.DataFrame, ctx: dict) -> pd.DataFrame:
        if df is not None:
            return self._next(df, ctx)
        
        try:
            df = pd.read_csv(ctx['file_path'], low_memory=False)
            print(f"Загружено: {len(df)} строк")
            return self._next(df, ctx)
        except Exception as e:
            print(f"Ошибка загрузки: {e}")
            return None


class CleanHandler(Handler):
    """Очистка данных"""
    
    def handle(self, df: pd.DataFrame, ctx: dict) -> pd.DataFrame:
        df = df.copy()
        
        # Зарплата
        if 'ЗП' in df.columns:
            df['salary'] = df['ЗП'].apply(DataCleaner.clean_salary)
        
        # Возраст и пол
        if 'Пол, возраст' in df.columns:
            df['age'] = df['Пол, возраст'].apply(DataCleaner.extract_age)
            df['gender'] = df['Пол, возраст'].apply(DataCleaner.extract_gender)
            df['is_male'] = (df['gender'] == 'male').astype(int)
        
        # Город
        if 'Город' in df.columns:
            df['city'] = df['Город'].apply(DataCleaner.extract_city)
            # One-hot для топ-5 городов
            top_cities = df['city'].value_counts().head(5).index
            for city in top_cities:
                df[f'city_{city}'] = (df['city'] == city).astype(int)
        
        print(f"После очистки: {len(df)} строк")
        return self._next(df, ctx)


class FillNaHandler(Handler):
    """Заполнение пропусков"""
    
    def handle(self, df: pd.DataFrame, ctx: dict) -> pd.DataFrame:
        df = df.copy()
        
        # Числовые колонки - медиана
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if df[col].isna().any():
                median = df[col].median()
                df[col] = df[col].fillna(median)
        
        # Удаляем строки без зарплаты
        if 'salary' in df.columns:
            before = len(df)
            df = df.dropna(subset=['salary'])
            removed = before - len(df)
            if removed > 0:
                print(f"Удалено {removed} строк без зарплаты")
        
        print(f"После заполнения пропусков: {len(df)} строк")
        return self._next(df, ctx)


class OutlierHandler(Handler):
    """Удаление выбросов"""
    
    def handle(self, df: pd.DataFrame, ctx: dict) -> pd.DataFrame:
        df = df.copy()
        initial = len(df)
        
        # Выбросы по зарплате (правило 3х сигм)
        if 'salary' in df.columns:
            mean = df['salary'].mean()
            std = df['salary'].std()
            if std > 0:
                lower = max(mean - 3*std, 0)
                upper = mean + 3*std
                df = df[(df['salary'] >= lower) & (df['salary'] <= upper)]
        
        # Выбросы по возрасту (IQR)
        if 'age' in df.columns:
            Q1 = df['age'].quantile(0.25)
            Q3 = df['age'].quantile(0.75)
            IQR = Q3 - Q1
            if IQR > 0:
                lower = max(Q1 - 1.5*IQR, 16)
                upper = min(Q3 + 1.5*IQR, 80)
                df = df[(df['age'] >= lower) & (df['age'] <= upper)]
        
        removed = initial - len(df)
        if removed > 0:
            print(f"Удалено выбросов: {removed}")
        
        print(f"После удаления выбросов: {len(df)} строк")
        return self._next(df, ctx)


class FeatureHandler(Handler):
    """Создание признаков"""
    
    def handle(self, df: pd.DataFrame, ctx: dict) -> pd.DataFrame:
        df = df.copy()
        
        # Нормализуем возраст если есть
        if 'age' in df.columns:
            df = FeatureEncoder.normalize(df, ['age'])
        
        # Бинарные признаки из занятости
        if 'Занятость' in df.columns:
            df['is_full_time'] = df['Занятость'].astype(str).str.contains('полная', case=False, na=False).astype(int)
        
        print(f"Создано признаков: {len(df.columns)}")
        return self._next(df, ctx)


class PrepareHandler(Handler):
    """Подготовка к сохранению"""
    
    def handle(self, df: pd.DataFrame, ctx: dict) -> pd.DataFrame:
        # Выбираем только числовые колонки (кроме текстовых)
        num_cols = df.select_dtypes(include=[np.number]).columns
        
        # Исключаем исходные текстовые колонки
        exclude = ['ЗП', 'Пол, возраст', 'Город', 'Занятость', 'gender', 'city']
        feature_cols = [c for c in num_cols if c not in exclude and c != 'salary']
        
        ctx['features'] = feature_cols
        ctx['target'] = 'salary'
        
        print(f"Готово к сохранению. Признаков: {len(feature_cols)}")
        return df