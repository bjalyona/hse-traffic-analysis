import numpy as np
import pandas as pd
from handlers import *


class Pipeline:
    """Пайплайн обработки"""
    
    def __init__(self):
        # Создаем цепочку
        load = LoadHandler()
        clean = CleanHandler()
        fill = FillNaHandler()
        outlier = OutlierHandler()
        features = FeatureHandler()
        prepare = PrepareHandler()
        
        load.set_next(clean).set_next(fill).set_next(outlier).set_next(features).set_next(prepare)
        self.first = load
        self.ctx = {}
    
    def run(self, file_path: str):
        """Запуск обработки"""
        self.ctx = {'file_path': file_path}
        
        print("=" * 50)
        print("ЗАПУСК ОБРАБОТКИ")
        print("=" * 50)
        
        try:
            df = self.first.handle(None, self.ctx)
            
            if df is None or len(df) == 0:
                print("ОШИБКА: нет данных")
                return None
            
            # Получаем X и y
            X_cols = self.ctx.get('features', [])
            y_col = self.ctx.get('target', 'salary')
            
            if y_col not in df.columns:
                print(f"ОШИБКА: нет колонки {y_col}")
                return None
            
            X = df[X_cols].values.astype(np.float32)
            y = df[y_col].values.astype(np.float32)
            
            # Заполняем оставшиеся NaN
            X = np.nan_to_num(X)
            y = np.nan_to_num(y)
            
            print(f"\nРЕЗУЛЬТАТ:")
            print(f"X: {X.shape} (признаки: {X_cols})")
            print(f"y: {y.shape}")
            print(f"Пример зарплат: {y[:5]}")
            
            return X, y
            
        except Exception as e:
            print(f"ОШИБКА: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def save(self, X, y, path="."):
        """Сохранение в .npy"""
        import os
        
        try:
            x_path = os.path.join(path, "x_data.npy")
            y_path = os.path.join(path, "y_data.npy")
            
            np.save(x_path, X)
            np.save(y_path, y)
            
            print(f"\nСОХРАНЕНО:")
            print(f"{x_path} - {X.shape}")
            print(f"{y_path} - {y.shape}")
            return True
            
        except Exception as e:
            print(f"ОШИБКА СОХРАНЕНИЯ: {e}")
            return False