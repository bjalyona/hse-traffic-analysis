from process_data.handlers import *


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
        
        try:
            df = self.first.handle(None, self.ctx)
            
            if df is None or len(df) == 0:
                print("Ошибка: нет данных")
                return None
            
            # Получаем X и y
            X_cols = self.ctx.get('features', [])
            y_col = self.ctx.get('target', 'salary')
            
            if y_col not in df.columns:
                print(f"Ошибка: нет колонки {y_col}")
                return None
            
            X = df[X_cols].values.astype(np.float32)
            y = df[y_col].values.astype(np.float32)
            
            # Заполняем оставшиеся NaN
            X = np.nan_to_num(X)
            y = np.nan_to_num(y)
            
            return X, y
            
        except Exception as e:
            print(f"Ошибка: {e}")
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

            print(f"Данные сохранены:")
            print(f"  Признаки: {x_path} ({X.shape[0]} строк, {X.shape[1]} признаков)")
            print(f"  Целевые значения: {y_path} ({y.shape[0]} значений)")
            return True
            
        except Exception as e:
            print(f"Ошибка сохранения: {e}")
            return False