#!/usr/bin/env python3
"""
Минимальный скрипт обработки
Использование: python app.py путь/к/файлу.csv
"""

import sys
import os
from pipeline import Pipeline


def main():
    if len(sys.argv) < 2:
        print("Укажите путь к CSV файлу")
        print("Пример: python app.py hh.csv")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    if not os.path.exists(file_path):
        print(f"Файл не найден: {file_path}")
        sys.exit(1)
    
    # Запускаем пайплайн
    pipeline = Pipeline()
    result = pipeline.run(file_path)
    
    if result:
        X, y = result
        pipeline.save(X, y)
    else:
        print("Ошибка обработки")
        sys.exit(1)


if __name__ == "__main__":
    main()