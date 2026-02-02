# HSE Traffic Analysis - Прогнозирование зарплат

## Установка
```bash
pip install -r requirements.txt
```

## Использование

### 1. Обработка CSV файла
```bash
python app.py parse hh.csv
```
Создает файлы `x_data.npy` (признаки) и `y_data.npy` (зарплаты в рублях).

### 2. Обучение модели
```bash
python app.py train x_data.npy y_data.npy
```
Обучает модели и сохраняет лучшую в папку `resources/`.

### 3. Предсказание зарплат
```bash
python app.py predict данные_для_предсказания.npy
```
Создает файл `predictions.npy` с предсказанными зарплатами в рублях.

## Пример полного пайплайна
```bash
# 1. Обработка данных
python app.py parse hh.csv

# 2. Обучение модели
python app.py train x_data.npy y_data.npy

# 3. Предсказание (на тех же данных)
python app.py predict x_data.npy
```

## Файлы
- `x_data.npy` - признаки для обучения/предсказания
- `y_data.npy` - целевые значения (зарплаты) в рублях
- `predictions.npy` - результаты предсказаний
- `resources/best_model.joblib` - сохраненная лучшая модель