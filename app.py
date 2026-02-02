import sys
import argparse
from pathlib import Path
import numpy as np

from process_data import Pipeline
from model.trainer import ModelTrainer
from model.predictor import SalaryPredictor


def main():
    parser = argparse.ArgumentParser(description="Обработка данных hh.ru и предсказание зарплат")

    # Создаем подкоманды
    subparsers = parser.add_subparsers(dest='command', help='Команды')

    # Команда parse
    parse_parser = subparsers.add_parser('parse', help='Обработка CSV файла')
    parse_parser.add_argument('csv_file', help='Путь к CSV файлу')

    # Команда train
    train_parser = subparsers.add_parser('train', help='Обучение модели')
    train_parser.add_argument('x_file', help='Путь к x_data.npy')
    train_parser.add_argument('y_file', help='Путь к y_data.npy')

    # Команда predict
    predict_parser = subparsers.add_parser('predict', help='Предсказание зарплат')
    predict_parser.add_argument('npy_file', help='Путь к .npy файлу с данными')

    args = parser.parse_args()

    if args.command == 'parse':
        # Обработка CSV
        pipeline = Pipeline()
        X, y = pipeline.run(args.csv_file)
        Pipeline.save(X, y, ".")
        print("Данные сохранены в x_data.npy, y_data.npy")

    elif args.command == 'train':
        # Обучение модели
        if not Path(args.x_file).exists():
            print(f"Ошибка: файл {args.x_file} не найден")
            sys.exit(1)
        if not Path(args.y_file).exists():
            print(f"Ошибка: файл {args.y_file} не найден")
            sys.exit(1)

        trainer = ModelTrainer()
        trainer.train_from_files(args.x_file, args.y_file)

    elif args.command == 'predict':
        # Предсказание
        if not Path("resources/best_model.joblib").exists():
            print("Ошибка: модель не обучена")
            print("Сначала выполните: python app.py train x_data.npy y_data.npy")
            sys.exit(1)

        predictor = SalaryPredictor()
        predictions = predictor.predict_from_file(args.npy_file)
        np.save("predictions.npy", predictions)
        print(f"Предсказания сохранены в predictions.npy")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()