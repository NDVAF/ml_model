# ML Web App

## Описание

Этот проект представляет собой веб-приложение для обучения моделей машинного обучения на загружаемых пользователем данных, 
где целевой признак является **бинарным** (то есть, имеет только два возможных значения, например, 0 и 1, или "да" и "нет"). 

## Приложение позволяет пользователям:

- Загружать CSV файлы с данными.
- Выполнять предобработку (удаление категориальных признаков, не влияющих на результат обучения).
- Выбирать **бинарный** целевой столбец и модель машинного обучения (логистическая регрессия, дерево решений, случайный лес).
- Настраивать параметры для модели "дерево решений".
- Просматривать метрики обученной модели (F1-score, recall, precision, accuracy).
- Оценить качество модели.
- Визуализировать матрицу ошибок.
- Сохранять результаты обучения в базе данных SQLite.
- Экспортировать результаты обучения в файл Excel.

## Технологии

- **Python:** Основной язык программирования.
- **FastAPI:** Веб-фреймворк для создания API.
- **Pydantic:** Для валидации и управления параметрами моделей.
- **Jinja2:** Шаблонизатор для HTML.
- **Scikit-learn:** Библиотека для машинного обучения.
- **Pandas:** Библиотека для работы с данными.
- **Matplotlib и Seaborn:** Библиотеки для визуализации.
- **SQLite:** База данных для хранения результатов.
- **SQLAlchemy:** ORM для работы с базами данных.

## Структура проекта

ml-web-app/
├── app/
│   ├── __init__.py
│   ├── app.py           # Основной файл приложения FastAPI
│   ├── decorators.py    # Декораторы
│   ├── factory.py       # Фабрика для создания моделей
│   ├── helper.py        # Вспомогательные функции
│   ├── models.py        # Pydantic модель для параметров
│   ├── static/          # Статические файлы (CSS, изображения)
│   │   └── styles.css
│   └── templates/       # HTML шаблоны
│       ├── base.html
│       ├── data_table.html
│       ├── error.html
│       ├── model_selection.html
│       ├── results.html
│       ├── select_columns.html
│       ├── table.html
│       └── table_dataset.html
├── result.db            # База данных SQLite для результатов
├── dataset.db           # База данных SQLite для датасета
├── data                 # Датасет для примера работы приложения  
└── README.md


## Как запустить

1. **Клонировать репозиторий:**

   ```bash
   git clone https://github.com/NDVAF/ml_model.git
   cd ml-web-app

2. **Создать и активировать виртуальное окружение (рекомендовано):**

python -m venv venv
source venv/bin/activate  # Для Linux/macOS
venv\Scripts\activate  # Для Windows

3. **Установить зависимости:**

pip install -r requirements.txt

4. **Запустить приложение:**

uvicorn main:app --reload

После запуска, приложение будет доступно по адресу http://localhost:8000.

## Использование

1. Загрузка данных:
    - Откройте веб-страницу в браузере.
    - Перейдите на страницу “Загрузка” и загрузите CSV файл. Для демонстрации работы приложения используйте датасет в папке data\churn.csv (Данные для прогнозирования оттока клиентов.)
    - Просмотрите таблицу с загруженными данными.
2. Удаление колонок:
    - Перейдите на страницу “Выбор столбцов” и выберите столбцы, которые хотите исключить из анализа (категориальные признаки не влияющие на результат обучения).
      В предложенном датасете это признаки: "RowNumber", "CustomerId", "Surname". 
3. Сохранить датасет:
    - После удаления столбцов перейдите на страницу “Сохранить датасет” для записи текущего датасета в базу.
4. Выбор модели и обучение:
    - Перейдите на страницу “Обработка данных”.
    - Выберите целевой столбец  и модель машинного обучения. В примере "Exited"
    - Если вы выбрали модель “Дерево решений”, укажите необходимые параметры.
    - Нажмите “Рассчитать”, чтобы запустить процесс обучения и получить метрики.
5. Просмотр результатов:
    - Просмотрите метрики обученной модели и матрицу ошибок.
    - При необходимости, сохраните результаты в базу данных или экспортируйте в Excel.
6. Просмотр результатов в базе:
    - Перейдите на страницу “Показать результат обучения модели”.
    - Просмотрите результаты обучения модели, сохраненные в базе данных.
7. Просмотр датасета в базе:
    - Перейдите на страницу “Показать датасет”.
    - Просмотрите сохраненный датасет.
8. Просмотр матрицы ошибок:
    - Перейдите на страницу “Показать матрицу ошибок”.
    - Просмотрите матрицу ошибок модели.

## Планируемые улучшения

    Добавить поддержку других типов моделей машинного обучения.
    Расширить функциональность для предобработки данных.
    Добавить более подробную визуализацию результатов.
    Добавить возможность предсказания класса после выбора модели, которая показала лучший результат.
    Улучшить интерфейс пользователя.
    Развернуть приложение в облаке.

## Автор
Волкова Наталья
