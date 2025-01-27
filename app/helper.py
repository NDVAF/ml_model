import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os
import logging
from app.decorators import quality
from io import BytesIO
#from starlette.responses import StreamingResponse


# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def prepare_data(data, target_column, columns_to_drop=[],encode=True):
    try:
        # Удаление выбранных столбцов
        logger.info(f"Droping columns: {columns_to_drop}")
        if columns_to_drop:
            data = data.drop(columns=columns_to_drop, errors='ignore')
    except Exception as e:
          logger.error(f"Error when droping columns: {e}")
          raise
    
    try:
        logger.info("Handling missing values")
        # Обработка пропущенных значений (NaN): заполнение средними значениями
        numeric_cols = data.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            data[col] = data[col].fillna(data[col].mean())
        # Обработка пропущенных значений (NaN): заполнение модой
        for col in data.select_dtypes(exclude=['number']):
            data[col] = data[col].fillna(data[col].mode()[0])

        logger.info("Handling duplicates")
        # Удаление дубликатов
        data.drop_duplicates(inplace=True)

        logger.info(f"Data types before encoding: {data.dtypes}")
        if target_column not in data.columns:
            logger.error(f"Target column '{target_column}' not found in data")
            raise ValueError(f"Target column '{target_column}' not found in data")

        if encode:
            logger.info("Encoding categorical features")
            num_cols_before = len(data.columns)
            logger.info(f"Number of columns before One-Hot Encoding: {num_cols_before}")
            data = pd.get_dummies(data)
            num_cols_after = len(data.columns)
            print(num_cols_after)
            logger.info(f"Number of columns after One-Hot Encoding: {num_cols_after}")
            if num_cols_after > 200:
                logger.warning("The number of columns is greater than 200.  Consider reducing dimensionality.")
        else:
            logger.info("Skipping encoding of categorical features")

        logger.info("Preparing data")

        # Разделение на признаки (X) и целевую переменную (y)
        X = data.drop(target_column, axis=1, errors='ignore')
        y = data[target_column]
        # Разбиение данных на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    except Exception as e:
            logger.error(f"Error preparing {e}")
            raise

    # Масштабирование числовых признаков
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    logger.info("Data prepared successfully")
    return X_train, X_test, y_train, y_test


def calculate_metrics(y_train, y_pred_train, y_test, y_pred_test):
    logger.info("Calculating metrics")
    f1_train = metrics.f1_score(y_train, y_pred_train)
    f1_test = metrics.f1_score(y_test, y_pred_test)
    message_train, f1_train_quality  = quality(f1_train)
    message_test, f1_test_quality = quality(f1_test)
   

    metrics_result = {
        "train": metrics.classification_report(y_train, y_pred_train, output_dict=True),
        "test": metrics.classification_report(y_test, y_pred_test, output_dict=True),
         "f1_train":  f1_train_quality,
         "f1_test":   f1_test_quality,
          "quality_train":  message_train,
           "quality_test":  message_test
    }
    logger.info("Metrics calculated successfully")
    return metrics_result

def generate_confusion_matrix(y_train, y_pred_train, y_test, y_pred_test, filename):
    logger.info("Generating confusion matrix")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(metrics.confusion_matrix(y_train, y_pred_train), annot=True, fmt='', ax=axes[0], cmap='Blues')
    axes[0].set_title('Матрица ошибок для тренировочной выборки')
    axes[0].set_xlabel('Прогноз')
    axes[0].set_ylabel('Факт')
    sns.heatmap(metrics.confusion_matrix(y_test, y_pred_test), annot=True, fmt='', ax=axes[1], cmap='Blues')
    axes[1].set_title('Матрица ошибок для тестовой выборки')
    axes[1].set_xlabel('Прогноз')
    axes[1].set_ylabel('Факт')

    plt.savefig(filename)
    plt.close()
    logger.info(f"Confusion matrix generated successfully")
    return filename
    

def save_dataframe_to_sqlite(df, db_path, table_name):
    import sqlite3
    try:
        logger.info(f"Saving data to SQLite, database path:{db_path}, table_name: {table_name}")
        conn = sqlite3.connect(db_path)
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        conn.close()
        logger.info("Data saved to SQLite successfully")
        return True
    except Exception as e:
        logger.error(f"Error saving to SQLite: {e}")
        return False