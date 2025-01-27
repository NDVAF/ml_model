import time
import logging
from app.factory import ModelFactory
# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Декоратор для измерения времени выполнения
def time_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"Время выполнения функции {func.__name__}: {end_time - start_time:.4f} секунд")
        return result  # Возвращаем результат функции
    return wrapper

# Декоратор для оценки качества модели по F1-score
def model_quality(massage):
    def decorator(func):
        def wrapper(f1):
            quality_message = ""
            if 1 > f1 > 0.5:
                quality_message =  f"Модель обучена. {massage} > 0.5"
            elif f1 == 1:
                quality_message =  f"Модель переобучена. {massage} = 1"
            else:
                quality_message =  f"Модель недообучена. {massage} < 0.5"
            return  quality_message, func(f1) # Возвращаем сообщение и результат
        return wrapper
    return decorator
    
@model_quality("Значение метрики F1-score")
def quality(f1):
    return f1

@time_decorator
def metric_calc_time():
   start_time = time.time()
   factory = ModelFactory()
   print(factory)
   time.sleep(2)
   end_time = time.time()
   calc_time = f"Время обучения модели: {end_time - start_time:.4f} секунд"
   return calc_time