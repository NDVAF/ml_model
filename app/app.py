import io
import os
import logging
import sqlite3
import uuid
from fastapi import FastAPI, File, UploadFile, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List, Optional
import pandas as pd
from app.helper import prepare_data, calculate_metrics, generate_confusion_matrix, save_dataframe_to_sqlite
from app.factory import ModelFactory
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from app.decorators import metric_calc_time
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse


# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")
engine = create_engine("sqlite:///result.db", echo=True)
engine_dataset= create_engine("sqlite:///dataset.db", echo=True)
global_data = None
global_columns_to_drop = []
df_result = {}
img_path = os.path.join("app","static", f"confusion_matrix.png")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    logger.info("Rendering main page")
    return templates.TemplateResponse("base.html", {"request": request})

@app.post("/upload", response_class=HTMLResponse)
async def upload(request: Request, file: UploadFile = File(...)):
    global global_data
    try:
        logger.info(f"Uploading file: {file.filename}")
        contents = await file.read()
        global_data = pd.read_csv(io.StringIO(contents.decode('utf-8')), encoding='utf-8')
        if global_data.empty:
            logger.warning("Uploaded file is empty")
            return templates.TemplateResponse("error.html", {"request": request, "error": "Загруженный файл пуст"})
        logger.info(f"File {file.filename} uploaded successfully")
        return templates.TemplateResponse("data_table.html", {"request": request, "data": global_data.to_html(index=False)})
    except Exception as e:
        logger.error(f"Error during file upload: {e}")
        return templates.TemplateResponse("error.html", {"request": request, "error": str(e)})


@app.get("/select_columns", response_class=HTMLResponse)
async def select_columns_route(request: Request):
    global global_data
    if global_data is None:
        logger.warning("No data to display for column selection")
        return templates.TemplateResponse("error.html", {"request": request, "error": "Нет данных для отображения"})
    logger.info("Rendering column selection page")
    return templates.TemplateResponse("select_columns.html", {"request": request, "columns": global_data.columns})
    
@app.post("/remove_columns", response_class=HTMLResponse)
async def remove_columns(request: Request, columns_to_drop: List[str] = Form(None)):
    global global_data
    global global_columns_to_drop
    global_columns_to_drop = columns_to_drop if columns_to_drop else []
    try:
        logger.info(f"Columns to drop: {global_columns_to_drop}")
        if global_columns_to_drop:
            global_data = global_data.drop(columns=global_columns_to_drop, errors='ignore')
        logger.info("Columns dropped successfully")
        return templates.TemplateResponse("data_table.html", {"request": request, "data": global_data.to_html(index=False)})
    except Exception as e:
        logger.error(f"Error during column drop: {e}")
        return templates.TemplateResponse("error.html", {"request": request, "error": f"Ошибка при удалении столбцов: {e}"})

@app.post("/save_to_db", response_class=HTMLResponse)
async def save_to_db_route(request: Request):
     global global_data
     if global_data is None:
         logger.warning("No data to save to the database")
         return templates.TemplateResponse("error.html", {"request": request, "error": "Нет данных для записи"})
     try:
         logger.info("Saving data to the database")
         save_dataframe_to_sqlite(global_data, 'dataset.db', 'dataset')
         logger.info("Data saved to database successfully")
         return templates.TemplateResponse("results.html", {"request": request, "message": "Данные успешно записаны в базу данных"})
     except Exception as e:
         logger.error(f"Error during saving to db: {e}")
         return templates.TemplateResponse("error.html", {"request": request, "error": f"Ошибка записи в базу данных: {e}"})

@app.get("/model_selection", response_class=HTMLResponse)
async def model_selection(request: Request, model_type: str = None):
    global global_data
    if global_data is None:
        logger.warning("No data to display for model selection")
        return templates.TemplateResponse("error.html", {"request": request, "error": "Нет данных для обработки"})
    logger.info("Rendering model selection page")
    return templates.TemplateResponse("model_selection.html", {"request": request, "columns": global_data.columns, "model_type": model_type})


@app.post("/process", response_class=HTMLResponse)
async def process(
    request: Request,
    target_column: str = Form(...),
    model_type: str = Form(...),
    criterion: Optional[str] = Form(None),
    max_depth: Optional[int] = Form(None),
    min_samples_leaf: Optional[int] = Form(None),
    random_state: Optional[int] = Form(None),
):
   global global_data
   global global_columns_to_drop
   global df_result
   global img_path
   try:
       logger.info(f"Processing data with target: {target_column}, model: {model_type}")
       if global_data is None:
           logger.error("No data to process")
           return templates.TemplateResponse("error.html", {"request": request, "error": "Нет данных для обработки"})
        
       num_cols_before = len(global_data.columns)
              
       X_train, X_test, y_train, y_test = prepare_data(global_data, target_column, global_columns_to_drop, encode=True)
       num_cols_after = len(X_train[0])
       
       if num_cols_after>200:
           logger.warning(f"The number of columns {num_cols_after} is greater than 200, skipping One-Hot Encoding")
           return templates.TemplateResponse("results_.html", {"request": request, "message": f"Кол-во столбцов: {num_cols_after} после горячего кодирование превышает 200. Необходимо пересмотреть категориальные признаки в данных и удалить не влияющие на результат обучения. "}) 
       elif 20<num_cols_after<200:
           X_train, X_test, y_train, y_test = prepare_data(global_data, target_column, global_columns_to_drop, encode=False)
           logger.info(f"After turning off One-Hot Encoding number of columns: {len(X_train[0])}")

           params = {}
           if model_type == "decision_tree":
                params["criterion"] = criterion if criterion else "entropy"
                params["max_depth"] = max_depth if max_depth else 8
                params["min_samples_leaf"] = min_samples_leaf if min_samples_leaf else 10
                params["random_state"] = random_state if random_state else 42
 


           factory = ModelFactory()
           model = factory.create_model(model_type, params) if model_type == "decision_tree" else factory.create_model(model_type)
           y_pred_train, y_pred_test = model.train_and_predict(X_train, X_test, y_train)
           metrics_result = calculate_metrics(y_train, y_pred_train, y_test, y_pred_test)
           
           generate_confusion_matrix(y_train, y_pred_train, y_test, y_pred_test, img_path)
           
           calc_time = metric_calc_time()
           logger.info("Data processed successfully, skipping One-Hot Encoding")
           #
           f1_train = metrics_result.get('f1_train', 'N/A')
           f1_test = metrics_result.get('f1_test', 'N/A')
           quality_train = metrics_result.get('quality_train', 'N/A')
           quality_test = metrics_result.get('quality_test', 'N/A')

           result_model = {
            "model_type": [model_type],
            "target_column": [target_column],
            "num_cols_before": [num_cols_before],
            "num_cols_after": [num_cols_after],
            "calc_time": [calc_time], 
            "f1_score_train": [f1_train],
            "f1_score_test": [f1_test],
            "quality_train":[quality_train],
            "quality_test":[quality_test],
            
           }
           df_result = pd.DataFrame(result_model)
           return  templates.TemplateResponse("results.html", {"request": request, "metrics": metrics_result, "calc_time": calc_time, "message": f"Кол-во столбцов: {num_cols_after}. One-Hot Encoding был пропущен."})
 
       else:
           
           logger.info(f"After turning on One-Hot Encoding number of columns: {len(X_train[0])}")
           params = {}
           if model_type == "decision_tree":
                params["criterion"] = criterion if criterion else "entropy"
                params["max_depth"] = max_depth if max_depth else 8
                params["min_samples_leaf"] = min_samples_leaf if min_samples_leaf else 10
                params["random_state"] = random_state if random_state else 42

           factory = ModelFactory()
           model = factory.create_model(model_type, params) if model_type == "decision_tree" else factory.create_model(model_type)
           y_pred_train, y_pred_test = model.train_and_predict(X_train, X_test, y_train)
           metrics_result = calculate_metrics(y_train, y_pred_train, y_test, y_pred_test)
                     
           #img_path = os.path.join("app","static", f"confusion_matrix.png")
           generate_confusion_matrix(y_train, y_pred_train, y_test, y_pred_test, img_path)
           
           
           

           calc_time = metric_calc_time()
           logger.info("Data processed successfully with One-Hot Encoding")
           
           f1_train = metrics_result.get('f1_train', 'N/A')
           f1_test = metrics_result.get('f1_test', 'N/A')
           quality_train = metrics_result.get('quality_train', 'N/A')
           quality_test = metrics_result.get('quality_test', 'N/A')


           result_model = {
            "model_type": [model_type],
            "target_column": [target_column],
            "num_cols_before": [num_cols_before],
            "num_cols_after": [num_cols_after],
            "calc_time": [calc_time], 
            "f1_score_train": [f1_train],
            "f1_score_test": [f1_test],
            "quality_train":[quality_train],
            "quality_test":[quality_test],
                        
            }
           df_result = pd.DataFrame(result_model)
           
                     
           return  templates.TemplateResponse("results.html", {"request": request, "metrics": metrics_result, "calc_time": calc_time, "model_type": model_type, "num_cols_before": num_cols_before,"num_cols_after": num_cols_after,})
   except Exception as e:
       logger.error(f"Error during data processing: {e}")
       return templates.TemplateResponse("error.html", {"request": request, "error": str(e)})

@app.post("/save_results", response_class=HTMLResponse)
async def save_results_route(request: Request):
    global df_result
    try:
        logger.info("Saving results to the database")
        
         
        with engine.connect() as conn:
            df_result.to_sql('results', conn, if_exists='append', index=False)
        logger.info("Results saved to database successfully")
        return templates.TemplateResponse("results.html", {"request": request, "message": "Данные успешно записаны в базу данных"})
    except SQLAlchemyError as e:
         logger.error(f"Error during saving to db: {e}")
         return templates.TemplateResponse("error.html", {"request": request, "error": f"Ошибка записи в базу данных: {e}"})
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        return templates.TemplateResponse("error.html", {"request": request, "error": f"Произошла непредвиденная ошибка: {e}"})

@app.get("/show_data_result", response_class=HTMLResponse)
def show_data(request: Request):
    with engine.connect() as conn:
        # Выполняем запрос
        result = conn.execute(text("SELECT * FROM results"))
        columns = result.keys()         # получаем список столбцов
        rows = result.fetchall()        # получаем данные
    df = pd.DataFrame(rows, columns=columns)

    return templates.TemplateResponse(
        "table.html",
        {
            "request": request,
            "data": df.to_dict(orient="records"),
            "cols": list(df.columns),
        }
    )
@app.get("/show_data", response_class=HTMLResponse)
async def show_data(request: Request):
    with engine_dataset.connect() as conn:
        # Выполняем запрос
        result = conn.execute(text("SELECT * FROM dataset"))
        columns = result.keys()         # получаем список столбцов
        rows = result.fetchall()        # получаем данные
    df = pd.DataFrame(rows, columns=columns)

    return templates.TemplateResponse(
        "table_dataset.html",
        {
            "request": request,
            "data": df.to_dict(orient="records"),
            "cols": list(df.columns),
        }
    )

@app.get("/model_selection", response_class=HTMLResponse)
async def model_selection(request: Request):
    global global_data
    if global_data is None:
        logger.warning("No data to display for model selection")
        return templates.TemplateResponse("error.html", {"request": request, "error": "Нет данных для обработки"})
    logger.info("Rendering model selection page")
    return templates.TemplateResponse("model_selection.html", {"request": request, "columns": global_data.columns})


@app.get("/confusion_matrix", response_class = FileResponse)
async def confusion_matrix():
    global img_path
    return  img_path

@app.get("/save_excel", response_class=FileResponse)
async def save_excel(request: Request):
    try:
        logger.info("Saving results to Excel")
        with engine.connect() as conn:
            result = conn.execute(text("SELECT * FROM results"))
            columns = result.keys()
            rows = result.fetchall()
        df = pd.DataFrame(rows, columns=columns)
        excel_filename = os.path.join("app", "static", f"results.xlsx")
        df.to_excel(excel_filename, index=False)
        logger.info("Results saved to Excel successfully")
        return FileResponse(excel_filename, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", filename="results.xlsx")
    except Exception as e:
        logger.error(f"Error during saving Excel: {e}")
        return HTMLResponse(f"An error occurred: {e}", status_code=500)