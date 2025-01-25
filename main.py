from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib
import pandas as pd
import sqlite3
from dotenv import load_dotenv
import os
from database import update_results, get_leaderboard_results, create_results_table


create_results_table()
app = FastAPI()

# Mount static files (CSS, JS, images)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Load model and scaler
model = joblib.load('model/trained_model2.pkl')
scaler = joblib.load('model/scaler2.pkl')


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict/", response_class=HTMLResponse)
async def predict(
    request: Request,
    username: str = Form(...),
    # efficiency: float = Form(...),
    true_answer_amount: int = Form(...),
    total_questions: int = Form(...),
    difficulty_encoded: int = Form(...),
    time_spent_task: float = Form(...),
    avg_spent_time: float = Form(...)
):
    # Process user input
    feature_names = ['efficiency', 'TrueAnswerAmount', 'Total questions', 'difficulty_encoded', 'time_spent_task',
                     'avg_spent_time']
    user_input = pd.DataFrame(
        [[true_answer_amount/total_questions, true_answer_amount, total_questions, difficulty_encoded, time_spent_task,
          avg_spent_time]],
        # [[efficiency, true_answer_amount, total_questions, difficulty_encoded, time_spent_task, avg_spent_time]],
        columns=feature_names)
    user_input_scaled = scaler.transform(user_input)
    knowledge_pred = model.predict(user_input_scaled)

    print(knowledge_pred)
    update_results(true_answer_amount/total_questions, true_answer_amount, total_questions, difficulty_encoded,
                   time_spent_task, avg_spent_time, int(knowledge_pred[0]), username)

    # Return prediction result
    return templates.TemplateResponse("index.html", {
        "request": request,
        "username": username,
        "prediction": knowledge_pred[0]
    })


@app.get("/leaderboard")
async def get_leaderboard(request: Request):

    results = get_leaderboard_results()
    print(results)
    leaderboard = [{"username": row[0], "efficiency": row[1], "prediction": int(row[2])} for row in results]
    return templates.TemplateResponse("leaderboard.html", {"request": request, "leaderboard": leaderboard})


if __name__ == '__main__':
    pass

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
