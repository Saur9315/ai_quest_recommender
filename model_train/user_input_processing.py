import joblib
import numpy as np
import pandas as pd


def predict_user_knowledge(user_inp_example2=None, feature_names=None):
    # Загрузить модель
    if user_inp_example2 is None:
        user_inp_example2 = [100 / 120, 100, 120, 3, 100., 900.]
    if feature_names is None:
        feature_names = ['efficiency', 'TrueAnswerAmount', 'Total questions', 'difficulty_encoded',
                         'time_spent_task',
                         'avg_spent_time']
    # Загрузить model
    model = joblib.load('model/trained_model2.pkl')
    # Загрузить стандартайзер
    scaler = joblib.load('model/scaler2.pkl')

    # Create a DataFrame for the user input
    user_input_x = pd.DataFrame([user_inp_example2], columns=feature_names)
    user_input_x_scaled = scaler.transform(user_input_x)

    # Предсказание знаний
    knowledge_pred = model.predict(user_input_x_scaled)
    return knowledge_pred  # [1, 1, 2]
