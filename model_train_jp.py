from cProfile import label

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
# from imblearn.over_sampling import SMOTE

import joblib

f_name = 'datasets/results/df_game_data.xlsx'


def load_data(file_name: str, cols_ls: list) -> pd.DataFrame:
    loaded_data = pd.read_excel(file_name)
    if cols_ls is not None:
        loaded_data = loaded_data[cols_ls]
    return loaded_data


df = load_data(f_name, ['Date', 'Time', 'ID', 'Gender', 'BlockName', 'TaskName', 'TrueAnswerAmount', 'XP_rewarded'])

difficulty_mapping = {
    "Опросник по цифровой грамотности": "easy",
    "Угадай профессию по картинке": "easy",
    "Задание от Binance (Web 3.0)": "moderate",
    "Вопросы Web3.0": "moderate",
    "Угадай запрос по сгенерированной картинке  (через AI)": "moderate_difficult",
    "Определи технологию по визуальной концепции": "moderate",
    "AI Challenge": "moderate_difficult",
    "Определи бизнес модель": "moderate",
    "Стартап культура": "easy",
    "4 картинки 1 слово (BigTech)": "moderate",
    "SuperGame": "supergame",
    "Feed Back": "easy"
}

df["difficulty"] = df["TaskName"].map(difficulty_mapping).fillna("easy")


# # Define time spent distributions based on difficulty
def simulate_time_spent(dff, n_questions):
    if n_questions == 0:
        n_questions = 5
    if dff == "easy":
        base_time = np.random.normal(20, 5, n_questions)
    elif dff == "moderate":
        base_time = np.random.normal(30, 7, n_questions)
    elif dff == "moderate_difficult":
        base_time = np.random.normal(40, 10, n_questions)
    elif dff == "supergame":
        base_time = np.full(n_questions, 15)  # Fixed time for SuperGame
    else:
        base_time = np.random.normal(20, 5, n_questions)  # Default to easy

    return np.clip(base_time, 5, None)  # Minimum time spent is 5 seconds


# Simulate the time spent for each task
# time_spent_data = []
df["time_spent_task"] = np.nan  # Initialize the column first
df["avg_time_per_question_task"] = np.nan  # Initialize the column first

for _, row in df.iterrows():
    block_name = row["BlockName"]  # Replace with actual column name
    task_name = row["TaskName"]  # Replace with actual column name
    num_questions = row["TrueAnswerAmount"]  # Replace with actual column name

    # difficulty = difficulty_mapping.get(task_name, "easy")
    difficulty = row['difficulty']
    time_spent_question = simulate_time_spent(difficulty, num_questions)
    time_spent_task = np.sum(time_spent_question)
    avg_time_per_question_task = np.mean(time_spent_question)

    # df.loc[_, "difficulty"] = difficulty
    df.loc[_, "time_spent_task"] = time_spent_task
    df.loc[_, "avg_time_per_question_task"] = avg_time_per_question_task


def arrange_columns(df_to_arrange: pd.DataFrame, new_column: str, origin_column: str) -> pd.DataFrame:
    cols = df_to_arrange.columns.tolist()
    cols.remove(new_column)
    column_idx = cols.index(origin_column)
    cols.insert(column_idx + 1, new_column)
    df_to_arrange = df_to_arrange[cols]
    return df_to_arrange


def encode_fun(data: pd.DataFrame, col_name: str) -> pd.DataFrame:
    keys = df[col_name].unique()
    vals = list(range(len(df[col_name].unique())))
    my_dict = dict(zip(keys, vals))

    data[col_name + '_encoded'] = data[col_name].map(my_dict)
    return data


# init
def get_question_n(data: pd.DataFrame, col_name: str, is_xp: bool, is_question: bool) -> pd.DataFrame:
    task_names = list(df[col_name].unique())
    if is_xp:
        xp_values = [216, 30, 30, 30, 60, 30, 30, 24, 21, 24, 30, 5, 5, 5, 5, 5, 5, 5]
        tasks_max_val_dct = dict(zip(task_names, xp_values))
        data['Total_xp'] = data['TaskName'].map(tasks_max_val_dct).fillna(1)
        return data
    elif is_question:
        question_n = [72, 10, 10, 10, 20, 10, 10, 8, 7, 8, 10, 1, 1, 1, 1, 1, 1, 1]
        tasks_max_val_dct = dict(zip(task_names, question_n))
        data['Total questions'] = data['TaskName'].map(tasks_max_val_dct).fillna(1)
        return data
    return data


df = get_question_n(df, 'TaskName', False, True)

df_new = df.copy()

# encode
# Calculate overall player knowledge level using clustering


# Encoding and arranging columns
for col in ['ID', 'Gender', 'BlockName', 'TaskName', 'TrueAnswerAmount', 'difficulty']:
    df_new = encode_fun(df_new, col)
    df_new = arrange_columns(df_new, f'{col}_encoded', col)

    # Compute efficiency for the overall player score
    if col == 'TrueAnswerAmount':
        # Calculate efficiency as the ratio of correct answers to total questions
        df_new['efficiency'] = df_new['TrueAnswerAmount'] / df_new['Total questions']
        df_new['performance_time'] = df_new['TrueAnswerAmount'] / df_new['time_spent_task'].max()
        df_new = arrange_columns(df_new, 'Total questions', 'TrueAnswerAmount')
        df_new = arrange_columns(df_new, 'efficiency', 'Total questions')

kn_level = 'stat'  # Kmeans
if kn_level == 'Kmeans':
    features = ['TrueAnswerAmount', 'Total questions', 'efficiency', 'XP_rewarded']
    X = df_new.groupby('ID')[features].sum()  # Aggregate scores per player
    # Normalize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply clustering to assign knowledge levels
    kmeans = KMeans(n_clusters=3, random_state=42)  # Assume 3 knowledge levels
    X_scaled['KnowledgeLevel'] = kmeans.fit_predict(X_scaled)
    # Select features related to player performance
    # Merge knowledge levels back into the original dataframe
    df_new = df_new.merge(X[['KnowledgeLevel']], how='left', left_on='ID', right_index=True)
elif kn_level == 'stat':
    # Assign knowledge levels based on efficiency thresholds
    avg_time_per_task = df_new.groupby('TaskName')['time_spent_task'].mean().reset_index()
    avg_time_per_task.columns = ['TaskName', 'avg_spent_time']  # Rename columns for clarity
    # Merge the calculated average time back to the original dataframe
    df_new = df_new.merge(avg_time_per_task, on='TaskName', how='left')

    conditions = [
        df_new['efficiency'] <= 0.5,  # Condition for level 0
        (df_new['efficiency'] > 0.5) & (df_new['efficiency'] < 0.8),  # Condition for level 1
        (df_new['efficiency'] >= 0.8) & (df_new['time_spent_task'] > df_new['avg_spent_time'] * 1.25),
        # todo eval  # Condition for level 1
        df_new['efficiency'] >= 0.8  # Condition for level 2
    ]
    values = [0, 1, 1, 2]  # Corresponding knowledge levels

    # Use np.select to apply conditions
    df_new['KnowledgeLevel'] = np.select(conditions, values)

# Reorder KnowledgeLevel for better visibility
df_new = arrange_columns(df_new, 'KnowledgeLevel', 'efficiency')

df_ml = df_new[
    ['ID_encoded', 'Gender_encoded', 'BlockName_encoded', 'TaskName_encoded', 'TrueAnswerAmount', 'Total questions',
     'efficiency', 'XP_rewarded', 'KnowledgeLevel', 'difficulty_encoded', 'time_spent_task', 'avg_spent_time',
     'avg_time_per_question_task', 'performance_time']].copy()


def encode_cols(data: pd.DataFrame, cols) -> pd.DataFrame:
    from sklearn.preprocessing import OneHotEncoder

    if not cols:
        return data
    # Select columns to one-hot encode
    columns_to_encode = cols
    # Initialize the encoder
    encoder = OneHotEncoder(sparse_output=False, drop=None)  # sparse_output=False     new version
    # Fit and transform the selected columns
    encoded_data = encoder.fit_transform(data[columns_to_encode])
    # Convert encoded data to DataFrame and add column names
    # new sklearn version
    # encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(columns_to_encode))

    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(columns_to_encode))
    # Concatenate with the original dataframe (excluding the original encoded columns)
    df_encoded = pd.concat([data.drop(columns=columns_to_encode), encoded_df], axis=1)

    return df_encoded


df_ml_encoded = encode_cols(df_ml, ['BlockName_encoded', 'TaskName_encoded'])


def get_mode(series):
    mode = series.mode()
    return mode.iloc[0] if not mode.empty else None


grouped_df_ml = df_ml_encoded.groupby('ID_encoded').agg(
    {
        'efficiency': 'mean',  # Average efficiency
        # 'KnowledgeLevel': 'mean',           # Average knowledge level
        'KnowledgeLevel': get_mode,  # Average knowledge level
        'difficulty_encoded': 'mean',  # Average difficulty
        'avg_spent_time': 'mean',  # Average spent time
        'avg_time_per_question_task': 'mean',  # Average time per question
        'TrueAnswerAmount': 'sum',  # Total true answers
        'Total questions': 'sum',  # Total questions
        'XP_rewarded': 'sum',  # Total XP rewarded
        'time_spent_task': 'sum',  # Total time spent on tasks
        'performance_time': 'sum'  # Example: Max performance time (you can modify this)
    }
).reset_index()  # Reset index if needed

# Get the feature columns of label 2
label_2_data = grouped_df_ml[grouped_df_ml['KnowledgeLevel'] == 2]
X_label_2 = label_2_data[['efficiency', 'TrueAnswerAmount', 'Total questions', 'difficulty_encoded', 'time_spent_task',
                          'avg_spent_time']]

# Generate new synthetic data by sampling from the distribution of each feature
num_synthetic_samples = 100  # Adjust to how many synthetic samples you need
synthetic_data = pd.DataFrame(columns=X_label_2.columns)

for col in X_label_2.columns:
    mean = X_label_2[col].mean()
    std = X_label_2[col].std()
    synthetic_data[col] = np.random.normal(loc=mean, scale=std, size=num_synthetic_samples)

# Add the synthetic data to your original dataset
synthetic_labels = pd.Series([2] * num_synthetic_samples)
synthetic_data['KnowledgeLevel'] = synthetic_labels

#                   GENERATING SYNTHETIC

# Get the feature columns of label 2
X_label_2 = label_2_data[['efficiency', 'TrueAnswerAmount', 'Total questions', 'difficulty_encoded', 'time_spent_task',
                          'avg_spent_time']]

# Find nearest neighbors for each point in label 2
neighbors = NearestNeighbors(n_neighbors=2)  # Adjust to find the nearest neighbors
neighbors.fit(X_label_2)
distances, indices = neighbors.kneighbors(X_label_2)

# Generate synthetic data by interpolating between neighbors
num_synthetic_samples = 800  # Adjust to how many synthetic samples you need
synthetic_data = []

for i in range(num_synthetic_samples):
    # Choose a random sample and its nearest neighbor
    idx = np.random.choice(len(X_label_2))
    neighbor_idx = indices[idx][1]  # Get the nearest neighbor index

    # Interpolate between the points
    synthetic_point = (X_label_2.iloc[idx] + X_label_2.iloc[neighbor_idx]) / 2
    synthetic_data.append(synthetic_point)

# Convert the synthetic data to a DataFrame
synthetic_data = pd.DataFrame(synthetic_data)
synthetic_data['KnowledgeLevel'] = 2  # Label for synthetic data

# Add the synthetic data to your original dataset
grouped_df_ml = pd.concat([grouped_df_ml, synthetic_data])

# Feature and target split
X = grouped_df_ml[['efficiency', 'TrueAnswerAmount', 'Total questions', 'difficulty_encoded', 'time_spent_task',
                   'avg_spent_time']].copy()
# X = grouped_df_ml[['efficiency', 'TrueAnswerAmount', 'Total questions', 'XP_rewarded', 'difficulty_encoded',
# 'time_spent_task', 'avg_spent_time', 'avg_time_per_question_task', 'performance_time']].copy()

# X = df_ml_encoded[['Gender_encoded', 'TrueAnswerAmount', 'Total questions']]
# 'TaskName_encoded_0', 'TaskName_encoded_1', 'TaskName_encoded_2',
# 'TaskName_encoded_3', 'TaskName_encoded_4', 'TaskName_encoded_5',
# 'TaskName_encoded_6', 'TaskName_encoded_7', 'TaskName_encoded_8',
# 'TaskName_encoded_9', 'TaskName_encoded_10', 'TaskName_encoded_11',
# 'TaskName_encoded_12', 'TaskName_encoded_13', 'TaskName_encoded_14',
# 'TaskName_encoded_15', 'TaskName_encoded_16', 'TaskName_encoded_17',
# ]]   #[['TrueAnswerAmount', 'Total questions', 'XP_rewarded', 'Gender_encoded', 'efficiency']]     # basic
# X = df_ml_encoded.drop('KnowledgeLevel', axis=1)
# y = df_ml_encoded['KnowledgeLevel']
y = grouped_df_ml['KnowledgeLevel']

X = X.fillna(X.mean())
# print(X.isnull().sum())
# print(np.isinf(X).sum())
# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# # Apply SMOTE to the training data
# smote = SMOTE(sampling_strategy='auto', random_state=42)
# X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
# # Check the class distribution after resampling
# print("Original class distribution (train):")
# print(y_train.value_counts())
# print("\nResampled class distribution (train):")
# print(y_resampled.value_counts())


# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# model
model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))

# Сохранить модель
joblib.dump(model, 'model/trained_model2.pkl')

# Сохранить стандартайзер (если он нужен для предобработки данных)
joblib.dump(scaler, 'model/scaler2.pkl')

user_inp_example = [100 / 120, 100, 120, 3, 100., 900.]
feature_names = ['efficiency', 'TrueAnswerAmount', 'Total questions', 'difficulty_encoded', 'time_spent_task',
                 'avg_spent_time']
# Create a DataFrame for the user input
user_input_X = pd.DataFrame([user_inp_example], columns=feature_names)

user_input_X_scaled = scaler.transform(user_input_X)
knowledge_pred = model.predict(user_input_X_scaled)  # for 1 subject
print(f"Predicted Knowledge label is: {knowledge_pred}")

# # Загрузить модель
# model = joblib.load('trained_model2.pkl')
#
# # Загрузить стандартайзер
# scaler = joblib.load('scaler2.pkl')
#
# # Пример данных пользователя
# user_input_X = user_input_X  #.#reshape(1, -1)
# user_input_X_scaled = scaler.transform(user_input_X)
#
# # Предсказание знаний
# knowledge_pred = model.predict(user_input_X_scaled)
# print(knowledge_pred)  # [1, 1, 2]

best_params = model.get_params()
best_estim = model.estimators_

print("Best params: ", best_params)
print("Best estim:", best_estim)
print("\n\n",
      ['efficiency', 'TrueAnswerAmount', 'Total questions', 'difficulty_encoded', 'time_spent_task', 'avg_spent_time'])
print("Model Feature importance", model.feature_importances_)
