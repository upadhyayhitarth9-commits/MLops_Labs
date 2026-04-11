"""
dags/diabetes_dag.py
Airflow DAG for Diabetes Prediction using Decision Tree Classifier.
"""

import sys
import os
sys.path.insert(0, '/opt/airflow/dags')

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from airflow import configuration as conf

import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

conf.set("core", "enable_xcom_pickling", "True")

default_args = {
    "owner": "hitarth",
    "start_date": datetime(2024, 1, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

def load_data():
    df = pd.read_csv("/opt/airflow/working_data/diabetes.csv")
    print(f"[load_data] Loaded {df.shape[0]} rows and {df.shape[1]} columns.")
    return pickle.dumps(df)

def preprocess_data(serialized_data):
    df = pickle.loads(serialized_data)
    zero_not_valid = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[zero_not_valid] = df[zero_not_valid].replace(0, df[zero_not_valid].median())
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    with open("/opt/airflow/working_data/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print(f"[preprocess_data] Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")
    payload = {
        "X_train": X_train_scaled, "X_test": X_test_scaled,
        "y_train": y_train.values, "y_test": y_test.values,
    }
    return pickle.dumps(payload)

def train_model(serialized_splits, model_path):
    payload = pickle.loads(serialized_splits)
    model = DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42)
    model.fit(payload["X_train"], payload["y_train"])
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"[train_model] Model saved. Depth: {model.get_depth()}, Leaves: {model.get_n_leaves()}")
    return pickle.dumps({"X_test": payload["X_test"], "y_test": payload["y_test"], "model_path": model_path})

def evaluate_model(serialized_eval_payload):
    payload = pickle.loads(serialized_eval_payload)
    with open(payload["model_path"], "rb") as f:
        model = pickle.load(f)
    y_pred = model.predict(payload["X_test"])
    acc = accuracy_score(payload["y_test"], y_pred)
    report = classification_report(payload["y_test"], y_pred, target_names=["No Diabetes", "Diabetes"])
    print(f"[evaluate_model] Accuracy: {acc:.4f}\n{report}")
    with open("/opt/airflow/working_data/evaluation_results.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n{report}")
    return f"Pipeline complete. Accuracy: {acc:.4f}"

dag = DAG(
    "diabetes_prediction_dag",
    default_args=default_args,
    description="Decision Tree pipeline to predict diabetes",
    schedule_interval=None,
    catchup=False,
)

load_data_task = PythonOperator(task_id="load_data_task", python_callable=load_data, dag=dag)
preprocess_data_task = PythonOperator(task_id="preprocess_data_task", python_callable=preprocess_data, op_args=[load_data_task.output], dag=dag)
train_model_task = PythonOperator(task_id="train_model_task", python_callable=train_model, op_args=[preprocess_data_task.output, "/opt/airflow/working_data/diabetes_dt_model.pkl"], dag=dag)
evaluate_model_task = PythonOperator(task_id="evaluate_model_task", python_callable=evaluate_model, op_args=[train_model_task.output], dag=dag)

load_data_task >> preprocess_data_task >> train_model_task >> evaluate_model_task