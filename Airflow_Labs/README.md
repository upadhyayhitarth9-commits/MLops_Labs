# Airflow Lab — Diabetes Prediction Pipeline
 
An Apache Airflow DAG that runs a **Decision Tree Classifier** pipeline to predict diabetes using the **Pima Indians Diabetes dataset**. The pipeline is orchestrated via Airflow running inside Docker.
 
---
 
## Pipeline Overview
 
```
load_data_task → preprocess_data_task → train_model_task → evaluate_model_task
```
 
| Task | What it does |
|------|-------------|
| `load_data_task` | Reads `diabetes.csv`, serializes and passes it to the next task via XCom |
| `preprocess_data_task` | Replaces invalid zeros with median, scales features, splits 80/20 |
| `train_model_task` | Trains a Decision Tree (max_depth=5), saves model as `.pkl` |
| `evaluate_model_task` | Loads model, runs predictions, prints accuracy + classification report |
 
---
 
## Project Structure
 
```
Airflow_Labs/
├── dags/
│   └── diabetes_dag.py          # Airflow DAG + all ML functions
├── working_data/
│   └── diabetes.csv             # Auto-downloaded by download_data.py
├── logs/                        # Airflow task logs (auto-generated)
├── plugins/                     # Empty — reserved for custom Airflow plugins
├── docker-compose.yaml          # Airflow + Postgres Docker setup
├── .env                         # AIRFLOW_UID config
├── download_data.py             # Script to download dataset
├── requirements.txt             # Python dependencies (for reference)
└── README.md
```
 
---
 
## Prerequisites
 
- **Docker Desktop** installed and running
  - Allocate at least **4GB RAM** → Docker Desktop → Settings → Resources
- **Python 3.8+** (just for running `download_data.py`)
- **VS Code**
 
---
 
## Step-by-Step Setup
 
### Step 1 — Download the dataset
```bash
python download_data.py
```
This saves `diabetes.csv` into `working_data/` automatically.
 
### Step 2 — Set your Airflow UID
```bash
echo -e "AIRFLOW_UID=$(id -u)" > .env
```
 
### Step 3 — Initialize the Airflow database
```bash
docker compose up airflow-init
```
Wait until you see `exited with code 0`.
 
### Step 4 — Start Airflow
```bash
docker compose up -d
```
 
### Step 5 — Install scikit-learn in both containers
```bash
docker exec -it --user airflow airflow_labs-airflow-webserver-1 python -m pip install scikit-learn pandas
docker exec -it --user airflow airflow_labs-airflow-scheduler-1 python -m pip install scikit-learn pandas
```
 
### Step 6 — Create admin user
```bash
docker exec -it airflow_labs-airflow-webserver-1 airflow users create \
  --username admin --password admin123 \
  --firstname Hitarth --lastname Upadhyay \
  --role Admin --email hitarth@example.com
```
 
### Step 7 — Open Airflow UI
Go to: http://localhost:8080
 
Login:
- **Username:** `admin`
- **Password:** `admin123`
 
### Step 8 — Trigger the DAG
1. Click **DAGs** in the top menu
2. Find `diabetes_prediction_dag`
3. Toggle it **ON** (blue switch)
4. Click the ▶️ **Trigger DAG** button
5. Click on the DAG name to watch all 4 tasks turn **green**
 
### Step 9 — Check outputs
After all tasks complete, check `working_data/`:
```
working_data/
├── diabetes.csv                  # Input dataset
├── scaler.pkl                    # Saved StandardScaler
├── diabetes_dt_model.pkl         # Trained Decision Tree model
└── evaluation_results.txt        # Accuracy + classification report
```
 
### Step 10 — Stop Airflow
```bash
docker compose down
```
 
---
 
## Model Details
 
| Parameter | Value |
|-----------|-------|
| Algorithm | Decision Tree Classifier |
| Criterion | Gini Impurity |
| Max Depth | 5 |
| Min Samples Split | 10 |
| Test Split | 20% |
| Random State | 42 |
 
---
 
## Key Airflow Concepts Used
 
| Concept | Where used |
|---------|-----------|
| **DAG** | `diabetes_prediction_dag` — defines the full pipeline |
| **PythonOperator** | Each task calls a Python function |
| **XCom** | Passes serialized data (pickle) between tasks |
| **Task Dependencies** | `>>` operator chains tasks in order |
| **schedule_interval** | Set to `None` (manual trigger) |
 
---
 
## References
 
- [Apache Airflow Docs](https://airflow.apache.org/docs/)
- [Running Airflow in Docker](https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html)
- [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- [Scikit-learn Decision Tree](https://scikit-learn.org/stable/modules/tree.html)
 