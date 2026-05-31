# 🚀 BART Text Summarization with MLflow + FastAPI

This project demonstrates an end-to-end NLP pipeline using Hugging Face Transformers for **text summarization**, integrated with:

* ✅ **MLflow** → Experiment tracking + model registry
* ✅ **FastAPI** → Model serving (API layer)

---

## 📌 Project Overview

* Model: `facebook/bart-large-cnn`
* Task: Text Summarization
* Frameworks:

  * Hugging Face Transformers
  * PyTorch
  * MLflow
  * FastAPI

---

## 📁 Project Structure

```
.
├── main.py              # Train + evaluate + log to MLflow
├── train.py             # Training logic
├── evaluate.py          # Evaluation (ROUGE)
├── app.py               # FastAPI app (serving)
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

---

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📊 Step 1: Run MLflow UI

```bash
mlflow ui
```

Open:

```
http://127.0.0.1:5000
```

---

## ▶️ Step 2: Train & Log Model

```bash
python main.py
```

This will:

* Train model
* Evaluate (ROUGE)
* Log model + metrics to MLflow

---

## 📦 Step 3: Serve Model using MLflow

After training, get your **run_id** from MLflow UI.

```bash
mlflow models serve -m runs:/<RUN_ID>/model -p 1234 --no-conda
```

Test API:

```bash
curl -X POST http://127.0.0.1:1234/invocations \
-H "Content-Type: application/json" \
-d '{
  "inputs": "Your long article text here..."
}'
```

---

## ⚡ Step 4: Serve Model using FastAPI

### ▶️ Run FastAPI App

```bash
uvicorn app:app --reload
```

App runs at:

```
http://127.0.0.1:8000
```

---

### 🔹 Example FastAPI Endpoint

```python
from fastapi import FastAPI
import mlflow.pyfunc

app = FastAPI()

model = mlflow.pyfunc.load_model("runs:/<RUN_ID>/model")

@app.post("/summarize")
def summarize(text: str):
    result = model.predict([text])
    return {"summary": result[0]}
```

---

### 🔹 Test FastAPI

```bash
curl -X POST "http://127.0.0.1:8000/summarize" \
-H "Content-Type: application/json" \
-d '"Your long article text here..."'
```

---

## 📊 MLflow Tracking

### Parameters

* Model name
* Training configs

### Metrics

* ROUGE-1
* ROUGE-2
* ROUGE-L

### Artifacts

* Model
* Tokenizer

---

## 🔁 Workflow

```
Train → Evaluate → Log (MLflow)
                      ↓
              Load Model (MLflow)
                      ↓
          Serve via FastAPI / MLflow API
```

---

## ❌ Docker

This project does **NOT** use Docker.

---

## ✅ Future Improvements

* Add Docker for deployment
* CI/CD pipeline
* Model versioning (MLflow Registry)
* Deploy to cloud (AWS/GCP)

---

## 🏁 Conclusion

This project now covers:

* ✅ Model Training
* ✅ Experiment Tracking (MLflow)
* ✅ Model Serving (MLflow API)
* ✅ Production API (FastAPI)

A complete **mini MLOps pipeline** 🚀

---
