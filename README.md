# 🩺 End-to-End Kidney Disease Classification (CV & MLOps Pipeline)

Welcome to the **End-to-End Kidney Disease Classification** project! 

This repository contains a production-ready Machine Learning pipeline designed to automatically classify Kidney CT scans to detect the presence of tumors. It leverages state-of-the-art Computer Vision models (CNNs) and industry-standard MLOps practices for orchestration, experiment tracking, containerization, and automated cloud deployment.

---

## 🚀 Project Overview

The goal of this project is to take raw medical imaging data (CT Scans) and accurately predict whether the scan is Normal or exhibits signs of a Tumor.

*   **Dataset:** Kidney CT Scan image dataset, automatically ingested from a remote repository.
*   **Base Model:** Transfer Learning using pre-trained **VGG16**, heavily customized for binary classification.
*   **Interface:** A Web Application / REST API built with **Flask** to trigger model training and serve image predictions.
*   **Deployment:** Automated via **GitHub Actions** to an **AWS EC2** instance using **Docker** and AWS ECR.

---

## 🛠️ Tech Stack & Tools

*   **Deep Learning & Computer Vision:** TensorFlow 2, Keras, NumPy, Pandas
*   **Backend & API:** Python 3.10, Flask, Flask-Cors
*   **MLOps & Orchestration:** DVC (Data Version Control) for pipeline management
*   **Experiment Tracking:** MLflow integrated with DagsHub
*   **Containerization & Deployment:** Docker, AWS ECR, AWS EC2
*   **CI/CD:** GitHub Actions (Automated integration and delivery)

---

## 🧠 Machine Learning Pipeline Architecture

The core of this project is a highly modular, 4-stage Machine Learning pipeline. Each stage is strictly decoupled, ensuring maintainability, reusability, and reproducibility.

1.  **Stage 1: Data Ingestion (`stage_01_data_ingestion.py`)**
    *   Automatically downloads the raw compressed Kidney CT Scan dataset from the designated source URL.
    *   Extracts and stores the data locally in the `artifacts/data_ingestion` directory.
2.  **Stage 2: Prepare Base Model (`stage_02_prepare_base_model.py`)**
    *   Downloads the pre-trained `VGG16` architecture with `imagenet` weights.
    *   Freezes the base model layers to preserve pre-trained feature extraction capabilities.
    *   Appends a custom dense head optimized for our 2-class categorization (`Normal` vs `Tumor`).
3.  **Stage 3: Model Training (`stage_03_model_training.py`)**
    *   Utilizes TensorFlow `ImageDataGenerator` for dynamic data loading and image augmentation (controlled via `params.yaml`).
    *   Trains the newly added dense layers on the CT scan data and saves the updated `.h5` model to `artifacts/training`.
4.  **Stage 4: Model Evaluation (`stage_04_model_evaluation.py`)**
    *   Evaluates the trained model against unseen validation data.
    *   Calculates `Loss` and `Accuracy` metrics, saving them to `scores.json`.
    *   Automatically logs parameters and metrics to remote **MLflow (DagsHub)** servers and registers the `VGG16Model`.

---

## ⚙️ MLOps Orchestration (DVC & MLflow)

### DVC (Data Version Control)
Running an entire Deep Learning pipeline every time a small change is made is computationally expensive. This project utilizes **DVC** to orchestrate the workflow:

- **Pipeline Tracking:** `dvc.yaml` defines the inputs, dependencies, parameters, and outputs of every stage.
- **Smart Execution:** If you run `dvc repro`, DVC automatically detects which stages have changed (e.g., tweaking `EPOCHS` in `params.yaml`). It will *only* re-run the necessary stages (e.g., Training and Evaluation), entirely skipping Data Ingestion to save time and compute.
- **DAG Visualization:** Run `dvc dag` in the terminal to visually inspect the pipeline's dependency graph.

### MLflow & DagsHub
Experiment tracking is handled securely in the cloud:
- Every run is tracked via **MLflow**, logging vital hyperparameters (Epochs, Batch Size, Learning Rate) and resulting metrics (Accuracy, Loss).
- **DagsHub** acts as the remote backend, providing a collaborative GUI to visualize model improvements over time.

---

## 💻 How to Run Locally

Follow these steps to set up the project on your local machine.

### Step 1: Clone the Repository
```bash
git clone https://github.com/karim-nadim/Kidney-Disease-Classification.git
cd Kidney-Disease-Classification
```

### Step 2: Create a Virtual Environment (Conda)
```bash
conda create -n kidney python=3.10 -y
conda activate kidney
```

### Step 3: Install GPU Dependencies (Optional, for Windows Native)
```bash
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
```

### Step 4: Install Requirements
```bash
pip install -r requirements.txt
```

### Step 5: Run the Application
```bash
python app.py
```
*The Flask server will start on port `8080`. Open your browser and navigate to `http://localhost:8080` to access the web interface.*

---

## 🌐 Web Interface & API Endpoints

Once the application is running, you can interact with the system via the following endpoints:

*   `GET /`: Serves the main HTML Web UI (`index.html`).
*   `GET/POST /train`: Triggers the end-to-end `main.py` pipeline (or `dvc repro`). It executes Ingestion, Base Model Preparation, Training, and Evaluation sequentially.
*   `POST /predict`: Accepts a Base64 encoded image string, passes it through the trained CNN, and returns a JSON response classifying the image as either **'Tumor'** or **'Normal'**.

---

## ☁️ CI/CD & AWS Cloud Deployment

This project features a fully automated Continuous Integration and Continuous Deployment (CI/CD) pipeline built with **GitHub Actions**. Pushing to the main branch triggers a workflow that automatically deploys the latest model to AWS.

### CI/CD Workflow Overview:
1.  **Continuous Integration:** Lints the code, installs dependencies, and ensures stability.
2.  **Continuous Delivery (ECR):** Securely authenticates with AWS, builds a Docker Image of the application, and pushes it to an Amazon Elastic Container Registry (ECR).
3.  **Continuous Deployment (EC2):** A self-hosted runner on an Ubuntu EC2 instance pulls the latest Docker image from ECR, terminates the old container, and spins up the new container serving the Flask app.

### Setting up AWS for this project:
If you are forking this repo and want to replicate the AWS deployment:
1.  Create an IAM User with `AmazonEC2FullAccess` and `AmazonEC2ContainerRegistryFullAccess`.
2.  Create an ECR Repository to hold your Docker image (save the repository URI).
3.  Launch an EC2 Ubuntu Instance, install Docker, and configure it as a GitHub Self-Hosted Runner (via GitHub repository Settings > Actions > Runners).
4.  Add the following GitHub Repository Secrets:
    *   `AWS_ACCESS_KEY_ID`
    *   `AWS_SECRET_ACCESS_KEY`
    *   `AWS_REGION` (e.g., `us-east-1`)
    *   `AWS_ECR_LOGIN_URI` (e.g., `566373416292.dkr.ecr.us-east-1.amazonaws.com`)
    *   `ECR_REPOSITORY_NAME`

---

## 📂 Project Structure

```text
Kidney-Disease-Classification/
│
├── .github/workflows/          # CI/CD Pipeline configuration for GitHub Actions
├── src/cnnClassifier/
│   ├── components/             # Core Pipeline stage definitions (Ingestion, Model prep, etc.)
│   ├── config/                 # Configuration manager mapping to yaml files
│   ├── pipeline/               # Scripts to execute individual stages
│   ├── entity/                 # Custom data structures for configurations
│   ├── utils/                  # Helper functions (e.g., read_yaml, decodeImage)
│   └── constants/              # System-wide file paths and constants
│
├── config/config.yaml          # Directory paths and data source URLs
├── params.yaml                 # Tunable hyperparameters (EPOCHS, BATCH_SIZE, etc.)
├── dvc.yaml                    # DVC orchestration graph
├── app.py                      # Flask web server and endpoints
├── main.py                     # Pipeline execution entry point
├── Dockerfile                  # Container instructions
├── setup.py                    # Package setup script
└── requirements.txt            # Python dependencies
```

---

## 👨‍💻 Author

**Karim Nadim**  
Data Scientist & Machine Learning Engineer  
📧 Email: karim_ossama94@hotmail.com  
🔗 GitHub: karim-nadim
