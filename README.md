# House Price Prediction ML Application

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![MLflow](https://img.shields.io/badge/MLflow-tracking-blue.svg)](https://mlflow.org/)
[![Prefect](https://img.shields.io/badge/Prefect-orchestration-blue.svg)](https://www.prefect.io/)

> A production-ready machine learning application showcasing end-to-end ML engineering skills: from data exploration and model training with MLflow experiment tracking and Prefect workflow orchestration, to containerized deployment with Docker Swarm and Nginx load balancing.

## 🎯 Project Overview

This project demonstrates a **complete production ML pipeline** for predicting house prices using the King County housing dataset. It showcases industry best practices in:

- **ML Engineering**: Feature engineering, model versioning, experiment tracking
- **Software Engineering**: Type-safe modular design, separation of concerns
- **DevOps/MLOps**: Containerization, orchestration, scalable deployment, load balancing

**Live Demo**: [Add your deployed link if available]

## ✨ Key Technical Highlights

### Machine Learning & MLOps
- ✅ **Prefect Orchestration**: Automated workflow management and scheduling
- ✅ **MLflow Integration**: Complete experiment tracking and model registry
- ✅ **Feature Engineering**: Demographic data enrichment for improved predictions
- ✅ **Model Versioning**: Serialized models with tracked feature schemas
- ✅ **Reproducibility**: Versioned experiments with artifact storage
- ✅ **Model Registry**: Production-ready model versioning and staging
- ✅ **Pipeline Automation**: Scheduled retraining and deployment workflows

### Production Engineering
- ✅ **Microservices Architecture**: Separate model and application services
- ✅ **Container Orchestration**: Docker Swarm for high availability
- ✅ **Load Balancing**: Nginx reverse proxy for traffic distribution
- ✅ **Horizontal Scaling**: Dynamic service scaling based on demand
- ✅ **Health Monitoring**: Service health checks and auto-recovery

### Software Engineering
- ✅ **Type Safety**: Custom type definitions and type hints throughout
- ✅ **Modular Design**: Clear separation of concerns (model, UI, utils, types)
- ✅ **Environment Management**: Both pip and conda specifications
- ✅ **Code Organization**: Clean project structure following best practices

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────┐
│                     Client Layer                      │
└───────────────────────┬──────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────┐
│              Nginx Load Balancer                      │
│          (Port 80 - Traffic Distribution)             │
└───────────────────────┬──────────────────────────────┘
                        │
        ┌───────────────┴───────────────┐
        ▼                               ▼
┌─────────────────┐           ┌─────────────────┐
│  App Service 1  │           │  App Service N  │
│  ┌───────────┐  │           │  ┌───────────┐  │
│  │  UI Layer │  │    ...    │  │  UI Layer │  │
│  └─────┬─────┘  │           │  └─────┬─────┘  │
│        │        │           │        │        │
│  ┌─────▼─────┐  │           │  ┌─────▼─────┐  │
│  │   Model   │  │           │  │   Model   │  │
│  │  Service  │  │           │  │  Service  │  │
│  └───────────┘  │           │  └───────────┘  │
└─────────────────┘           └─────────────────┘
         │                             │
         └──────────────┬──────────────┘
                        ▼
        ┌───────────────────────────────┐
        │      Orchestration Layer       │
        │  ┌─────────────────────────┐  │
        │  │   Prefect Workflows     │  │
        │  │  - Data Ingestion       │  │
        │  │  - Model Training       │  │
        │  │  - Model Deployment     │  │
        │  └───────────┬─────────────┘  │
        │              │                 │
        │  ┌───────────▼─────────────┐  │
        │  │     MLflow Server       │  │
        │  │   (Experiments &        │  │
        │  │    Model Registry)      │  │
        │  └─────────────────────────┘  │
        └───────────────────────────────┘
```

## 📊 Project Structure

```
.
├── 📁 data/                          # Dataset management
│   ├── kc_house_data.csv            # Raw housing data
│   ├── zipcode_demographics.csv     # Demographic enrichment
│   ├── combined.csv                 # Processed training data
│   └── future_unseen_examples.csv   # Validation dataset
│
├── 📁 model/                         # Model artifacts
│   ├── create_model.py              # Training pipeline
│   ├── evaluator.py                 # Model evaluation
│   ├── model.pkl                    # Serialized model
│   └── model_features.json          # Feature schema
│
├── 📁 src/                           # Application code
│   ├── main.py                      # API entry point
│   ├── ui.py                        # Web interface
│   ├── model.py                     # Model serving logic
│   ├── utils.py                     # Helper functions
│   ├── types_.py                    # Type definitions
│   └── constants.py                 # Configuration
│
├── 📁 mlruns/                        # MLflow tracking
│   ├── 0/                           # Experiment runs
│   └── models/                      # Model registry
│
├── 📁 nginx/                         # Load balancer config
│   ├── Dockerfile                   
│   └── nginx.conf                   # Routing & balancing
│
├── 📁 swarm/                         # Production deployment
│   ├── app-compose.yml              # Application stack
│   └── model-compose.yml            # Model service stack
│
├── 📁 demo/                          # Deployment utilities
│   ├── init-swarm.sh                # Swarm initialization
│   ├── init-compose.sh              # Compose setup
│   └── reload.py                    # Hot reload dev tool
│
├── 📓 explore.ipynb                  # Data exploration
├── 🐳 Dockerfile                     # Container definition
├── 🐳 docker-compose.yml             # Local orchestration
├── 📋 requirements.txt               # Python dependencies
└── 📋 conda_environment.yml          # Conda environment
```

## 🚀 Getting Started

### Prerequisites

```bash
# Required
- Docker 20.10+
- Docker Compose 2.0+
- Python 3.12+

# Optional (for local development)
- Conda/Miniconda
- Jupyter Notebook
```

### Quick Start (Docker)

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd house-price-prediction

# 2. Build and run with Docker Compose
docker-compose up --build

# 3. Access the application
# Web UI: http://localhost
# MLflow UI: http://localhost:5000
```

### Local Development Setup

```bash
# Option 1: Using pip
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Option 2: Using conda
conda env create -f conda_environment.yml
conda activate house-prediction

# Train model (if needed)
python model/create_model.py

# Explore data
jupyter notebook explore.ipynb

# Run application locally
python src/main.py
```

## 🐳 Production Deployment

### Docker Swarm (Recommended for Production)

```bash
# 1. Initialize Docker Swarm
docker swarm init

# 2. Deploy using the initialization script
cd demo
./init-swarm.sh

# 3. Or deploy manually
docker stack deploy -c swarm/app-compose.yml -c swarm/model-compose.yml housing-app

# 4. Scale services dynamically
docker service scale housing-app_model=5
docker service scale housing-app_web=3

# 5. Monitor services
docker service ls
docker service logs housing-app_model
```

### Deployment Features

- **Auto-scaling**: Services automatically recover from failures
- **Zero-downtime updates**: Rolling updates without service interruption
- **Load distribution**: Nginx distributes requests across healthy instances
- **Health checks**: Automatic unhealthy instance removal
- **Resource limits**: Configured memory and CPU constraints

## 🧪 MLflow Experiment Tracking & Prefect Orchestration

### MLflow for Experiment Management

This project uses **MLflow** for comprehensive experiment tracking:

```bash
# Start MLflow UI
mlflow ui --backend-store-uri ./mlruns

# View experiments at http://localhost:5000
```

**Tracked Metrics:**
- Model performance (RMSE, MAE, R²)
- Training hyperparameters
- Feature importance
- Model artifacts and versions

**Model Registry:**
- Version control for models
- Stage transitions (Staging → Production)
- Model lineage and metadata

### Prefect for Workflow Orchestration

**Prefect** manages the end-to-end ML pipeline:

```bash
# Start Prefect server
prefect server start

# View Prefect UI at http://localhost:4200

# Run a workflow
python model/create_model.py
```

**Automated Workflows:**
- **Data Ingestion Flow**: Scheduled data updates and preprocessing
- **Training Pipeline**: Automated model retraining with new data
- **Deployment Flow**: Model validation and promotion to production
- **Monitoring Flow**: Performance tracking and alert notifications

**Key Features:**
- Task dependency management
- Automatic retries and error handling
- Scheduled and event-triggered runs
- Distributed execution capabilities
- Real-time monitoring dashboard

## 🎯 Skills Demonstrated

### Machine Learning
- Feature engineering with domain knowledge
- Model selection and hyperparameter tuning
- Cross-validation and performance evaluation
- Handling real-world tabular data

### MLOps
- Experiment tracking with MLflow
- Workflow orchestration with Prefect
- Model versioning and registry management
- Reproducible training pipelines
- Model artifact management
- Automated retraining and deployment

### Software Engineering
- Type-safe Python with type hints
- Modular, maintainable code architecture
- Separation of concerns (MVC pattern)
- Configuration management

### DevOps & Infrastructure
- Containerization with Docker
- Microservices architecture
- Container orchestration (Docker Swarm)
- Load balancing and reverse proxy (Nginx)
- Service scaling and high availability
- Infrastructure as Code

### Data Engineering
- Data preprocessing and cleaning
- Feature engineering pipelines
- Data integration from multiple sources
- Schema management and validation

## 🛠️ Technical Stack

| Category | Technologies |
|----------|-------------|
| **ML/Data** | Scikit-learn, Pandas, NumPy, MLflow |
| **Orchestration** | Prefect |
| **Backend** | Python 3.12, FastAPI/Flask |
| **Frontend** | HTML/CSS/JavaScript |
| **DevOps** | Docker, Docker Compose, Docker Swarm |
| **Infrastructure** | Nginx, Linux |
| **Development** | Jupyter, Git, VS Code |

## 🧪 Testing

```bash
# Run unit tests
python -m pytest tests/

# Test model predictions
python model/evaluator.py

# Load test the API
# (Add your load testing commands)
```

## 🔧 Configuration

Key configuration files:
- `requirements.txt` - Python dependencies
- `conda_environment.yml` - Conda environment
- `docker-compose.yml` - Local development setup
- `swarm/*.yml` - Production deployment configs
- `nginx/nginx.conf` - Load balancer configuration

## 🚧 Roadmap

- [x] Implement A/B testing framework
- [x] Add model monitoring and drift detection
- [ ] Add automated model retraining pipeline
- [ ] Add CI/CD pipeline (GitHub Actions)
- [ ] Create REST API documentation (OpenAPI/Swagger)
- [ ] Implement feature store
- [ ] Add comprehensive unit and integration tests

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- King County housing dataset
- MLflow for experiment tracking capabilities
- Prefect for workflow orchestration
- Docker community for containerization best practices

---

⭐ **Star this repository if you found it helpful!**

*This project was created to demonstrate production ML engineering skills for portfolio purposes.*