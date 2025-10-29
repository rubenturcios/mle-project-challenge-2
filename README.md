# House Price Prediction ML Application

A production-ready machine learning application demonstrating end-to-end ML engineering skills, from data exploration and model training to deployment with Docker Swarm and load balancing.

## Overview

This project showcases a complete machine learning workflow for predicting house prices using the King County housing dataset. The application demonstrates best practices in ML engineering, including data preprocessing, model persistence, containerization, orchestration, and serving predictions through a user-friendly web interface.

## Project Structure
```
.
├── Dockerfile
├── conda_environment.yml
├── create_model.py
├── data/
│   ├── combinded.csv
│   ├── future_unseen_examples.csv
│   ├── kc_house_data.csv
│   └── zipcode_demographics.csv
├── demo/
│   ├── init-compose.sh
│   ├── init-swarm.sh
│   └── reload.py
├── docker-compose.yml
├── explore.ipynb
├── model/
│   ├── model.pkl
│   └── model_features.json
├── nginx/
│   ├── Dockerfile
│   └── nginx.conf
├── requirements.txt
├── src/
│   ├── constants.py
│   ├── main.py
│   ├── model.py
│   ├── types_.py
│   ├── ui.py
│   └── utils.py
└── swarm/
    ├── app-compose.yml
    └── model-compose.yml
```

## Key Features

### Machine Learning
- **Data Integration**: Combines housing data with demographic information for enhanced predictions
- **Feature Engineering**: Incorporates zipcode demographics for improved model accuracy
- **Model Persistence**: Serialized model with tracked feature schemas for reproducibility
- **Future Predictions**: Capability to score new, unseen examples

### Engineering & DevOps
- **Containerization**: Fully Dockerized application for consistent deployment
- **Orchestration**: Docker Compose and Docker Swarm configurations
- **Load Balancing**: Nginx reverse proxy for production-grade traffic management
- **Scalability**: Swarm deployment enables horizontal scaling of model serving
- **Environment Management**: Both pip and conda environment specifications

### Application Architecture
- **Modular Design**: Separated concerns (model, UI, utilities, types)
- **Type Safety**: Custom type definitions for better code reliability
- **Interactive UI**: User-friendly interface for making predictions
- **API Layer**: RESTful model serving capability

## Technical Stack

- **ML Framework**: scikit-learn (or similar - based on .pkl format)
- **Web Framework**: Python-based web server
- **Containerization**: Docker & Docker Compose
- **Orchestration**: Docker Swarm
- **Load Balancer**: Nginx
- **Data Processing**: Pandas, NumPy
- **Environment**: Python 3.12

## Getting Started

### Prerequisites

- Docker and Docker Compose installed
- Python 3.12+ (for local development)
- 4GB+ RAM recommended

### Local Development

1. **Clone the repository**
```bash
   git clone <repository-url>
   cd <project-directory>
```

2. **Set up Python environment**
```bash
   # Using pip
   pip install -r requirements.txt
   
   # Or using conda
   conda env create -f conda_environment.yml
   conda activate <environment-name>
```

3. **Train the model** (optional - pre-trained model included)
```bash
   python create_model.py
```

4. **Explore the data**
```bash
   jupyter notebook explore.ipynb
```

### Docker Deployment

#### Single Machine (Docker Compose)
```bash
docker-compose up --build
```

#### Production Cluster (Docker Swarm)

1. **Initialize Swarm**
```bash
   cd demo
   ./init-swarm.sh
```

2. **Deploy the stack**
```bash
   docker stack deploy -c swarm/app-compose.yml -c swarm/model-compose.yml housing-prediction
```

3. **Scale the services**
```bash
   docker service scale housing-prediction_model=3
```

## Usage

Once deployed, access the application:

- **Web UI**: http://localhost (or your server IP)
- **API Endpoint**: http://localhost/predict (POST requests)

### Making Predictions

The application accepts house features including:
- Square footage
- Number of bedrooms/bathrooms
- Location (zipcode)
- Additional demographic features
- And more...

## Project Highlights

This project demonstrates:

✅ **End-to-End ML Pipeline**: From raw data to deployed model  
✅ **Production-Ready Architecture**: Load balancing, container orchestration  
✅ **Scalable Design**: Horizontal scaling with Docker Swarm  
✅ **Code Quality**: Type hints, modular structure, separation of concerns  
✅ **Data Engineering**: Feature engineering with demographic enrichment  
✅ **DevOps Practices**: Containerization, orchestration, automated deployment  
✅ **Documentation**: Clear structure and reproducible workflows  

## Data

The project uses:
- **kc_house_data.csv**: King County house sales data
- **zipcode_demographics.csv**: Demographic data by zipcode
- **combined.csv**: Merged dataset used for training
- **future_unseen_examples.csv**: Test cases for model validation

## Model Performance

[Add your model metrics here]
- Training accuracy: [metric]
- Validation accuracy: [metric]
- Key features: [top features]

## Architecture Diagram
```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│    Nginx    │  (Load Balancer)
└──────┬──────┘
       │
       ▼
┌─────────────────────────┐
│   Application Service   │
│  ┌─────────────────┐   │
│  │   UI (ui.py)    │   │
│  └────────┬────────┘   │
│           │             │
│  ┌────────▼────────┐   │
│  │  Model Service  │   │
│  │   (model.py)    │   │
│  └─────────────────┘   │
└─────────────────────────┘
```

## Development Scripts

- **create_model.py**: Train and serialize the ML model
- **explore.ipynb**: Jupyter notebook for data exploration and analysis
- **demo/reload.py**: Hot reload utility for development
- **demo/init-compose.sh**: Initialize Docker Compose demo
- **demo/init-swarm.sh**: Initialize Docker Swarm cluster

## Contributing

This is a portfolio project, but suggestions and feedback are welcome!