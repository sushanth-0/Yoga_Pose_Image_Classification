<div align="center">
  <h1>Yoga Pose Image Classification</h1>
  <p>
    An end-to-end deep learning application for Yoga Pose Image Classification, deployed to AWS EC2 with Docker and Jenkins.
  </p>
</div>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#tech-stack">Tech Stack</a></li>
      </ul>
    </li>
    <li><a href="#how-to-run">How to Run</a></li>
    <li><a href="#training-pipeline">Training Pipeline</a></li>
    <li><a href="#evaluation">Evaluation</a></li>
    <li><a href="#deployment">Deployment</a></li>
    <li><a href="#workflow">Workflow</a></li>
  </ol>
</details>

## About The Project

I have developed an end-to-end state-of-the-art pipeline for a computer vision project to classify Yoga Poses. I have implemented a deep learning Convolutional Neural Network (CNN).

### Tech Stack

- **Programming Language**: Python
- **Deep Learning Framework**: TensorFlow, Keras
- **Web Framework**: Flask
- **Pre-trained Model**: VGG16
- **Data Management**: gdown, DVC
- **Experiment Tracking**: MLflow
- **Containerization**: Docker
- **CI/CD**: Jenkins
- **Version Control**: GitHub
- **Cloud Services**: 
  - **AWS ECR**: Amazon Elastic Container Registry
  - **AWS EC2**: Amazon Elastic Compute Cloud
  - **AWS ElastiCache**: Amazon ElastiCache

## How to Run

Instructions to set up your local environment for running the project.

```bash
git clone https://github.com/yourusername/yourproject.git

cd yourproject

# Set up a virtual environment
python3 -m venv venv

source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

Instructions on how to use the project.

```bash
# Run the application
python app.py
```

## Training Pipeline

1. **Data Ingestion**: 
   - Ingested the Yoga Pose images from Google Drive using the `gdown` package.

2. **Prepare Base Model**:
   - Customized a pre-trained CNN VGG16 model to train on our dataset.
   - Dropped the dense layer since ImageNet had 1000 classes, added a custom dense layer for our 107 classes dataset.

3. **Model Trainer**:
   - Trained the custom CNN model on the prepared dataset.

4. **Model Evaluation**:
   - Evaluated the model's performance on a test dataset and calculated metrics like accuracy, precision, recall, and F1-score.

## Evaluation

- **MLflow**:
  - Integrated MLflow in the model trainer and evaluator components to track experiments and manage models effectively.

```bash
export MLFLOW_TRACKING_URI= MLFLOW_TRACKING_URI
export MLFLOW_TRACKING_USERNAME= MLFLOW_TRACKING_USERNAME
export MLFLOW_TRACKING_PASSWORD= MLFLOW_TRACKING_PASSWORD
```

- **DVC**:
  - Integrated Data Version Control (DVC) in the data ingestion component to manage large datasets efficiently by providing a Git-like interface for data versioning.

## Deployment

To ensure seamless integration and continuous delivery of the project, I utilized a robust CI/CD pipeline for deploying the DL pipeline to AWS using the following tools:

1. **Jenkins**: 
   - Jenkins is used for continuous integration and delivery, automating the process of building, testing, and publishing code changes to ensure efficiency and reliability.

2. **GitHub**:
   - As a version control system, GitHub provides collaborative features, enabling seamless teamwork and efficient code management.

3. **AWS ECR**:
   - Amazon Elastic Container Registry (ECR) facilitates the secure storage and management of Docker container images, ensuring easy access and scalability.

4. **AWS EC2**:
   - Amazon Elastic Compute Cloud (EC2) serves as the backbone of the project, providing scalable and reliable computing capacity in the cloud. It allows us to deploy and run applications quickly.

### Steps to Deploy

1. **Set Up Jenkins**:
   - Install Jenkins on an AWS EC2 instance or any server of your choice.
   - Configure Jenkins by installing necessary plugins like AWS Credentials Plugin, Pipeline Plugin, and Docker Plugin.

2. **Configure AWS Services**:
   - **AWS EC2**: Launch an EC2 instance, install Docker, and configure it to run your application.
   - **AWS ECR**: Create an ECR repository to store your Docker images.

3. **Create a Jenkins Pipeline**:
   - Create a new pipeline job in Jenkins and define the pipeline script to automate the build, push, and deployment processes.

4. **Automate Deployment**:
   - Configure the Jenkins pipeline to trigger on code changes, build the Docker image, push it to AWS ECR, and deploy it to the AWS EC2 instance.

5. **Set Environment Variables**:
   - Configure necessary environment variables in Jenkins and your application to ensure proper deployment and operation.

### Export the Environment Variables

```bash
export AWS_ACCESS_KEY_ID=<AWS_ACCESS_KEY_ID>
export AWS_SECRET_ACCESS_KEY=<AWS_SECRET_ACCESS_KEY>
```


## Workflow

1. **Data Ingestion Pipeline**: 
    - Downloads and extracts the data from Google Drive.
    - Configuration and execution in `data_ingestion_pipeline.py`.
  
2. **Prepare Base Model Pipeline**: 
    - Downloads and prepares the base VGG16 model.
    - Configuration and execution in `prepare_base_model_pipeline.py`.

3. **Model Training Pipeline**: 
    - Trains the custom model on the dataset.
    - Configuration and execution in `model_training_pipeline.py`.

4. **Model Evaluation Pipeline**: 
    - Evaluates the trained model and logs metrics to MLflow.
    - Configuration and execution in `evaluation_pipeline.py`.

5. **Prediction Pipeline**:
    - Built a **Flask** web application to serve the model predictions.
    - Handles image inputs for classification and returns the prediction results.
