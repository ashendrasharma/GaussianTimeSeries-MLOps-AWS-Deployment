# Time Series Gaussian Model and AWS Deployment

## Business Objective

A time series is a sequence of data points ordered in time. Typically, time is the independent variable, and the primary goal is to make future forecasts. Time series data has various applications in everyday activities, such as:

- Tracking daily, hourly, or weekly weather data
- Monitoring changes in application performance
- Visualizing real-time vitals in medical devices

Gaussian Processes are a generalization of the Gaussian probability distribution, serving as the foundation for sophisticated non-parametric machine learning algorithms for classification and regression. Gaussian probability distribution functions describe the distribution of random variables, while Gaussian processes capture properties of functions, including their parameters.

Gaussian processes can be employed as a machine learning algorithm for classification predictive modeling.

Deployment involves integrating a machine learning model into an existing production environment for making practical business decisions based on data. MLOps (Machine Learning Operations) is a framework for continuous delivery and deployment of machine learning models. It emphasizes automation and monitoring at all stages of ML system construction, including integration, testing, releasing, deployment, and infrastructure management.

In this project, we aim to create an MLOps project for the time series Gaussian model using Python on the AWS cloud platform (Amazon Web Services) with a focus on cost optimization.

---

## Data Description

The dataset is "Call-centers" data, organized at a monthly level, where calls are categorized by domain as the call center operates for various domains. The dataset also includes external regressors like the number of channels and phone lines, which indicate traffic predictions by in-house analysts and available resources.

The dataset contains 132 rows and 8 columns, including:
- Month
- Healthcare
- Telecom
- Banking
- Technology
- Insurance
- Number of Phone Lines
- Number of Channels

---

### Aim

- Build a Gaussian model using the provided dataset.
- Create an MLOps pipeline using the Amazon Web Services (AWS) platform to deploy the time series Gaussian model in a production environment.

---

## Tech Stack

- **Language:** `Python`
- **Libraries:** `Flask`, `pickle`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`
- **Services:** `Flask`, `AWS`, `Docker`, `Lightsail`, `EC2`

---

## Approach

### Data Preparation and Analysis

1. **Import Libraries and Load the Dataset**
2. **Descriptive Analysis**
3. **Data Pre-processing**
    - Convert dates to numerical format
    - Set date as an index

### Exploratory Data Analysis (EDA)

4. **Exploratory Data Analysis (EDA)**
    - Data Visualization

5. **Check for Normality**
    - Density plots
    - QQ-plots

### Gaussian Process Model

6. **Gaussian Process Modeling**
    - Initialize kernels
    - Perform train-test split
    - Create a Gaussian process regressor model
    - Fit the model
    - Generate predictions
    - Visualize results

7. **Difference Modeling**
    - Create a residual column (difference)
    - Check for normality
    - Perform train-test split
    - Initialize kernel
    - Create a Gaussian model
    - Fit the model
    - Generate predictions on test data
    - Visualize the results

### Model Deployment

8. **Model Creation**
    - Save the model in pickle format (.pkl)

9. **Flask Application**
    - Create a Flask app

10. **EC2 Machine Setup**
    - Create an instance on the AWS Management Console
    - Launch the instance
    - Install the 'Putty' tool for remote access

11. **EC2 and Docker Setup**
    - Follow the instructions in the 'install-docker.sh' file

12. **AWS CLI Installation**
    - Refer to the steps in the 'install-aws-cli.sh' file

13. **Lightsail Installation**
    - Follow the steps in the 'install-lightsail-cli.sh' file

14. **Upload Files to EC2 Machine**
    - **Method 1:**
        - Upload the code file in zip format via AWS Console (Cloud Shell)
    - **Method 2:**
        - Create an S3 storage bucket
        - Copy the object URL and use it on the EC2 machine to download the code
        - Unzip the Bitbucket folder

15. **Deployment**
    - Follow the deployment instructions in 'lightsail-deployment.md'

---

## Project Structure

- **Input:** CallCenterData.xlsx
- **MLPipeline:** Contains functions organized into different Python files
- **Notebook:** IPython notebook for the time series Gaussian model
- **Output:** Gaussian model saved in a pickle format
- **App.py:** Flask app configuration
- **Dockerfile:** Docker image configuration
- **Engine.py:** File that calls functions from MLPipeline
- **install-aws-cli.sh:** Steps for AWS CLI installation
- **install-docker.sh:** Steps for Docker installation
- **install-lightsail-cli.sh:** Steps for Lightsail installation
- **lightsail-deployment.md:** Readme file with Lightsail deployment instructions
- **requirements.txt:** List of essential libraries with their versions

---
