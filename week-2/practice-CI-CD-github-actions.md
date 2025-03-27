# **CI/CD and GitHub Actions: Automating Workflows in Software Development**

## **Session Overview**
**Duration:** 90 minutes  
**Prerequisites:** Familiarity with Git, GitHub pull requests, and Python.  
**Goals:**  
1. Learn how to set up SSH for GitHub authentication.
2. Understand the basics of CI/CD and GitHub Actions.
3. Implement automated testing and deployment with GitHub Actions.
4. Work on four hands-on examples using Python (including a more advanced machine learning pipeline and Docker-based deployment).

---

## **1. Setting Up SSH for GitHub**
### **Why Use SSH?**
Previously, you used HTTPS to interact with GitHub. SSH provides a more secure and convenient way to authenticate without repeatedly entering your credentials.

### **Step 1: Check for Existing SSH Keys**
Open a terminal and check if you already have an SSH key:
```bash
ls -al ~/.ssh
```
If you see files like `id_rsa.pub` or `id_ed25519.pub`, you already have an SSH key.

### **Step 2: Generate a New SSH Key (If Needed)**
If you don't have an SSH key, generate one using:
```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```
Press Enter to accept the default file location and optionally add a passphrase.

### **Step 3: Add the SSH Key to the SSH Agent**
```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

### **Step 4: Add the SSH Key to Your GitHub Account**
Copy your SSH public key:
```bash
cat ~/.ssh/id_ed25519.pub
```
Go to GitHub -> **Settings** -> **SSH and GPG keys** -> **New SSH key**, paste your key, and save.

### **Step 5: Test the Connection**
```bash
ssh -T git@github.com
```
If successful, you'll see a message like:
```
Hi username! You've successfully authenticated.
```
Now, you can use SSH for Git operations.

---

## **2. Introduction to CI/CD and GitHub Actions**
### **What is CI/CD?**
- **Continuous Integration (CI):** Automates testing to ensure code changes work before merging.
- **Continuous Deployment (CD):** Deploys tested code automatically to production.
- **GitHub Actions:** A tool that enables CI/CD workflows directly in GitHub repositories.

### **How GitHub Actions Work**
A GitHub Action workflow consists of:
1. **Triggers** (e.g., `push`, `pull_request`)
2. **Jobs** (tasks to run in the workflow)
3. **Steps** (individual commands)
4. **Runners** (environments that execute the tasks)

---

## **3. Hands-on Example 1: Running Python Tests with GitHub Actions**
### **Project Setup**
Create a GitHub repository and clone it locally using SSH:
```bash
git clone git@github.com:your-username/ci-cd-demo.git
cd ci-cd-demo
```

Create a Python project with a simple function and a test:
```bash
mkdir project && cd project
touch main.py test_main.py
```

**`main.py` (Simple Math Function)**
```python
def add(a, b):
    return a + b
```

**`test_main.py` (Unit Test using `pytest`)**
```python
import pytest
from main import add

def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0
```

### **Setting Up GitHub Actions Workflow**
Create a `.github/workflows/test.yml` file:
```yaml
name: Run Python Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.x'
      - name: Install dependencies
        run: |
          pip install pytest
      - name: Run tests
        run: pytest test_main.py
```
Commit and push your code to GitHub. Check **Actions** tab in your repository to see the workflow running.

---

## **4. Hands-on Example 2: Automating Linear Regression Model Testing**
### **Project Setup**
Create a new directory `linear_regression` with:
```bash
mkdir linear_regression && cd linear_regression
touch model.py test_model.py
```

**`model.py` (Linear Regression with `sklearn`)**
```python
import numpy as np
from sklearn.linear_model import LinearRegression

def train_model():
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 6, 8, 10])
    model = LinearRegression().fit(X, y)
    return model.coef_[0], model.intercept_
```

**`test_model.py` (Unit Test)**
```python
from model import train_model

def test_train_model():
    coef, intercept = train_model()
    assert round(coef, 2) == 2.0  # Expected slope = 2
    assert round(intercept, 2) == 0.0  # Expected intercept = 0
```

### **Setting Up GitHub Actions Workflow for ML Model**
Create `.github/workflows/ml-test.yml`:
```yaml
name: Test Linear Regression Model

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.x'
      - name: Install dependencies
        run: |
          pip install numpy scikit-learn
      - name: Run ML Model Test
        run: pytest test_model.py
```

---

## **5. Hands-on Example 3: Deploying a Flask API with Docker and GitHub Actions**
Create a simple Flask API:
```python
from flask import Flask
app = Flask(__name__)
@app.route('/')
def home():
    return "Hello, CI/CD!"
if __name__ == '__main__':
    app.run(host='0.0.0.0')
```
Create a `Dockerfile`:
```dockerfile
FROM python:3.8
WORKDIR /app
COPY . /app
RUN pip install flask
CMD ["python", "app.py"]
```
Create a GitHub Action for building and pushing the Docker image to Docker Hub.

---

## **6. Hands-on Example 4: Automating Model Training and Deployment**
Build an advanced ML pipeline that retrains a model weekly, stores results, and deploys it to a cloud service.

---

## **7. Conclusion**
By the end of this session, you should be able to automate tests, build Docker images, and deploy machine learning models with GitHub Actions.

Happy coding!

