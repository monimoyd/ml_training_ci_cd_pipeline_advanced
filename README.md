![Build Status](https://github.com/monimoyd/ml_training_ci_cd_pipeline_advanced/actions/workflows/ml-pipeline.yml/badge.svg)

# Steps to run locally:
## 1. Create a virtual environment:
Bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
## 2. Install dependencies:
bash
pip install -r requirements.txt

## 3. Train the model:

bash
python src/train.py

## 4.Run tests:

bash
pytest tests/test_model.py -v

## 5. To deploy to GitHub:

* Create a new repository on GitHub

* Initialize local git repository:

bash
git init
git add .
git commit -m "Initial commit"
git remote add origin <your-repo-url>
git push -u origin main

### The GitHub Actions workflow will automatically:

* Set up a Python environment

* Install dependencies

* Train the model

* Run all tests

* Save the trained model as an artifact

## 6. The tests check for:

i. Model parameter count (< 25000)

ii.Input shape compatibility (28x28)

iii.Model accuracy (> 95%)

iv. Test rotation invariance of model

v. Test the size and shape of MNIST dataset

vi. Test if classes are balanced in the MNIST dataset

vii. Test data integrity and statistical properties

6. 10 Augmented Images from MNIST by adding Noise, Rotated, Change brightness
![Images](/all_digit_augmentation.png)
