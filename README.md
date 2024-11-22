![Build Status](https://github.com/monimoyd/ml_training_ci_cd_pipeline_advanced/actions/workflows/ml-pipeline.yml/badge.svg)


Steps to run locally:
1. Create a virtual environment:
Bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
2. Install dependencies:
Bash
pip install -r requirements.txt

3. Train the model:
Bash
python src/train.py

4.Run tests:
Bash
pytest tests/test_model.py -v

5. To deploy to GitHub:
i. Create a new repository on GitHub
ii.Initialize local git repository:
Bash
git init
git add .
git commit -m "Initial commit"
git remote add origin <your-repo-url>
git push -u origin main

The GitHub Actions workflow will automatically:
i. Set up a Python environment
ii.Install dependencies
iii.Train the model
iv. Run all tests

v.Save the trained model as an artifact

The tests check for:
i. Model parameter count (< 25000)
ii.Input shape compatibility (28x28)
iii.Model accuracy (> 95%)


