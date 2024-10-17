<h1>Liver Cancer Detection Using Machine Learning</h1>


Project Overview

This project focuses on detecting liver cancer using various machine learning algorithms. The dataset contains features relevant to liver health, and the goal is to predict whether a patient has liver cancer or not based on these features.

Algorithms Used:

K-Nearest Neighbors (KNN)
Support Vector Classification (SVC)
Logistic Regression (LR)
Decision Tree (DT)
Gaussian Naive Bayes (GNB)
Random Forest (RF)
Gradient Boosting (GB)
XGBoost (XGB)

Dataset

The dataset used in this project can be downloaded from Kaggle and includes the following features:
Age
BMI
Blood test results (like albumin, bilirubin, etc.)
Other health indicators related to liver function

Make sure to preprocess the dataset correctly (e.g., handle missing values, normalize/scale features) before training the models.

You can install the required packages using the following command:

pip install pandas numpy scikit-learn xgboost matplotlib seaborn


Project Structure

The project consists of the following steps:

Data Loading: The dataset is loaded from a CSV file.
Data Preprocessing: This includes handling missing values, scaling the features, and splitting the dataset into training and testing sets.
Model Training: Several machine learning algorithms are trained on the training data.
Model Evaluation: The models are evaluated using accuracy, confusion matrix, and ROC-AUC score.
Visualization: ROC curves and other visualizations are plotted to compare model performance.

How to Run
1. Clone the Repository: Clone this project repository from GitHub or download the source files.

       git clone https://github.com/Murugavl/liver_cancer_detection.git
       cd liver-cancer-detection
2. Run the Jupyter Notebook: Open the project notebook (Liver_Cancer_Detection.ipynb) in JupyterLab or Jupyter Notebook.

       jupyter notebook Liver_Cancer_Detection.ipynb

3. Model Training and Evaluation:

    After loading the dataset, you'll train multiple machine learning models (Logistic Regression, KNN, Random Forest, etc.).
    Evaluate each model based on metrics like accuracy and AUC (Area Under the Curve).

4. Results:

    The best-performing model in the project was XGBoost, which achieved the highest AUC score.
    Visualize the ROC curves to compare the performance of different algorithms.

Future Work

  * Experiment with feature selection techniques to improve model performance.
  * Tune hyperparameters of the machine learning models for better results.
  * Try different datasets or ensemble methods for potential improvements.

