<h1>Liver Cancer Detection Using Machine Learning</h1>


<h2>Project Overview</h2>
This project focuses on detecting liver cancer using various machine learning algorithms. The dataset contains features relevant to liver health, and the goal is to predict whether a patient has liver cancer or not based on these features.

<h2>Algorithms Used:</h2>

K-Nearest Neighbors (KNN)
Support Vector Classification (SVC)
Logistic Regression (LR)
Decision Tree (DT)
Gaussian Naive Bayes (GNB)
Random Forest (RF)
Gradient Boosting (GB)
XGBoost (XGB)

<h2>Dataset</h2>

The dataset used in this project can be downloaded from Kaggle and includes the following features:
Age
BMI
Blood test results (like albumin, bilirubin, etc.)
Other health indicators related to liver function

Make sure to preprocess the dataset correctly (e.g., handle missing values, normalize/scale features) before training the models.

<h3>You can install the required packages using the following command:</h3>

       pip install pandas numpy scikit-learn xgboost matplotlib seaborn


<h2>Project Structure</h2>

The project consists of the following steps:

<h4>Data Loading:</h4>       The dataset is loaded from a CSV file.
<h4>Data Preprocessing:</h4>        This includes handling missing values, scaling the features, and splitting the dataset into training and testing sets.
<h4>Model Training:</h4>        Several machine learning algorithms are trained on the training data.
<h4>Model Evaluation:</h4>        The models are evaluated using accuracy, confusion matrix, and ROC-AUC score.
<h4>Visualization:</h4>        ROC curves and other visualizations are plotted to compare model performance.

<h2>How to Run</h2>

<h4>1. Clone the Repository: Clone this project repository from GitHub or download the source files.</h4>

       git clone https://github.com/Murugavl/liver_cancer_detection.git
       cd liver-cancer-detection
<h4>2. Run the Jupyter Notebook: Open the project notebook (Liver_Cancer_Detection.ipynb) in JupyterLab or Jupyter Notebook.</h4>

       jupyter notebook Liver_Cancer_Detection.ipynb

<h4>3. Model Training and Evaluation:</h4>

    After loading the dataset, you'll train multiple machine learning models (Logistic Regression, KNN, Random Forest, etc.).
    Evaluate each model based on metrics like accuracy and AUC (Area Under the Curve).

<h4>4. Results:</h4>

    The best-performing model in the project was XGBoost, which achieved the highest AUC score.
    Visualize the ROC curves to compare the performance of different algorithms.

<h2>Future Work</h2>

  * Experiment with feature selection techniques to improve model performance.
  * Tune hyperparameters of the machine learning models for better results.
  * Try different datasets or ensemble methods for potential improvements.

