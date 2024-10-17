<h1>Liver Cancer Detection Using Machine Learning</h1>


<h2>Project Overview</h2>
<p> &emsp; This project focuses on detecting liver cancer using various machine learning algorithms. The dataset contains features relevant to liver health, and the goal is to predict whether a patient has liver cancer or not based on these features.</p>

<h2>Algorithms Used:</h2>

<p> &emsp; K-Nearest Neighbors (KNN)</p>
<p> &emsp; Support Vector Classification (SVC)</p>
<p> &emsp; Logistic Regression (LR)</p>
<p> &emsp; Decision Tree (DT)</p>
<p> &emsp; Gaussian Naive Bayes (GNB)</p>
<p> &emsp; Random Forest (RF)</p>
<p> &emsp; Gradient Boosting (GB)</p>
<p> &emsp; XGBoost (XGB)</p>

<h2>Dataset</h2>

The dataset used in this project can be downloaded from Kaggle and includes the following features:
<p> &emsp; * Age</p>
<p> &emsp; * BMI</p>
<p> &emsp; * Blood test results (like albumin, bilirubin, etc.)</p>
<p> &emsp; * Other health indicators related to liver function</p>

Make sure to preprocess the dataset correctly (e.g., handle missing values, normalize/scale features) before training the models.

<h3>You can install the required packages using the following command:</h3>

       pip install pandas numpy scikit-learn xgboost matplotlib seaborn


<h2>Project Structure</h2>

The project consists of the following steps:

<h4>Data Loading:</h4> <p> &emsp; The dataset is loaded from a CSV file.</p>
<h4>Data Preprocessing:</h4> <p> &emsp; This includes handling missing values, scaling the features, and splitting the dataset into training and testing sets.</p>
<h4>Model Training:</h4> <p> &emsp; Several machine learning algorithms are trained on the training data.</p>
<h4>Model Evaluation:</h4> <p> &emsp; The models are evaluated using accuracy, confusion matrix, and ROC-AUC score.</p>
<h4>Visualization:</h4> <p> &emsp; ROC curves and other visualizations are plotted to compare model performance.</p>

<h2>How to Run</h2>

<h4>1. Clone the Repository: Clone this project repository from GitHub or download the source files.</h4>

       git clone https://github.com/Murugavl/liver_cancer_detection.git
       cd liver-cancer-detection
<h4>2. Run the Jupyter Notebook: Open the project notebook (Liver_Cancer_Detection.ipynb) in JupyterLab or Jupyter Notebook.</h4>

       jupyter notebook Liver_Cancer_Detection.ipynb

<h4>3. Model Training and Evaluation:</h4>

   <p> &emsp; After loading the dataset, you'll train multiple machine learning models (Logistic Regression, KNN, Random Forest, etc.).
   Evaluate each model based on metrics like accuracy and AUC (Area Under the Curve).</p>

<h4>4. Results:</h4>

   <p> &emsp; The best-performing model in the project was XGBoost, which achieved the highest AUC score.
   Visualize the ROC curves to compare the performance of different algorithms.</p>

<h2>Future Work</h2>

  <p> &emsp; * Experiment with feature selection techniques to improve model performance.</p>
  <p> &emsp; * Tune hyperparameters of the machine learning models for better results.</p>
  <p> &emsp; * Try different datasets or ensemble methods for potential improvements.</p>

