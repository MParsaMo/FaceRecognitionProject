# Face Recognition with Eigenfaces & Model Comparison (Olivetti Faces)
This project demonstrates a complete machine learning pipeline for face recognition using the Olivetti Faces dataset, PCA-based dimensionality reduction (Eigenfaces), and the evaluation of multiple classifiers using GridSearchCV.

📌 Features
✅ Load and explore the Olivetti Faces dataset
✅ Visualize representative faces and full sets for a given person
✅ Apply PCA for dimensionality reduction and whitening
✅ Train and evaluate multiple models using GridSearchCV
✅ Compare models using accuracy and cross-validation scores
✅ Visualize results with comparative bar charts

🗂️ Project Structure
.
├── face_recognition_olivetti.py      # Main project script
├── README.md                         # This file
└── requirements.txt                  # (optional) Python dependencies

🧪 Models Used
The following models are trained and evaluated:

Logistic Regression

Support Vector Classifier (SVC)

Gaussian Naive Bayes

Each model is fine-tuned using GridSearchCV with appropriate hyperparameter grids.

📊 Dataset
Name: Olivetti Faces

Source: Built-in Scikit-learn dataset

Size: 400 grayscale images (64x64), 40 individuals, 10 images per person

Labels: Person IDs (0 to 39)

Normalized: All pixel values are scaled between 0 and 1

📈 PCA - Eigenfaces
Dimensionality is reduced using Principal Component Analysis (PCA).

PCA is applied with:

n_components = 0.99 → 99% variance retained

whiten=True → decorrelates features and standardizes variance

This process generates Eigenfaces, which are linearly uncorrelated features best suited for facial classification tasks.

📊 Model Evaluation
For each classifier:

Best hyperparameters are found using GridSearchCV (5-fold cross-validation)

Final performance is evaluated on a held-out test set

Evaluation metrics include:

Accuracy on the test set

Best cross-validation accuracy

Visual output:

Side-by-side bar charts of model performance

Example output:
Model Comparison Results:
         Model                          Best Params  Best CV Accuracy  Test Accuracy
0  LogisticRegression  {'C': 1, 'solver': 'liblinear'}           0.9250          0.9150
1                 SVC   {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}           0.9300          0.9200
2          GaussianNB     {'var_smoothing': 1e-09}           0.9050          0.9000

🖼️ Visualizations
Sample Faces Grid: One face per person from the dataset

All Images for Person 0: A strip of images showing all 10 variations

Bar Charts: Compare test accuracy and CV accuracy across models

🚀 How to Run
pip install -r requirements.txt  # Install dependencies
python FaceRecognitionProject.py

📚 Key Concepts Covered
Principal Component Analysis (PCA) for image compression

Eigenfaces method for face recognition

GridSearchCV for model selection

Comparison of classification algorithms

Visual data exploration

📌 Notes
The dataset is already normalized, so no need for additional scaling

PCA must be fit on training data only, then applied to test data

Stratified splits ensure balanced classes during train/test split

random_state values are used for reproducibility

🤓 Educational Value
This project is perfect for:

Learning face recognition basics

Practicing PCA on image datasets

Comparing classifiers using systematic evaluation

Understanding the importance of dimensionality reduction in high-dimensional data

🧑‍💻 Author
Mohamed Parsa Moazam

If you found this useful or educational, feel free to fork, share, or improve it.
