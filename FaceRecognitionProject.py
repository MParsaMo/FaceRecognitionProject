import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # Standard way to import pyplot
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_olivetti_faces_data():
    """
    Loads the Olivetti Faces dataset from scikit-learn.

    This dataset consists of 400 grayscale face images of 40 different people.
    Each person has 10 images, and each image is 64x64 pixels.
    The data is already normalized between 0 and 1.

    Returns:
        sklearn.utils.Bunch: A scikit-learn Bunch object containing data (features),
                             target (person IDs), and image data.
    """
    print("Loading Olivetti Faces dataset...")
    olivetti_data = fetch_olivetti_faces(shuffle=True, random_state=42) # Add random_state for consistent shuffle
    print(f"Dataset loaded: {olivetti_data.data.shape[0]} images, {olivetti_data.data.shape[1]} features.")
    print(f"Number of distinct targets (people): {len(np.unique(olivetti_data.target))}")
    print(f"Image dimensions: {olivetti_data.images[0].shape} (pixels)")
    return olivetti_data

def visualize_dataset_samples(features, targets, num_rows=5, num_cols=8):
    """
    Visualizes a grid of unique face IDs from the dataset.

    Args:
        features (numpy.ndarray): The flattened image data (features).
        targets (numpy.ndarray): The target labels (person IDs).
        num_rows (int): Number of rows for the subplot grid.
        num_cols (int): Number of columns for the subplot grid.
    """
    print('\n--- Visualizing Photos of 40 different people ---')
    fig, subplots = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2))
    subplots = subplots.flatten() # Flatten the 2D array of axes for easy iteration

    # Iterate through unique user IDs and display one image per unique user
    # Assumes unique_user_id * 8 provides an image for that user.
    # A more robust way might be to find the first occurrence of each unique ID.
    for i, unique_user_id in enumerate(np.unique(targets)):
        # Find an index for this unique_user_id
        # np.where returns a tuple of arrays, take the first element (array of indices), then the first index
        image_index = np.where(targets == unique_user_id)[0][0]
        subplots[i].imshow(features[image_index].reshape(64, 64), cmap='gray')
        subplots[i].set_xticks([]) # Hide x-axis ticks
        subplots[i].set_yticks([]) # Hide y-axis ticks
        subplots[i].set_title(f'ID: {unique_user_id}', fontsize=8) # Set title for each subplot

    plt.suptitle('Representative Faces from Olivetti Dataset', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle
    plt.show()

def visualize_person_images(features, person_id=0, num_images=10):
    """
    Visualizes all images for a specific person from the dataset.

    Args:
        features (numpy.ndarray): The flattened image data (features).
        person_id (int): The ID of the person to visualize.
        num_images (int): The number of images to display for that person (should be 10 for Olivetti).
    """
    print(f'\n--- Visualizing {num_images} images of person with ID {person_id} ---')
    fig, subplots = plt.subplots(1, num_images, figsize=(num_images * 2.5, 2.5)) # Adjust figsize
    # Find indices for the given person_id
    person_indices = np.where(olivetti_data.target == person_id)[0][:num_images]

    for i, img_idx in enumerate(person_indices):
        subplots[i].imshow(features[img_idx].reshape(64, 64), cmap='gray')
        subplots[i].set_xticks([])
        subplots[i].set_yticks([])
        subplots[i].set_title(f'Img idx: {img_idx}', fontsize=8)

    plt.suptitle(f'All {num_images} Images for Face ID {person_id}', fontsize=16)
    plt.subplots_adjust(wspace=0.3, hspace=0.3) # Adjust spacing between subplots
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def split_data(features, targets, test_size=0.25, random_state=0):
    """
    Splits the dataset into training and testing sets.

    Args:
        features (numpy.ndarray): The feature data (flattened images).
        targets (numpy.ndarray): The target labels (person IDs).
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): Controls the shuffling applied to the data before splitting.
                            Ensures reproducibility.

    Returns:
        tuple: A tuple containing (x_train, x_test, y_train, y_test).
    """
    print(f"\nSplitting data into training ({(1-test_size)*100:.0f}%) and testing ({test_size*100:.0f}%) sets...")
    # `stratify=targets` is crucial for multi-class classification, especially when dealing
    # with varying numbers of samples per class. It ensures that the proportion of classes
    # in the training and testing sets is roughly the same as in the original dataset.
    x_train, x_test, y_train, y_test = train_test_split(
        features, targets, test_size=test_size, stratify=targets, random_state=random_state
    )
    print(f"Training set size: {len(x_train)} samples")
    print(f"Testing set size: {len(x_test)} samples")
    return x_train, x_test, y_train, y_test

def apply_pca_dimensionality_reduction(x_train, x_test, n_components=0.99, whiten=True):
    """
    Applies PCA for dimensionality reduction, retaining a specified percentage of variance.
    Also applies whitening if specified.

    Whitening transforms the data such that it has unit variance and zero covariance.
    This can sometimes improve the performance of downstream estimators.

    Args:
        x_train (numpy.ndarray): Training features.
        x_test (numpy.ndarray): Testing features.
        n_components (float or int): If float (0-1), it specifies the variance to be retained.
                                     If int, it's the number of components.
        whiten (bool): Whether to whiten the data.

    Returns:
        tuple: (x_train_pca, x_test_pca, pca_model)
    """
    print(f"\n--- Applying PCA (retaining {n_components*100:.0f}% variance, whiten={whiten}) ---")
    pca = PCA(n_components=n_components, whiten=whiten, random_state=42) # Add random_state for consistency
    # Fit PCA on training data ONLY
    x_train_pca = pca.fit_transform(x_train)
    # Transform test data using the PCA model fitted on training data
    x_test_pca = pca.transform(x_test)

    print(f'Number of components selected by PCA: {pca.n_components_}')
    print(f'Cumulative explained variance by selected components: {np.sum(pca.explained_variance_ratio_):.4f}')
    print(f"Shape after PCA (Training): {x_train_pca.shape}")
    print(f"Shape after PCA (Testing): {x_test_pca.shape}")

    # Optional: Display cumulative explained variance plot
    # plt.figure(figsize=(10, 6))
    # plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--')
    # plt.xlabel('Number of Components')
    # plt.ylabel('Cumulative Explained Variance')
    # plt.title('Cumulative Explained Variance by PCA Components')
    # plt.grid(True)
    # plt.axhline(y=n_components, color='r', linestyle='-', label=f'{n_components*100:.0f}% Variance Threshold')
    # plt.axvline(x=pca.n_components_, color='g', linestyle='--', label=f'{pca.n_components_} Components for {n_components*100:.0f}%')
    # plt.legend()
    # plt.show()

    return x_train_pca, x_test_pca, pca

def define_models_and_param_grids():
    """
    Defines the classification models and their hyperparameter grids for GridSearchCV.

    Returns:
        dict: A dictionary where keys are model names and values are dictionaries
              containing 'model' (estimator) and 'params' (parameter grid).
    """
    print("\n--- Defining Models and Hyperparameter Grids ---")
    models = {
        'LogisticRegression': {
            'model': LogisticRegression(max_iter=5000, random_state=42), # Increased max_iter, added random_state
            'params': {
                'C': [0.01, 0.1, 1, 10, 100], # Regularization parameter
                'solver': ['liblinear', 'lbfgs'] # Algorithms to use for optimization problem
            }
        },
        'SVC': {
            'model': SVC(random_state=42), # Added random_state
            'params': {
                "C": [0.1, 1, 10], # Regularization parameter
                "kernel": ['linear', 'rbf'], # Reduced poly and sigmoid for faster execution if needed
                "gamma": ['scale', 'auto'] # Kernel coefficient
            }
        },
        "GaussianNB": {
            "model": GaussianNB(), # No random_state for GaussianNB
            "params": {
                "var_smoothing": np.logspace(-9, 0, 10) # Laplace smoothing parameter
            }
        }
    }
    return models

def train_and_evaluate_with_gridsearch(models_dict, x_train_pca, y_train, x_test_pca, y_test):
    """
    Performs GridSearchCV for each defined model and evaluates its performance.

    Args:
        models_dict (dict): Dictionary of models and their parameter grids.
        x_train_pca (numpy.ndarray): PCA-transformed training features.
        y_train (numpy.ndarray): Training target labels.
        x_test_pca (numpy.ndarray): PCA-transformed testing features.
        y_test (numpy.ndarray): Testing target labels.

    Returns:
        pandas.DataFrame: A DataFrame summarizing the results for each model.
    """
    results = []
    print("\n--- Training and Evaluating Models with GridSearchCV ---")
    for model_name, model_info in models_dict.items():
        print(f'\nTraining {model_name}...')
        # Initialize GridSearchCV
        # scoring='accuracy': Metric to optimize for during cross-validation.
        # n_jobs=-1: Use all available CPU cores for parallel processing.
        # cv=5: Using 5-fold cross-validation by default in GridSearchCV
        grid = GridSearchCV(
            estimator=model_info['model'],
            param_grid=model_info['params'],
            scoring='accuracy',
            n_jobs=-1,
            cv=5, # Explicitly set cross-validation folds
            verbose=1 # Print progress
        )
        grid.fit(x_train_pca, y_train)

        # Predict on the held-out test set using the best estimator
        y_pred = grid.best_estimator_.predict(x_test_pca)
        test_accuracy = accuracy_score(y_test, y_pred)

        print(f"  Best parameters for {model_name}: {grid.best_params_}")
        print(f"  Best cross-validation accuracy: {grid.best_score_:.4f}")
        print(f"  Test set accuracy: {test_accuracy:.4f}")

        # You can also print more detailed reports for the best model if desired
        # print(f"\nConfusion Matrix for {model_name}:\n{confusion_matrix(y_test, y_pred)}")
        # print(f"\nClassification Report for {model_name}:\n{classification_report(y_test, y_pred)}")

        results.append({
            'Model': model_name,
            'Best Params': grid.best_params_,
            'Best CV Accuracy': grid.best_score_,
            'Test Accuracy': test_accuracy
        })
    return pd.DataFrame(results)

def plot_accuracy_results(results_df):
    """
    Plots the test accuracy and best cross-validation accuracy for each model.

    Args:
        results_df (pandas.DataFrame): DataFrame containing model comparison results.
    """
    print('\n--- Plotting Model Comparison Results ---')
    fig, axes = plt.subplots(1, 2, figsize=(16, 7)) # Adjust figure size for better readability

    # Plot Test Accuracy
    axes[0].bar(results_df["Model"], results_df["Test Accuracy"], color='skyblue', label='Test Accuracy')
    axes[0].set_title('Model Test Accuracy')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_xlabel('Model')
    axes[0].set_ylim([0.8, 1.0]) # Set y-axis limits for better comparison
    axes[0].legend()
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)
    for index, value in enumerate(results_df["Test Accuracy"]):
        axes[0].text(index, value + 0.01, f'{value:.2f}', ha='center', va='bottom', fontsize=9)


    # Plot Best CV Accuracy
    axes[1].bar(results_df["Model"], results_df["Best CV Accuracy"], color='orange', alpha=0.7, label='Best CV Accuracy')
    axes[1].set_title('Model Best Cross-Validation Accuracy')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_xlabel('Model')
    axes[1].set_ylim([0.8, 1.0]) # Set y-axis limits for better comparison
    axes[1].legend()
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)
    for index, value in enumerate(results_df["Best CV Accuracy"]):
        axes[1].text(index, value + 0.01, f'{value:.2f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle('Performance Comparison of Different Classification Models', fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust spacing and make room for suptitle
    plt.show()


if __name__ == "__main__":
    # Define parameters for data processing and model training
    TEST_SPLIT_SIZE = 0.25
    DATA_RANDOM_STATE = 0 # For reproducibility of train-test split
    PCA_VARIANCE_RETAINED = 0.99 # Percentage of variance to retain with PCA
    PCA_WHITEN = True # Whether to apply whitening

    # 1. Load the Olivetti Faces Dataset
    olivetti_data = load_olivetti_faces_data()
    features = olivetti_data.data
    targets = olivetti_data.target

    # 2. Visualize Sample Images from the Dataset
    visualize_dataset_samples(features, targets)
    visualize_person_images(features, person_id=0) # Example: Show images for the first person

    # 3. Split the Data into Training and Test Sets
    x_train, x_test, y_train, y_test = split_data(
        features, targets,
        test_size=TEST_SPLIT_SIZE,
        random_state=DATA_RANDOM_STATE
    )

    # 4. Apply PCA for Dimensionality Reduction (Eigenfaces)
    # The data is already normalized by fetch_olivetti_faces, so no explicit StandardScaler is needed.
    x_train_pca, x_test_pca, pca_model = apply_pca_dimensionality_reduction(
        x_train, x_test,
        n_components=PCA_VARIANCE_RETAINED,
        whiten=PCA_WHITEN
    )

    # 5. Define Models and Hyperparameter Grids for GridSearchCV
    models_to_compare = define_models_and_param_grids()

    # 6. Train and Evaluate Each Model using GridSearchCV
    comparison_results_df = train_and_evaluate_with_gridsearch(
        models_to_compare,
        x_train_pca, y_train,
        x_test_pca, y_test
    )

    # 7. Display and Plot Results
    print('\nModel Comparison Results:')
    print(comparison_results_df)

    plot_accuracy_results(comparison_results_df)

    print("\nScript execution complete.")
