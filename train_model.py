"""
================================================================================
PARKINSON'S DISEASE CLASSIFICATION - MODEL TRAINING SCRIPT
================================================================================

Course: SECB3203 Bioinformatics 2
Project: Early Detection of Parkinson's Disease Using Machine Learning

This script trains the SVM model and saves it for deployment in the Streamlit app.
Run this script first before running the Streamlit app to generate:
1. parkinson_model.pkl - The trained SVM model
2. scaler.pkl - The StandardScaler for preprocessing input data
3. model_metrics.pkl - Performance metrics for display in the app

Group Members:
- MUHAMMAD AMIRUN IRFAN BIN SAMSUL SHAH (A23CS0121)
- MUHAMMAD HAFIZ BIN MOHD SHAHARUDDIN (A23CS0130)
- MUHAMMAD ZAFRIL IHSAN BIN MOHAMAD NASIR (A23CS0304)
================================================================================
"""

# ================================================================================
# SECTION 1: IMPORTING LIBRARIES
# ================================================================================
# These are the tools we need to build our machine learning model

import pandas as pd          # For reading and manipulating data (like Excel in Python)
import numpy as np           # For numerical computations
import joblib               # For saving/loading our trained model
import warnings

# Scikit-learn: The main machine learning library
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, roc_auc_score
)

warnings.filterwarnings('ignore')

print("=" * 70)
print("PARKINSON'S DISEASE CLASSIFICATION - MODEL TRAINING")
print("=" * 70)

# ================================================================================
# SECTION 2: LOADING AND UNDERSTANDING THE DATASET
# ================================================================================
"""
EXPLANATION: What is this dataset?

The Parkinson's Disease dataset contains voice measurements from 195 people:
- 147 people WITH Parkinson's disease (status = 1)
- 48 people WITHOUT Parkinson's disease (status = 0)

Each person's voice was recorded and 22 different features were extracted:
- Frequency measures (MDVP:Fo, Fhi, Flo) - How high or low the voice pitch is
- Jitter measures - Small variations in the voice pitch
- Shimmer measures - Small variations in the voice amplitude (loudness)
- NHR/HNR - Noise-to-harmonic ratio (how "noisy" the voice sounds)
- RPDE, DFA, D2 - Nonlinear measures of voice complexity
- PPE - Pitch period entropy (randomness in voice pitch)
- spread1, spread2 - Measures of fundamental frequency variation

Why voice? Parkinson's disease often affects the muscles that control speech,
causing subtle changes in voice that can be detected before other symptoms appear.
"""

print("\n" + "=" * 70)
print("SECTION 1: Loading Dataset")
print("=" * 70)

# Load the dataset
# pd.read_csv() reads a CSV file into a DataFrame (like a table)
df = pd.read_csv('Parkinsson_disease.csv')

# Remove the 'name' column as it's just an identifier, not a feature
# We use drop() to remove columns we don't need
df = df.drop(columns=['name'])

print(f"\n✓ Dataset loaded successfully!")
print(f"  - Total samples: {len(df)}")
print(f"  - Features: {len(df.columns) - 1}")
print(f"  - Target column: 'status' (1=Parkinson's, 0=Healthy)")

# Count how many of each class we have
class_counts = df['status'].value_counts()
print(f"\n  Class Distribution:")
print(f"  - Parkinson's (1): {class_counts[1]} samples ({class_counts[1]/len(df)*100:.1f}%)")
print(f"  - Healthy (0): {class_counts[0]} samples ({class_counts[0]/len(df)*100:.1f}%)")

# ================================================================================
# SECTION 3: DATA PREPARATION
# ================================================================================
"""
EXPLANATION: Why do we need to prepare data?

Before we can train a machine learning model, we need to:

1. SEPARATE FEATURES AND TARGET:
   - Features (X): The input data (voice measurements) - what we use to make predictions
   - Target (y): The output we want to predict (has Parkinson's or not)

2. SPLIT INTO TRAINING AND TESTING SETS:
   - Training set (80%): Used to teach the model
   - Testing set (20%): Used to evaluate how well the model learned
   
   Why split? If we tested on the same data we trained on, the model might just
   memorize the answers instead of learning the patterns. Testing on unseen data
   shows us how well it will work in the real world.

3. FEATURE SCALING (StandardScaler):
   - Different features have different scales (e.g., frequency in Hz vs ratios)
   - Some features might range from 0-1, others from 100-300
   - Machine learning algorithms work better when all features are on similar scales
   - StandardScaler transforms each feature to have mean=0 and std=1
   
   Formula: scaled_value = (original_value - mean) / standard_deviation
"""

print("\n" + "=" * 70)
print("SECTION 2: Data Preparation")
print("=" * 70)

# Step 1: Separate features (X) and target (y)
# X = all columns except 'status'
# y = only the 'status' column
X = df.drop(columns=['status'])
y = df['status']

feature_names = list(X.columns)  # Save feature names for later use

print(f"\n✓ Features and target separated")
print(f"  - Feature names: {feature_names}")

# Step 2: Split into training and testing sets
# test_size=0.2 means 20% for testing, 80% for training
# random_state=42 ensures reproducibility (same split every time)
# stratify=y ensures both sets have similar proportions of each class
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\n✓ Data split completed")
print(f"  - Training samples: {len(X_train)}")
print(f"  - Testing samples: {len(X_test)}")

# Step 3: Feature scaling
# fit_transform on training data: learns the mean and std, then transforms
# transform on test data: uses the same mean and std (doesn't learn new ones)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n✓ Feature scaling completed using StandardScaler")
print(f"  - All features now have mean ≈ 0 and std ≈ 1")

# ================================================================================
# SECTION 4: MODEL TRAINING WITH HYPERPARAMETER TUNING
# ================================================================================
"""
EXPLANATION: What is SVM (Support Vector Machine)?

SVM is a powerful classification algorithm that works by:
1. Finding the best "boundary" (called hyperplane) that separates the two classes
2. This boundary maximizes the "margin" - the distance between the boundary
   and the closest points from each class

Think of it like this: Imagine you have red and blue marbles on a table.
SVM finds the best way to draw a line that separates them, keeping the line
as far as possible from both colors.

For complex data, SVM uses the "kernel trick":
- RBF (Radial Basis Function) kernel: Can draw curved boundaries
- This is like curving the line to better separate the marbles

Key parameters:
- C (Regularization): Higher C = model tries harder to classify all points correctly
                      but might overfit. Lower C = simpler boundary, more generalization
- gamma: Controls how far the influence of a single point reaches
         High gamma = only nearby points matter (can overfit)
         Low gamma = far away points also matter (can underfit)

WHAT IS GRID SEARCH?

Instead of guessing the best C and gamma values, we try many combinations
systematically and pick the one that performs best. This is called Grid Search.

Grid Search with Cross-Validation:
1. Define a "grid" of parameter values to try
2. For each combination, perform 5-fold cross-validation
3. Pick the combination with the best average score
"""

print("\n" + "=" * 70)
print("SECTION 3: Model Training with Hyperparameter Tuning")
print("=" * 70)

# Define the parameter grid to search
# We'll try all combinations of these values
param_grid = {
    'C': [0.1, 1, 10, 100],           # Regularization strength
    'gamma': ['scale', 'auto', 0.1, 0.01],  # Kernel coefficient
    'kernel': ['rbf']                  # Radial Basis Function kernel
}

print("\n→ Performing Grid Search to find optimal hyperparameters...")
print(f"  Parameter grid: {param_grid}")
print(f"  Total combinations to try: {len(param_grid['C']) * len(param_grid['gamma'])}")

# Create base SVM model
# probability=True allows us to get probability scores, not just predictions
svm_base = SVC(
    probability=True,
    class_weight='balanced',
    random_state=42
)

# Perform Grid Search with 5-fold cross-validation
# cv=5 means: split training data into 5 parts, train on 4, validate on 1, repeat 5 times
grid_search = GridSearchCV(
    estimator=svm_base,
    param_grid=param_grid,
    cv=5,                    # 5-fold cross-validation
    scoring='accuracy',       # Optimize for accuracy
    n_jobs=-1,               # Use all CPU cores
    verbose=1
)

# Fit the grid search (this trains many models!)
grid_search.fit(X_train_scaled, y_train)

# Get the best model
best_model = grid_search.best_estimator_

print(f"\n✓ Grid Search completed!")
print(f"  Best parameters found:")
print(f"  - C: {grid_search.best_params_['C']}")
print(f"  - gamma: {grid_search.best_params_['gamma']}")
print(f"  - kernel: {grid_search.best_params_['kernel']}")
print(f"  - Best CV score: {grid_search.best_score_*100:.2f}%")

# ================================================================================
# SECTION 5: MODEL EVALUATION
# ================================================================================
"""
EXPLANATION: How do we evaluate a classification model?

We use several metrics to understand how well our model performs:

1. ACCURACY: What percentage of predictions are correct?
   Formula: (Correct predictions) / (Total predictions)
   Example: 37 correct out of 39 = 94.87% accuracy

2. PRECISION: When we predict "Parkinson's", how often are we right?
   Formula: True Positives / (True Positives + False Positives)
   Important when: False positives are costly (e.g., unnecessary treatment)

3. RECALL (Sensitivity): Of all actual Parkinson's cases, how many did we catch?
   Formula: True Positives / (True Positives + False Negatives)
   Important when: Missing cases is dangerous (medical diagnosis!)
   High recall = few missed cases

4. F1-SCORE: Harmonic mean of Precision and Recall
   Formula: 2 * (Precision * Recall) / (Precision + Recall)
   Useful when you want a balance between precision and recall

5. AUC-ROC: Area Under the Receiver Operating Characteristic Curve
   Measures the model's ability to distinguish between classes
   Ranges from 0.5 (random guessing) to 1.0 (perfect)

CONFUSION MATRIX:
                    Predicted
                  Neg    Pos
Actual    Neg    [TN     FP]
          Pos    [FN     TP]

- TN (True Negative): Correctly predicted healthy
- TP (True Positive): Correctly predicted Parkinson's
- FN (False Negative): Had Parkinson's but predicted healthy (DANGEROUS!)
- FP (False Positive): Was healthy but predicted Parkinson's
"""

print("\n" + "=" * 70)
print("SECTION 4: Model Evaluation")
print("=" * 70)

# Make predictions on test set
y_pred = best_model.predict(X_test_scaled)
y_prob = best_model.predict_proba(X_test_scaled)[:, 1]  # Probability of Parkinson's

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print(f"\n✓ Model Performance on Test Set:")
print(f"  - Accuracy:  {accuracy*100:.2f}%")
print(f"  - Precision: {precision*100:.2f}%")
print(f"  - Recall:    {recall*100:.2f}%")
print(f"  - F1-Score:  {f1*100:.2f}%")
print(f"  - AUC-ROC:   {auc:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\n✓ Confusion Matrix:")
print(f"                  Predicted")
print(f"                Healthy  Parkinson's")
print(f"  Actual Healthy    {cm[0,0]}        {cm[0,1]}")
print(f"  Actual Parkinson's  {cm[1,0]}       {cm[1,1]}")

# Interpretation
tn, fp, fn, tp = cm.ravel()
print(f"\n  Interpretation:")
print(f"  - True Negatives (correctly identified healthy): {tn}")
print(f"  - True Positives (correctly identified Parkinson's): {tp}")
print(f"  - False Negatives (missed Parkinson's cases): {fn}")
print(f"  - False Positives (healthy wrongly flagged): {fp}")

# Cross-validation score
cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5)
print(f"\n✓ 5-Fold Cross-Validation:")
print(f"  - Mean CV Score: {cv_scores.mean()*100:.2f}%")
print(f"  - Std Deviation: {cv_scores.std()*100:.2f}%")

# ================================================================================
# SECTION 6: SAVING THE MODEL AND ARTIFACTS
# ================================================================================
"""
EXPLANATION: Why do we save the model?

Once we've trained a good model, we don't want to retrain it every time!
We save it to a file so it can be loaded and used for predictions later.

We use 'joblib' to save:
1. The trained model - for making predictions
2. The scaler - to preprocess new input data the same way
3. Metrics - to display in the app

These files will be used by the Streamlit app.
"""

print("\n" + "=" * 70)
print("SECTION 5: Saving Model and Artifacts")
print("=" * 70)

# Save the trained model
joblib.dump(best_model, 'parkinson_model.pkl')
print("\n✓ Model saved: parkinson_model.pkl")

# Save the scaler (needed to preprocess new data)
joblib.dump(scaler, 'scaler.pkl')
print("✓ Scaler saved: scaler.pkl")

# Save the feature names (needed to know input order)
joblib.dump(feature_names, 'feature_names.pkl')
print("✓ Feature names saved: feature_names.pkl")

# Save model metrics for display in app
metrics = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1': f1,
    'auc': auc,
    'confusion_matrix': cm,
    'cv_mean': cv_scores.mean(),
    'cv_std': cv_scores.std(),
    'best_params': grid_search.best_params_,
    'n_train': len(X_train),
    'n_test': len(X_test)
}
joblib.dump(metrics, 'model_metrics.pkl')
print("✓ Model metrics saved: model_metrics.pkl")

# Find the most confidently healthy and Parkinson samples
X_scaled = scaler.transform(X)

probs = best_model.predict_proba(X_scaled)

healthy_idx = np.argmin(probs[:, 1])      # lowest Parkinson probability
parkinson_idx = np.argmax(probs[:, 1])    # highest Parkinson probability

sample_data = {
    'healthy_sample': X.iloc[healthy_idx].to_dict(),
    'parkinson_sample': X.iloc[parkinson_idx].to_dict(),
    'feature_min': X.min().to_dict(),
    'feature_max': X.max().to_dict(),
    'feature_mean': X.mean().to_dict(),
    'feature_std': X.std().to_dict()
}
joblib.dump(sample_data, 'sample_data.pkl')
print("✓ Sample data saved: sample_data.pkl")

print("\n" + "=" * 70)
print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 70)
print("\nGenerated files for deployment:")
print("  1. parkinson_model.pkl - Trained SVM model")
print("  2. scaler.pkl - StandardScaler for preprocessing")
print("  3. feature_names.pkl - List of feature names")
print("  4. model_metrics.pkl - Performance metrics")
print("  5. sample_data.pkl - Sample data for testing")
print("\nNext step: Run the Streamlit app using:")
print("  streamlit run app.py")
print("=" * 70)
