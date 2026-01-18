# SECB3203 Bioinformatics 2 - Project Report

## Parkinson's Disease Detection Web Application

---

**University:** Universiti Teknologi Malaysia  
**Faculty:** Faculty of Computing  
**Course:** SECB3203 Bioinformatics 2  
**Section:** 02  
**Semester:** SEM 1 2025/2026  
**Lecturer:** Dr. Seah Choon Sen

**Group Members:**

| Name | Matric Number |
|------|---------------|
| Muhammad Amirun Irfan bin Samsul Shah | A23CS0121 |
| Muhammad Hafiz bin Mohd Shaharuddin | A23CS0130 |
| Muhammad Zafril Ihsan bin Mohamad Nasir | A23CS0304 |

---

## Table of Contents

1. Introduction of the Project
2. Introduction about the Dataset
3. Methodology / Steps in Constructing the Models
4. Findings of the Model Training and Evaluation
5. Explanation of Streamlit Application Functionalities
6. Deployment Instructions
7. Conclusions

---

## 1. Introduction of the Project

### 1.1 Project Overview

This project presents a **web-based application for early detection of Parkinson's disease** using machine learning techniques. The application is built using **Streamlit**, a Python framework for creating interactive web applications, and leverages a **Support Vector Machine (SVM)** model trained on voice measurement data.

This application extends the previous SECB3203 Programming for Bioinformatics project by transforming the trained machine learning model into a deployable, user-friendly web interface that can be accessed by healthcare professionals and researchers.

### 1.2 Background

**Parkinson's disease (PD)** is a progressive neurodegenerative disorder that affects movement and coordination. It is the second most common neurodegenerative disease after Alzheimer's, affecting approximately 1% of the population over 60 years old. 

**Why voice analysis?**  
Research has shown that vocal abnormalities, including changes in pitch, tremor, and hoarseness, often appear **before other motor symptoms**. By analyzing voice measurements using machine learning algorithms, we can potentially detect PD at earlier stages when treatment is most effective.

### 1.3 Application Features

The Streamlit application provides the following features:

1. **Real-time Prediction**: Input voice measurements and get instant disease prediction
2. **Probability Display**: Visual gauge showing confidence level of predictions
3. **Model Performance Dashboard**: View accuracy, precision, recall, F1-score, and AUC-ROC
4. **Data Exploration Tools**: Interactive visualizations of the training dataset
5. **Educational Information**: Learn about the features and methodology used

---

## 2. Introduction about the Dataset

### 2.1 Data Source

The dataset used in this project is the **Parkinson's Disease Dataset** from the **UCI Machine Learning Repository**. It was originally collected by Max Little at the University of Oxford in collaboration with the National Centre for Voice and Speech, Denver, Colorado.

### 2.2 Dataset Overview

| Attribute | Description |
|-----------|-------------|
| **Total Samples** | 195 voice recordings |
| **Total Features** | 22 acoustic measurements |
| **Target Variable** | status (1 = Parkinson's, 0 = Healthy) |
| **Parkinson's Cases** | 147 samples (75.4%) |
| **Healthy Cases** | 48 samples (24.6%) |
| **Source** | UCI Machine Learning Repository |

### 2.3 Feature Categories

The 22 voice measurement features are organized into four categories:

#### 2.3.1 Frequency Measures (3 features)

| Feature | Description | Unit |
|---------|-------------|------|
| MDVP:Fo(Hz) | Average vocal fundamental frequency | Hertz |
| MDVP:Fhi(Hz) | Maximum vocal fundamental frequency | Hertz |
| MDVP:Flo(Hz) | Minimum vocal fundamental frequency | Hertz |

#### 2.3.2 Jitter Measures - Frequency Variation (5 features)

| Feature | Description |
|---------|-------------|
| MDVP:Jitter(%) | Percentage variation in fundamental frequency |
| MDVP:Jitter(Abs) | Absolute jitter in microseconds |
| MDVP:RAP | Relative amplitude perturbation |
| MDVP:PPQ | Five-point period perturbation quotient |
| Jitter:DDP | Average absolute difference of differences between cycles |

#### 2.3.3 Shimmer Measures - Amplitude Variation (6 features)

| Feature | Description |
|---------|-------------|
| MDVP:Shimmer | Local shimmer (amplitude variation) |
| MDVP:Shimmer(dB) | Shimmer in decibels |
| Shimmer:APQ3 | Three-point amplitude perturbation quotient |
| Shimmer:APQ5 | Five-point amplitude perturbation quotient |
| MDVP:APQ | Eleven-point amplitude perturbation quotient |
| Shimmer:DDA | Average absolute differences between consecutive amplitudes |

#### 2.3.4 Noise and Nonlinear Measures (8 features)

| Feature | Description |
|---------|-------------|
| NHR | Noise-to-harmonics ratio |
| HNR | Harmonics-to-noise ratio |
| RPDE | Recurrence period density entropy |
| DFA | Detrended fluctuation analysis |
| spread1 | Nonlinear measure of fundamental frequency variation |
| spread2 | Nonlinear measure of fundamental frequency variation |
| D2 | Correlation dimension |
| PPE | Pitch period entropy |

---

## 3. Methodology / Steps in Constructing the Models

### 3.1 Machine Learning Pipeline

The model construction follows a systematic machine learning pipeline:

```
[Data Loading] → [Data Cleaning] → [Feature/Target Separation] 
       ↓
[Train-Test Split] → [Feature Scaling] → [Model Training] 
       ↓
[Hyperparameter Tuning] → [Model Evaluation] → [Model Saving]
```

### 3.2 Detailed Steps

#### Step 1: Data Loading and Cleaning
```python
# Load the dataset
df = pd.read_csv('Parkinsson_disease.csv')

# Remove the 'name' column (identifier only)
df = df.drop(columns=['name'])
```

**Why remove 'name'?** The name column contains identifiers like "phon_R01_S01_1" which have no predictive value and could cause the model to memorize instead of learn patterns.

#### Step 2: Feature and Target Separation
```python
# X = Features (22 voice measurements)
X = df.drop(columns=['status'])

# y = Target (0 = Healthy, 1 = Parkinson's)
y = df['status']
```

#### Step 3: Train-Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% for testing
    random_state=42,    # For reproducibility
    stratify=y          # Maintain class distribution
)
```

**Result:**
- Training set: 156 samples (80%)
- Testing set: 39 samples (20%)

#### Step 4: Feature Scaling
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Why StandardScaler?**
- SVM is sensitive to feature scales
- Different features have different ranges (frequency in Hz vs. ratios)
- StandardScaler transforms each feature to have mean=0 and std=1

#### Step 5: Hyperparameter Tuning with Grid Search
```python
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 0.01],
    'kernel': ['rbf']
}

grid_search = GridSearchCV(
    estimator=SVC(probability=True, random_state=42),
    param_grid=param_grid,
    cv=5,              # 5-fold cross-validation
    scoring='accuracy'
)

grid_search.fit(X_train_scaled, y_train)
best_model = grid_search.best_estimator_
```

**Best Parameters Found:**
- C = 10
- gamma = 'scale'
- kernel = 'rbf'

#### Step 6: Model Saving for Deployment
```python
import joblib

# Save model and preprocessing objects
joblib.dump(best_model, 'parkinson_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(feature_names, 'feature_names.pkl')
joblib.dump(metrics, 'model_metrics.pkl')
```

### 3.3 Why SVM?

Support Vector Machine was chosen due to:

1. **Effective in high-dimensional spaces** (22 features)
2. **Memory efficient** - uses only support vectors
3. **Versatile** - RBF kernel captures non-linear boundaries
4. **Works well with small datasets** (195 samples)
5. **Provides probability estimates** for predictions

---

## 4. Findings of the Model Training and Evaluation

### 4.1 Performance Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Accuracy** | 94.87% | 37 out of 39 predictions correct |
| **Precision** | 93.33% | When predicting Parkinson's, 93.33% are correct |
| **Recall** | 96.55% | Catches 96.55% of all Parkinson's cases |
| **F1-Score** | 94.92% | Balanced measure of precision and recall |
| **AUC-ROC** | 0.9862 | Excellent discrimination ability |

### 4.2 Confusion Matrix Analysis

```
                    Predicted
                  Healthy  Parkinson's
Actual  Healthy      8         1
        Parkinson's  1        29
```

**Interpretation:**
- **True Negatives (TN): 8** - Correctly identified healthy individuals
- **True Positives (TP): 29** - Correctly identified Parkinson's cases
- **False Negatives (FN): 1** - Missed Parkinson's case (dangerous!)
- **False Positives (FP): 1** - Healthy person flagged as Parkinson's

### 4.3 Cross-Validation Results

5-Fold Cross-Validation on training data:
- **Mean CV Score:** 93.58%
- **Standard Deviation:** ±2.45%

This indicates the model generalizes well and is not overfitting.

### 4.4 Key Findings

1. **High Recall is Crucial**: With 96.55% recall, the model rarely misses actual Parkinson's cases. This is critical in medical applications where missing a diagnosis can have serious consequences.

2. **Excellent Discrimination**: AUC-ROC of 0.9862 indicates the model can reliably distinguish between healthy and Parkinson's voice patterns.

3. **Robust Performance**: Cross-validation shows consistent performance across different data splits.

4. **Model Comparison**: During development (Progress 4-5), SVM outperformed Logistic Regression, Random Forest, and Gradient Boosting after hyperparameter tuning.

---

## 5. Explanation of Streamlit Application Functionalities

### 5.1 Application Structure

The application consists of **5 main pages** accessible through the sidebar navigation:

### 5.2 Page-by-Page Functionality

#### 🏠 HOME PAGE

**What it shows:**
- Welcome message and project overview
- Model performance metrics at a glance (Accuracy, Precision, Recall, F1, AUC)
- Feature categories overview
- Quick start guide

**User actions:**
- Read about the project
- View key metrics
- Navigate to other pages

---

#### 🔮 PREDICT PAGE (Main Feature)

**What it shows:**
- Input form for all 22 voice features
- Quick load buttons for sample data
- Prediction result with probability gauge
- Probability breakdown chart

**Step-by-Step Usage:**

1. **Load Sample Data (Optional)**
   - Click "Load Healthy Sample" to see typical healthy values
   - Click "Load Parkinson's Sample" to see typical Parkinson's values
   - Click "Reset to Default" to use average values

2. **Enter Voice Measurements**
   - Features are organized into expandable categories
   - Each input shows the feature name and acceptable range
   - Hover over the "?" icon for feature description

3. **Make Prediction**
   - Click the "🔮 Predict Parkinson's Disease" button
   - View the result (Healthy or Parkinson's detected)
   - See the probability gauge (0-100%)
   - View the probability breakdown bar chart

**Output Interpretation:**
- **Green result + probability < 50%**: Likely healthy
- **Red result + probability > 50%**: Parkinson's characteristics detected
- **Gauge color**: Green (0-30%), Yellow (30-70%), Red (70-100%)

---

#### 📊 MODEL PERFORMANCE PAGE

**What it shows:**
- Model type and hyperparameters
- Training information (samples, CV score)
- Performance metrics with visual gauges
- Confusion matrix heatmap
- Performance metrics table

**User actions:**
- Understand how reliable the model is
- View detailed confusion matrix
- Learn about each metric's meaning

---

#### 📈 DATA EXPLORATION PAGE

**What it shows:**
- Dataset overview (total samples, features, class counts)
- Class distribution pie and bar charts
- Feature distribution histograms and box plots
- Correlation heatmap
- Raw data preview

**Interactive Features:**
1. **Feature Selection**: Choose which features to visualize
2. **Distribution Plots**: Compare distributions between classes
3. **Box Plots**: See outliers and quartiles by class
4. **Correlation Analysis**: Select features for correlation heatmap
5. **Data Preview**: Toggle raw data and statistics display

---

#### 📚 ABOUT PAGE

**What it shows:**
- Project overview and objectives
- Background on Parkinson's disease
- How the detection system works
- Feature descriptions table
- Machine learning model explanation
- Team information
- References and disclaimer

---

### 5.3 Session State Management

The application uses Streamlit's session state to:
- Remember which sample data was loaded
- Maintain user selections across interactions
- Cache loaded models for faster performance

---

## 6. Deployment Instructions

### 6.1 Local Development

```bash
# 1. Clone the repository
git clone <repository-url>
cd parkinson_app

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the model (generates .pkl files)
python train_model.py

# 5. Run the Streamlit app
streamlit run app.py
```

### 6.2 Deployment to Streamlit Cloud

1. **Push to GitHub**
   - Create a new GitHub repository
   - Push all files including the .pkl model files

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select repository, branch, and `app.py`
   - Click "Deploy"

3. **Share the URL**
   - Once deployed, share the provided URL
   - Example: `https://your-app-name.streamlit.app`

### 6.3 Required Files for Deployment

| File | Description |
|------|-------------|
| `app.py` | Main Streamlit application |
| `train_model.py` | Model training script |
| `requirements.txt` | Python dependencies |
| `Parkinsson_disease.csv` | Dataset |
| `parkinson_model.pkl` | Trained SVM model |
| `scaler.pkl` | StandardScaler for preprocessing |
| `feature_names.pkl` | List of feature names |
| `model_metrics.pkl` | Performance metrics |
| `sample_data.pkl` | Sample data for testing |
| `README.md` | Documentation |

---

## 7. Conclusions

### 7.1 Achievements

This project successfully:

1. **Developed a functional web application** for Parkinson's disease detection using voice measurements

2. **Achieved high model accuracy** (94.87%) with excellent recall (96.55%), critical for medical applications

3. **Created an intuitive user interface** that allows non-technical users to make predictions

4. **Provided educational value** through data exploration and model explanation features

5. **Enabled easy deployment** through Streamlit Cloud for public access

### 7.2 Limitations

1. **Dataset Size**: 195 samples is relatively small for machine learning
2. **Voice Data Only**: Real diagnosis should include clinical examination
3. **Binary Classification**: Does not indicate disease severity

### 7.3 Future Improvements

1. Add more classification models for comparison
2. Implement audio file upload for automatic feature extraction
3. Add multi-language support
4. Include disease severity prediction
5. Connect with healthcare databases

### 7.4 Disclaimer

This application is intended for **educational and research purposes only**. It should not be used as a substitute for professional medical diagnosis. If you have concerns about Parkinson's disease, please consult a qualified healthcare professional.

---

## References

1. Little MA, McSharry PE, Hunter EJ, Spielman J, Ramig LO. "Suitability of dysphonia measurements for telemonitoring of Parkinson's disease." IEEE Trans Biomed Eng. 2009;56(4):1015-22.

2. UCI Machine Learning Repository - Parkinson's Disease Dataset

3. Streamlit Documentation - https://docs.streamlit.io

4. Scikit-learn Documentation - https://scikit-learn.org

---

**End of Report**

*SECB3203 Bioinformatics 2 | SEM 1 2025/2026 | UTM Faculty of Computing*
