# 🧠 Parkinson's Disease Detection System

A Streamlit web application for early detection of Parkinson's disease using machine learning analysis of voice measurements.

## 📋 Project Information

- **Course**: SECB3203 Bioinformatics 2
- **Semester**: SEM 1 2025/2026
- **Section**: 02
- **Lecturer**: Dr. Seah Choon Sen

## 👥 Group Members

| Name | Matric Number |
|------|---------------|
| Muhammad Amirun Irfan bin Samsul Shah | A23CS0121 |
| Muhammad Hafiz bin Mohd Shaharuddin | A23CS0130 |
| Muhammad Zafril Ihsan bin Mohamad Nasir | A23CS0304 |

## 🎯 Project Overview

This application extends the previous SECB3203 project (Parkinson's Disease Classification) into a deployable web application using Streamlit. The app allows users to:

1. Input voice measurement values
2. Get real-time Parkinson's disease predictions
3. View model performance metrics
4. Explore the training dataset
5. Learn about the features and methodology

## 📁 Project Structure

```
parkinson_app/
├── app.py                  # Main Streamlit application
├── train_model.py          # Model training script
├── requirements.txt        # Python dependencies
├── Parkinsson_disease.csv  # Dataset
├── README.md               # This file
│
# Generated after running train_model.py:
├── parkinson_model.pkl     # Trained SVM model
├── scaler.pkl              # StandardScaler for preprocessing
├── feature_names.pkl       # List of feature names
├── model_metrics.pkl       # Model performance metrics
└── sample_data.pkl         # Sample data for testing
```

## 🚀 Getting Started

### Local Development

1. **Clone/Download the repository**
   ```bash
   git clone <repository-url>
   cd parkinson_app
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model** (generates .pkl files)
   ```bash
   python train_model.py
   ```

5. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

6. **Open in browser**
   - The app will automatically open at `http://localhost:8501`

### Deployment to Streamlit Cloud

1. **Push to GitHub**
   - Create a new GitHub repository
   - Push all files including the .pkl model files

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account
   - Click "New app"
   - Select your repository, branch, and main file (app.py)
   - Click "Deploy"

3. **Important for deployment**:
   - Make sure all .pkl files are committed to the repository
   - Or include `train_model.py` and set it to run before `app.py`

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | 94.87% |
| Precision | 93.33% |
| Recall | 96.55% |
| F1-Score | 94.92% |
| AUC-ROC | 0.9862 |

## 🔬 Features Used

The model analyzes 22 voice measurement features:

### Frequency Measures
- MDVP:Fo(Hz) - Average vocal fundamental frequency
- MDVP:Fhi(Hz) - Maximum vocal fundamental frequency
- MDVP:Flo(Hz) - Minimum vocal fundamental frequency

### Jitter Measures (Frequency Variation)
- MDVP:Jitter(%) - Percentage jitter
- MDVP:Jitter(Abs) - Absolute jitter
- MDVP:RAP - Relative amplitude perturbation
- MDVP:PPQ - Five-point period perturbation quotient
- Jitter:DDP - Average absolute difference of differences

### Shimmer Measures (Amplitude Variation)
- MDVP:Shimmer - Local shimmer
- MDVP:Shimmer(dB) - Shimmer in dB
- Shimmer:APQ3 - Three-point amplitude perturbation quotient
- Shimmer:APQ5 - Five-point amplitude perturbation quotient
- MDVP:APQ - Eleven-point amplitude perturbation quotient
- Shimmer:DDA - Average absolute differences between amplitudes

### Noise Measures
- NHR - Noise-to-harmonics ratio
- HNR - Harmonics-to-noise ratio

### Nonlinear Measures
- RPDE - Recurrence period density entropy
- DFA - Detrended fluctuation analysis
- spread1 - Nonlinear measure of fundamental frequency variation
- spread2 - Nonlinear measure of fundamental frequency variation
- D2 - Correlation dimension
- PPE - Pitch period entropy

## ⚠️ Disclaimer

This application is intended for **educational and research purposes only**. It should not be used as a substitute for professional medical diagnosis. If you have concerns about Parkinson's disease, please consult a qualified healthcare professional.

## 📖 References

1. **Dataset**: UCI Machine Learning Repository - Parkinson's Disease Dataset
2. **Original Paper**: Little MA, McSharry PE, Hunter EJ, Spielman J, Ramig LO. "Suitability of dysphonia measurements for telemonitoring of Parkinson's disease." IEEE Trans Biomed Eng. 2009;56(4):1015-22.

## 📝 License

This project is created for educational purposes as part of SECB3203 course requirements at Universiti Teknologi Malaysia.
