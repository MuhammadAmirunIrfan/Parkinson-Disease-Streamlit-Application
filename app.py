# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Parkinson's Disease Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# LOAD MODEL AND DATA
@st.cache_resource
def load_model():
    """Load the trained model and scaler"""
    model = joblib.load('parkinson_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

@st.cache_resource
def load_metadata():
    """Load feature names, metrics, and sample data"""
    feature_names = joblib.load('feature_names.pkl')
    metrics = joblib.load('model_metrics.pkl')
    sample_data = joblib.load('sample_data.pkl')
    return feature_names, metrics, sample_data

@st.cache_data
def load_dataset():
    """Load the original dataset for exploration"""
    df = pd.read_csv('Parkinsson_disease.csv')
    df = df.drop(columns=['name'])
    return df

def generate_random_sample(sample_data):
    return {
        feature: np.random.uniform(
            sample_data['feature_min'][feature],
            sample_data['feature_max'][feature]
        )
        for feature in sample_data['feature_min']
    }

# Try to load the model files
try:
    model, scaler = load_model()
    feature_names, metrics, sample_data = load_metadata()
    df = load_dataset()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# FEATURE DESCRIPTIONS

FEATURE_DESCRIPTIONS = {
    'MDVP:Fo(Hz)': 'Average vocal fundamental frequency (pitch)',
    'MDVP:Fhi(Hz)': 'Maximum vocal fundamental frequency',
    'MDVP:Flo(Hz)': 'Minimum vocal fundamental frequency',
    'MDVP:Jitter(%)': 'Variation in fundamental frequency (%)',
    'MDVP:Jitter(Abs)': 'Absolute jitter in microseconds',
    'MDVP:RAP': 'Relative amplitude perturbation',
    'MDVP:PPQ': 'Five-point period perturbation quotient',
    'Jitter:DDP': 'Average absolute difference of differences between cycles',
    'MDVP:Shimmer': 'Variation in amplitude',
    'MDVP:Shimmer(dB)': 'Variation in amplitude (dB)',
    'Shimmer:APQ3': 'Three-point amplitude perturbation quotient',
    'Shimmer:APQ5': 'Five-point amplitude perturbation quotient',
    'MDVP:APQ': 'Eleven-point amplitude perturbation quotient',
    'Shimmer:DDA': 'Average absolute differences between amplitudes',
    'NHR': 'Noise-to-harmonics ratio',
    'HNR': 'Harmonics-to-noise ratio',
    'RPDE': 'Recurrence period density entropy',
    'DFA': 'Detrended fluctuation analysis',
    'spread1': 'Nonlinear measure of fundamental frequency variation',
    'spread2': 'Nonlinear measure of fundamental frequency variation',
    'D2': 'Correlation dimension',
    'PPE': 'Pitch period entropy'
}

# SIDEBAR NAVIGATION

st.sidebar.title("Navigation")
st.sidebar.markdown("---")

page = st.sidebar.selectbox(
    "Select Page",
    ["Home", "Predict", "Model Performance", "Data Exploration", "About"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Group Members")
st.sidebar.markdown("- Muhammad Amirun Irfan")
st.sidebar.markdown("- Muhammad Hafiz")
st.sidebar.markdown("- Muhammad Zafril Ihsan")
st.sidebar.markdown("---")
st.sidebar.markdown("**Course:** SECB3203 Bioinformatics 2")
st.sidebar.markdown("**Semester:** SEM 1 2025/2026")

# HOME PAGE

if page == "Home":
    st.title("Parkinson's Disease Detection System")
    st.markdown("""
    ### Welcome to the Early Detection of Parkinson's Disease Application
    
    This application uses **Machine Learning** to analyze voice measurements and predict 
    the likelihood of Parkinson's disease. Early detection can help in timely intervention 
    and better disease management.
    """)
    
    st.markdown("---")
    
    # Display key metrics if model is loaded
    if model_loaded:
        st.subheader("Model Performance Overview")
        
        # Create columns for metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                label="Accuracy",
                value=f"{metrics['accuracy']*100:.1f}%",
                delta="High"
            )
        
        with col2:
            st.metric(
                label="Precision",
                value=f"{metrics['precision']*100:.1f}%",
                delta="High"
            )
        
        with col3:
            st.metric(
                label="Recall",
                value=f"{metrics['recall']*100:.1f}%",
                delta="High"
            )
        
        with col4:
            st.metric(
                label="F1-Score",
                value=f"{metrics['f1']*100:.1f}%",
                delta="High"
            )
        
        with col5:
            st.metric(
                label="AUC-ROC",
                value=f"{metrics['auc']:.3f}",
                delta="Excellent"
            )
    
    st.markdown("---")
    
    # Feature categories
    st.subheader("Voice Features Analyzed")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        #### Frequency Measures
        - Fundamental frequency (Fo)
        - Maximum frequency (Fhi)
        - Minimum frequency (Flo)
        """)
    
    with col2:
        st.markdown("""
        #### Jitter Measures
        - Variation in pitch
        - Period perturbation
        - Cycle differences
        """)
    
    with col3:
        st.markdown("""
        #### Shimmer Measures
        - Amplitude variation
        - Perturbation quotients
        - Noise ratios
        """)
    
    st.markdown("---")
    
    st.info("""
    **How to use this application:**
    1. Navigate to the **Predict** page using the sidebar
    2. Enter voice measurement values or use sample data
    3. Click **Predict** to get the classification result
    4. Explore other pages to learn about the model and data
    """)

# PREDICTION PAGE

elif page == "Predict":
    st.title("Parkinson's Disease Prediction")
    
    if not model_loaded:
        st.error("""
        Model files not found! Please run the training script first:
        
        ```
        python train_model.py
        ```
        """)
    else:
        st.markdown("""
        Enter the voice measurement values below to predict Parkinson's disease.
        You can also load sample data for testing.
        """)
        
        st.markdown("---")
        
        # Sample data buttons
        st.subheader("Quick Load Sample Data")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("Load Healthy Sample", type="secondary"):
                st.session_state['sample_type'] = 'healthy'
                st.rerun()
        
        with col2:
            if st.button("Load Parkinson's Sample", type="secondary"):
                st.session_state['sample_type'] = 'parkinson'
                st.rerun()
        with col3:
            if st.button("Load Random Sample", type="secondary"):
                st.session_state['sample_type'] = 'random'
                st.rerun()

        with col4:
            if st.button("Reset to Default", type="secondary"):
                st.session_state['sample_type'] = 'default'
                st.rerun()
        
        # Determine which values to show
        sample_type = st.session_state.get('sample_type', 'default')
        
        if sample_type == 'healthy':
            current_values = sample_data['healthy_sample']
            st.success("Loaded typical HEALTHY voice pattern values")
        elif sample_type == 'parkinson':
            current_values = sample_data['parkinson_sample']
            st.warning("Loaded typical PARKINSON'S voice pattern values")
        elif sample_type == 'random':
            current_values = generate_random_sample(sample_data)
            st.info("Loaded RANDOM voice pattern values")
           
        else:
            current_values = sample_data['feature_mean']
        
        st.markdown("---")
        
        # Create input form
        st.subheader("Enter Voice Measurements")
        
        # Organize features into categories for better UX
        categories = {
            "Frequency Measures (Hz)": ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)'],
            "Jitter Measures": ['MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP'],
            "Shimmer Measures": ['MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA'],
            "Noise Measures": ['NHR', 'HNR'],
            "Nonlinear Measures": ['RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE']
        }
        
        input_values = {}
        
        for category, features in categories.items():
            with st.expander(category, expanded=True):
                cols = st.columns(3)
                for i, feature in enumerate(features):
                    with cols[i % 3]:
                        min_val = float(sample_data['feature_min'][feature])
                        max_val = float(sample_data['feature_max'][feature])
                        default_val = float(current_values[feature])
                        
                        # Expand the range to accommodate all possible values
                        # Some features like spread1 have negative values
                        range_size = max_val - min_val
                        adjusted_min = min_val - (range_size * 0.5)
                        adjusted_max = max_val + (range_size * 0.5)
                        
                        # Ensure default is within the adjusted range
                        default_val = max(adjusted_min, min(adjusted_max, default_val))
                        
                        input_values[feature] = st.number_input(
                            f"{feature}",
                            min_value=adjusted_min,
                            max_value=adjusted_max,
                            value=default_val,
                            format="%.6f",
                            help=FEATURE_DESCRIPTIONS.get(feature, "Voice measurement feature")
                        )
        
        st.markdown("---")
        
        # Prediction button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            predict_button = st.button("Predict Parkinson's Disease", type="primary", use_container_width=True)
        
        if predict_button:
            # Prepare input data in correct order
            input_array = np.array([[input_values[f] for f in feature_names]])
            
            # Scale the input
            input_scaled = scaler.transform(input_array)
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0]
            
            st.markdown("---")
            st.subheader("Prediction Results")
            
            # Display result with appropriate styling
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.error("### Parkinson's Disease Detected")
                    st.markdown("""
                    The model predicts that the voice pattern shows characteristics 
                    consistent with **Parkinson's disease**.
                    
                    **Note:** This is a screening tool only. Please consult a 
                    medical professional for proper diagnosis.
                    """)
                else:
                    st.success("### No Parkinson's Disease Detected")
                    st.markdown("""
                    The model predicts that the voice pattern appears **healthy** 
                    and does not show typical Parkinson's disease characteristics.
                    
                    **Note:** Regular check-ups are still recommended.
                    """)
            
            with col2:
                # Create probability gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=probability[1] * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Parkinson's Probability (%)", 'font': {'size': 16}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1},
                        'bar': {'color': "darkblue"},
                        'bgcolor': "white",
                        'steps': [
                            {'range': [0, 30], 'color': 'lightgreen'},
                            {'range': [30, 70], 'color': 'yellow'},
                            {'range': [70, 100], 'color': 'salmon'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)
            
            # Probability breakdown
            st.markdown("#### Probability Breakdown")
            prob_df = pd.DataFrame({
                'Classification': ['Healthy', "Parkinson's"],
                'Probability': [probability[0] * 100, probability[1] * 100]
            })
            
            fig = px.bar(
                prob_df, 
                x='Classification', 
                y='Probability',
                color='Classification',
                color_discrete_map={'Healthy': 'green', "Parkinson's": 'red'},
                text=prob_df['Probability'].apply(lambda x: f'{x:.1f}%')
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

# MODEL PERFORMANCE PAGE

elif page == "Model Performance":
    st.title("Model Performance Metrics")
    
    if not model_loaded:
        st.error("Model files not found! Please run the training script first.")
    else:
        st.markdown("""
        This page shows detailed performance metrics of the trained SVM model.
        Understanding these metrics helps assess the reliability of predictions.
        """)
        
        st.markdown("---")
        
        # Model info
        st.subheader("Model Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            **Model Type:** Support Vector Machine (SVM)
            
            **Best Parameters:**
            - C (Regularization): {metrics['best_params']['C']}
            - Gamma: {metrics['best_params']['gamma']}
            - Kernel: {metrics['best_params']['kernel']}
            """)
        
        with col2:
            st.markdown(f"""
            **Training Information:**
            - Training samples: {metrics['n_train']}
            - Testing samples: {metrics['n_test']}
            - Cross-validation: 5-fold
            - CV Score: {metrics['cv_mean']*100:.2f}% +/- {metrics['cv_std']*100:.2f}%
            """)
        
        st.markdown("---")
        
        # Performance metrics
        st.subheader("Classification Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        metric_data = [
            ("Accuracy", metrics['accuracy'], "What % of predictions are correct?"),
            ("Precision", metrics['precision'], "When predicting Parkinson's, how often are we right?"),
            ("Recall", metrics['recall'], "Of all Parkinson's cases, how many did we catch?"),
            ("F1-Score", metrics['f1'], "Harmonic mean of precision and recall")
        ]
        
        for col, (name, value, desc) in zip([col1, col2, col3, col4], metric_data):
            with col:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=value * 100,
                    title={'text': name, 'font': {'size': 14}},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 60], 'color': 'salmon'},
                            {'range': [60, 80], 'color': 'yellow'},
                            {'range': [80, 100], 'color': 'lightgreen'}
                        ]
                    }
                ))
                fig.update_layout(height=200, margin=dict(t=50, b=0, l=20, r=20))
                st.plotly_chart(fig, use_container_width=True)
                st.caption(desc)
        
        st.markdown("---")
        
        # Confusion Matrix
        st.subheader("Confusion Matrix")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            cm = metrics['confusion_matrix']
            fig = px.imshow(
                cm,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=['Healthy', "Parkinson's"],
                y=['Healthy', "Parkinson's"],
                color_continuous_scale='Blues',
                text_auto=True
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            tn, fp, fn, tp = cm.ravel()
            st.markdown("**Matrix Interpretation:**")
            st.markdown(f"- True Negatives: **{tn}**")
            st.markdown(f"- True Positives: **{tp}**")
            st.markdown(f"- False Negatives: **{fn}**")
            st.markdown(f"- False Positives: **{fp}**")
            
            st.info(f"""
            **Key Insight:**
            
            Recall = {metrics['recall']*100:.1f}% means we correctly 
            identified {tp} out of {tp+fn} Parkinson's cases.
            
            High recall is crucial in medical diagnosis to minimize 
            missed cases.
            """)
        
        st.markdown("---")
        
        # Performance summary table
        st.subheader("Model Performance Summary")
        
        summary_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
            'Value': [
                f"{metrics['accuracy']*100:.2f}%",
                f"{metrics['precision']*100:.2f}%",
                f"{metrics['recall']*100:.2f}%",
                f"{metrics['f1']*100:.2f}%",
                f"{metrics['auc']:.4f}"
            ],
            'Interpretation': [
                'Overall correct predictions',
                'Reliability of positive predictions',
                'Ability to find all positive cases',
                'Balance between precision and recall',
                'Discrimination ability (0.5=random, 1.0=perfect)'
            ]
        })
        st.table(summary_df)

# DATA EXPLORATION PAGE

elif page == "Data Exploration":
    st.title("Data Exploration")
    
    if not model_loaded:
        st.error("Model files not found! Please run the training script first.")
    else:
        st.markdown("""
        Explore the Parkinson's Disease dataset used to train the model.
        Understanding the data helps interpret the predictions better.
        """)
        
        st.markdown("---")
        
        # Dataset overview
        st.subheader("Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Samples", len(df))
        with col2:
            st.metric("Features", len(df.columns) - 1)
        with col3:
            st.metric("Parkinson's Cases", sum(df['status'] == 1))
        with col4:
            st.metric("Healthy Cases", sum(df['status'] == 0))
        
        st.markdown("---")
        
        # Class distribution
        st.subheader("Class Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            status_counts = df['status'].value_counts()
            fig = px.pie(
                values=status_counts.values,
                names=["Parkinson's", 'Healthy'],
                title="Distribution of Cases",
                color_discrete_sequence=['#ff6b6b', '#4ecdc4']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                x=['Healthy', "Parkinson's"],
                y=[status_counts[0], status_counts[1]],
                color=['Healthy', "Parkinson's"],
                color_discrete_map={'Healthy': '#4ecdc4', "Parkinson's": '#ff6b6b'},
                title="Case Counts"
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Feature selection for visualization
        st.subheader("Feature Distribution")
        
        features_to_plot = st.multiselect(
            "Select features to visualize",
            options=feature_names,
            default=['MDVP:Fo(Hz)', 'spread1', 'PPE']
        )
        
        if features_to_plot:
            for feature in features_to_plot:
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.histogram(
                        df, x=feature, color='status',
                        barmode='overlay',
                        color_discrete_map={0: '#4ecdc4', 1: '#ff6b6b'},
                        title=f"Distribution of {feature}",
                        labels={'status': 'Status', feature: feature}
                    )
                    fig.update_layout(legend_title="Status")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.box(
                        df, x='status', y=feature,
                        color='status',
                        color_discrete_map={0: '#4ecdc4', 1: '#ff6b6b'},
                        title=f"Box Plot of {feature} by Status"
                    )
                    fig.update_xaxes(ticktext=['Healthy', "Parkinson's"], tickvals=[0, 1])
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Correlation heatmap
        st.subheader("Feature Correlation Heatmap")
        
        top_features = st.multiselect(
            "Select features for correlation analysis",
            options=feature_names + ['status'],
            default=['MDVP:Fo(Hz)', 'spread1', 'PPE', 'HNR', 'RPDE', 'DFA', 'status']
        )
        
        if len(top_features) >= 2:
            corr_matrix = df[top_features].corr()
            fig = px.imshow(
                corr_matrix,
                text_auto='.2f',
                aspect='auto',
                color_continuous_scale='RdBu_r',
                title="Feature Correlation Matrix"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Raw data view
        st.subheader("Raw Data Preview")
        
        if st.checkbox("Show raw data"):
            st.dataframe(df)
        
        if st.checkbox("Show statistical summary"):
            st.dataframe(df.describe())

# ABOUT PAGE

elif page == "About":
    st.title("About This Project")
    
    st.markdown("""
    ## Project Overview
    
    This application was developed as part of the **SECB3203 Bioinformatics 2** course at 
    **Universiti Teknologi Malaysia (UTM)**. The project demonstrates the application of 
    machine learning in healthcare by creating a tool for early detection of Parkinson's 
    disease using voice measurements.
    
    ---
    
    ## What is Parkinson's Disease?
    
    Parkinson's disease is a progressive neurodegenerative disorder that affects movement. 
    Symptoms include tremors, stiffness, and difficulty with balance and coordination. 
    Importantly, **voice changes often appear before other symptoms**, making voice analysis 
    a valuable tool for early detection.
    
    ---
    
    ## How Does the Detection Work?
    
    1. **Voice Recording**: A sustained phonation (saying "ahhh") is recorded
    2. **Feature Extraction**: 22 acoustic features are extracted from the recording
    3. **Machine Learning**: An SVM model analyzes the features
    4. **Prediction**: The model outputs the likelihood of Parkinson's disease
    
    ---
    
    ## Features Used
    
    The model uses various voice measurements:
    
    | Category | Features | Description |
    |----------|----------|-------------|
    | Frequency | Fo, Fhi, Flo | Pitch characteristics |
    | Jitter | Jitter%, RAP, PPQ, DDP | Frequency variation |
    | Shimmer | Shimmer, APQ, DDA | Amplitude variation |
    | Noise | NHR, HNR | Voice quality |
    | Nonlinear | RPDE, DFA, D2, PPE | Complexity measures |
    
    ---
    
    ## The Machine Learning Model
    
    **Algorithm**: Support Vector Machine (SVM) with RBF kernel
    
    **Why SVM?**
    - Effective for binary classification
    - Works well with high-dimensional data
    - Can capture non-linear patterns using the kernel trick
    
    **Training Process:**
    1. Data preprocessing and scaling
    2. Hyperparameter tuning using Grid Search
    3. 5-fold cross-validation for robust evaluation
    4. Final model selection based on recall (to minimize missed cases)
    
    ---
    
    ## Team Members
    
    | Name | Matric Number | Role |
    |------|---------------|------|
    | Muhammad Amirun Irfan bin Samsul Shah | A23CS0121 | Developer |
    | Muhammad Hafiz bin Mohd Shaharuddin | A23CS0130 | Developer |
    | Muhammad Zafril Ihsan bin Mohamad Nasir | A23CS0304 | Developer |
    
    ---
    
    ## References
    
    1. **Dataset**: Kaggle - Parkinson's Disease Detection Dataset
       - https://www.kaggle.com/datasets/debasisdotcom/parkinson-disease-detection
    2. **Original Paper**: Little MA, et al. "Suitability of dysphonia measurements 
       for telemonitoring of Parkinson's disease" (2008)
    
    ---
    
    ## Disclaimer
    
    This application is intended for **educational and research purposes only**. 
    It should not be used as a substitute for professional medical diagnosis. 
    If you have concerns about Parkinson's disease, please consult a qualified 
    healthcare professional.
    
    ---
    
    ## Course Information
    
    - **Course**: SECB3203 Bioinformatics 2
    - **Section**: 01
    - **Semester**: 1 2025/2026
    - **Lecturer**: Ts. Dr. Chan Weng Howe
    - **Institution**: Faculty of Computing, UTM
    """)

# FOOTER
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p>Parkinson's Disease Detection System | SECB3203 Bioinformatics 2</p>
    <p>2025 UTM Faculty of Computing | For Educational Purposes Only</p>
</div>
""", unsafe_allow_html=True)
