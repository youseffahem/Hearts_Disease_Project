"""
Heart Disease Prediction Web Application
Streamlit UI for real-time heart disease risk prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #ff6b6b;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4ecdc4;
        margin-bottom: 1rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .high-risk {
        background-color: #ffe6e6;
        border-left: 5px solid #ff4757;
    }
    .low-risk {
        background-color: #e6f7ff;
        border-left: 5px solid #1abc9c;
    }
    .feature-importance {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_model_and_info():
    """Load the trained model and associated information"""
    try:
        # Load best model info
        with open('results/final_optimized_models_info.json', 'r') as f:
            models_info = json.load(f)

        # Get the best performing model
        best_dataset = None
        best_score = 0
        for dataset, info in models_info.items():
            if info['test_f1_score'] > best_score:
                best_score = info['test_f1_score']
                best_dataset = dataset

        if best_dataset:
            model_info = models_info[best_dataset]
            model = joblib.load(f"models/{model_info['filename']}")
            return model, model_info
        else:
            return None, None
    except FileNotFoundError:
        return None, None


@st.cache_data
def load_feature_info():
    """Load feature selection results"""
    try:
        with open('results/selected_features.json', 'r') as f:
            feature_info = json.load(f)
        return feature_info
    except FileNotFoundError:
        return None


def get_feature_descriptions():
    """Get descriptions for each feature"""
    descriptions = {
        'age': 'Age in years',
        'sex': 'Gender (1 = male, 0 = female)',
        'cp': 'Chest pain type (0: typical angina, 1: atypical angina, 2: non-anginal pain, 3: asymptomatic)',
        'trestbps': 'Resting blood pressure (in mm Hg)',
        'chol': 'Serum cholesterol in mg/dl',
        'fbs': 'Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)',
        'restecg': 'Resting ECG results (0: normal, 1: ST-T wave abnormality, 2: left ventricular hypertrophy)',
        'thalach': 'Maximum heart rate achieved',
        'exang': 'Exercise induced angina (1 = yes, 0 = no)',
        'oldpeak': 'ST depression induced by exercise relative to rest',
        'slope': 'Slope of peak exercise ST segment (0: upsloping, 1: flat, 2: downsloping)',
        'ca': 'Number of major vessels colored by fluoroscopy (0-3)',
        'thal': 'Thalassemia (0: normal, 1: fixed defect, 2: reversible defect)'
    }
    return descriptions


def main():
    """Main application function"""

    # Header
    st.markdown('<h1 class="main-header">❤️ Heart Disease Prediction System</h1>',
                unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <p style='font-size: 1.2rem; color: #666;'>
            AI-powered heart disease risk assessment using machine learning
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Load model and information
    model, model_info = load_model_and_info()
    feature_info = load_feature_info()

    if model is None:
        st.error("⚠️ Model files not found. Please ensure the model has been trained and saved.")
        st.info("Run the complete ML pipeline (notebooks 01-06) to generate the required model files.")
        return

    # Sidebar for model information
    with st.sidebar:
        st.markdown('<h2 class="sub-header">🔧 Model Information</h2>', unsafe_allow_html=True)

        if model_info:
            st.info(f"""
            **Model:** {model_info['model_name']}

            **Performance Metrics:**
            - F1 Score: {model_info['test_f1_score']:.3f}
            - Accuracy: {model_info['test_accuracy']:.3f}
            - AUC Score: {model_info['test_auc']:.3f}

            **Features Used:** {len(model_info['features'])}
            """)

        st.markdown("---")
        st.markdown("""
        **About This Application:**

        This tool uses machine learning to predict heart disease risk based on clinical parameters. 

        ⚠️ **Disclaimer:** This is for educational purposes only and should not replace professional medical advice.
        """)

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<h2 class="sub-header">📝 Patient Information</h2>', unsafe_allow_html=True)

        # Feature input form
        feature_descriptions = get_feature_descriptions()

        # Create input fields based on model features
        if model_info and 'features' in model_info:
            features = model_info['features']
        else:
            # Default features if model info not available
            features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

        # Organize inputs in columns
        input_col1, input_col2 = st.columns(2)

        user_inputs = {}

        with input_col1:
            # Demographic and basic info
            if 'age' in features:
                user_inputs['age'] = st.slider(
                    "Age", min_value=20, max_value=80, value=50,
                    help=feature_descriptions.get('age', '')
                )

            if 'sex' in features:
                user_inputs['sex'] = st.selectbox(
                    "Gender", options=[0, 1],
                    format_func=lambda x: "Female" if x == 0 else "Male",
                    help=feature_descriptions.get('sex', '')
                )

            if 'cp' in features:
                user_inputs['cp'] = st.selectbox(
                    "Chest Pain Type", options=[0, 1, 2, 3],
                    format_func=lambda x: ["Typical Angina", "Atypical Angina",
                                           "Non-Anginal Pain", "Asymptomatic"][x],
                    help=feature_descriptions.get('cp', '')
                )

            if 'trestbps' in features:
                user_inputs['trestbps'] = st.slider(
                    "Resting Blood Pressure (mm Hg)",
                    min_value=90, max_value=200, value=120,
                    help=feature_descriptions.get('trestbps', '')
                )

            if 'chol' in features:
                user_inputs['chol'] = st.slider(
                    "Cholesterol (mg/dl)",
                    min_value=100, max_value=600, value=240,
                    help=feature_descriptions.get('chol', '')
                )

            if 'fbs' in features:
                user_inputs['fbs'] = st.selectbox(
                    "Fasting Blood Sugar > 120 mg/dl", options=[0, 1],
                    format_func=lambda x: "No" if x == 0 else "Yes",
                    help=feature_descriptions.get('fbs', '')
                )

            if 'thalach' in features:
                user_inputs['thalach'] = st.slider(
                    "Maximum Heart Rate",
                    min_value=60, max_value=220, value=150,
                    help=feature_descriptions.get('thalach', '')
                )

        with input_col2:
            # Clinical measurements
            if 'restecg' in features:
                user_inputs['restecg'] = st.selectbox(
                    "Resting ECG", options=[0, 1, 2],
                    format_func=lambda x: ["Normal", "ST-T Wave Abnormality",
                                           "Left Ventricular Hypertrophy"][x],
                    help=feature_descriptions.get('restecg', '')
                )

            if 'exang' in features:
                user_inputs['exang'] = st.selectbox(
                    "Exercise Induced Angina", options=[0, 1],
                    format_func=lambda x: "No" if x == 0 else "Yes",
                    help=feature_descriptions.get('exang', '')
                )

            if 'oldpeak' in features:
                user_inputs['oldpeak'] = st.slider(
                    "ST Depression (Oldpeak)",
                    min_value=0.0, max_value=6.0, value=1.0, step=0.1,
                    help=feature_descriptions.get('oldpeak', '')
                )

            if 'slope' in features:
                user_inputs['slope'] = st.selectbox(
                    "ST Slope", options=[0, 1, 2],
                    format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x],
                    help=feature_descriptions.get('slope', '')
                )

            if 'ca' in features:
                user_inputs['ca'] = st.selectbox(
                    "Major Vessels (0-3)", options=[0, 1, 2, 3],
                    help=feature_descriptions.get('ca', '')
                )

            if 'thal' in features:
                user_inputs['thal'] = st.selectbox(
                    "Thalassemia", options=[0, 1, 2, 3],
                    format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect", "Unknown"][
                        x] if x < 4 else str(x),
                    help=feature_descriptions.get('thal', '')
                )

        # Prediction button
        st.markdown("---")
        if st.button("🔍 Predict Heart Disease Risk", use_container_width=True):
            # Prepare input data
            input_data = []
            for feature in features:
                if feature in user_inputs:
                    input_data.append(user_inputs[feature])
                else:
                    input_data.append(0)  # Default value for missing features

            input_df = pd.DataFrame([input_data], columns=features)

            # Make prediction
            try:
                prediction = model.predict(input_df)[0]
                prediction_proba = model.predict_proba(input_df)[0]

                # Display results in the second column
                with col2:
                    st.markdown('<h2 class="sub-header">📊 Prediction Results</h2>', unsafe_allow_html=True)

                    # Risk level
                    risk_probability = prediction_proba[1] * 100

                    if prediction == 1:
                        st.markdown(f"""
                        <div class="prediction-box high-risk">
                            <h3 style="color: #ff4757; margin-bottom: 1rem;">⚠️ High Risk Detected</h3>
                            <p style="font-size: 1.2rem; margin-bottom: 1rem;">
                                <strong>Risk Probability: {risk_probability:.1f}%</strong>
                            </p>
                            <p style="color: #666;">
                                The model indicates a higher likelihood of heart disease. 
                                Please consult with a healthcare professional for proper evaluation.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="prediction-box low-risk">
                            <h3 style="color: #1abc9c; margin-bottom: 1rem;">✅ Lower Risk</h3>
                            <p style="font-size: 1.2rem; margin-bottom: 1rem;">
                                <strong>Risk Probability: {risk_probability:.1f}%</strong>
                            </p>
                            <p style="color: #666;">
                                The model suggests a lower likelihood of heart disease. 
                                Continue maintaining a healthy lifestyle.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                    # Probability visualization
                    fig, ax = plt.subplots(figsize=(6, 4))
                    colors = ['#1abc9c' if prediction == 0 else '#ff4757',
                              '#ff4757' if prediction == 0 else '#1abc9c']

                    bars = ax.bar(['No Disease', 'Disease'], prediction_proba,
                                  color=colors, alpha=0.7)
                    ax.set_ylabel('Probability')
                    ax.set_title('Prediction Probabilities')
                    ax.set_ylim(0, 1)

                    # Add percentage labels on bars
                    for bar, prob in zip(bars, prediction_proba):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                                f'{prob * 100:.1f}%', ha='center', va='bottom')

                    st.pyplot(fig)
                    plt.close()

                    # Risk factors analysis
                    st.markdown("### 📋 Risk Factor Analysis")

                    # Highlight concerning values
                    risk_factors = []

                    if 'age' in user_inputs and user_inputs['age'] > 55:
                        risk_factors.append("Advanced age")
                    if 'sex' in user_inputs and user_inputs['sex'] == 1:
                        risk_factors.append("Male gender")
                    if 'cp' in user_inputs and user_inputs['cp'] in [0, 1]:
                        risk_factors.append("Chest pain present")
                    if 'trestbps' in user_inputs and user_inputs['trestbps'] > 140:
                        risk_factors.append("High blood pressure")
                    if 'chol' in user_inputs and user_inputs['chol'] > 240:
                        risk_factors.append("High cholesterol")
                    if 'fbs' in user_inputs and user_inputs['fbs'] == 1:
                        risk_factors.append("High fasting blood sugar")
                    if 'thalach' in user_inputs and user_inputs['thalach'] < 100:
                        risk_factors.append("Low maximum heart rate")
                    if 'exang' in user_inputs and user_inputs['exang'] == 1:
                        risk_factors.append("Exercise induced angina")
                    if 'oldpeak' in user_inputs and user_inputs['oldpeak'] > 2:
                        risk_factors.append("Significant ST depression")

                    if risk_factors:
                        st.warning("**Notable risk factors:**")
                        for factor in risk_factors[:5]:  # Show top 5
                            st.write(f"• {factor}")
                    else:
                        st.success("**Good news:** No major risk factors identified!")

            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.info("Please ensure all required fields are filled correctly.")

    # Additional information tabs
    st.markdown("---")
    st.markdown('<h2 class="sub-header">📚 Additional Information</h2>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dataset Info", "🧠 Model Details", "💡 Health Tips", "📈 Visualization"])

    with tab1:
        st.markdown("""
        ### Heart Disease Dataset Information

        This application uses the famous **Heart Disease UCI Dataset** which contains 303 instances 
        with 14 attributes. The dataset is widely used for heart disease prediction research.

        **Key Statistics:**
        - **Total Patients:** 303
        - **Features:** 13 clinical parameters
        - **Target:** Binary classification (Heart Disease: Yes/No)
        - **Class Distribution:** Approximately balanced

        **Data Source:** UCI Machine Learning Repository
        """)

        # Show sample data if available
        try:
            sample_data = pd.read_csv('data/heart_disease_cleaned.csv').head()
            st.subheader("Sample Data")
            st.dataframe(sample_data)
        except FileNotFoundError:
            st.info("Sample data not available. Run the preprocessing notebook to see sample data.")

    with tab2:
        st.markdown("### Model Architecture & Performance")

        if model_info:
            st.markdown(f"""
            **Selected Model:** {model_info['model_name']}

            **Hyperparameters:**
            """)

            # Display best parameters if available
            if 'best_params' in model_info:
                for param, value in model_info['best_params'].items():
                    st.write(f"- **{param}:** {value}")

            st.markdown(f"""
            **Performance Metrics:**
            - **F1 Score:** {model_info['test_f1_score']:.4f}
            - **Accuracy:** {model_info['test_accuracy']:.4f}
            - **ROC AUC:** {model_info['test_auc']:.4f}

            **Cross-Validation:** 5-fold stratified cross-validation
            **Optimization:** GridSearchCV with extensive hyperparameter tuning
            """)

        st.markdown("""
        ### Feature Engineering Applied:
        - Standard scaling of numerical features
        - Feature selection using multiple methods
        - PCA analysis for dimensionality reduction
        - Recursive feature elimination (RFE)
        """)

    with tab3:
        st.markdown("""
        ### 💖 Heart Health Tips

        **Lifestyle Modifications for Heart Health:**

        🥗 **Diet:**
        - Increase fruits and vegetables intake
        - Choose whole grains over refined grains
        - Limit saturated and trans fats
        - Reduce sodium intake
        - Include omega-3 rich foods

        🏃‍♂️ **Exercise:**
        - Aim for 150 minutes of moderate aerobic activity weekly
        - Include strength training exercises
        - Take regular walks
        - Use stairs instead of elevators

        🚭 **Lifestyle:**
        - Quit smoking
        - Limit alcohol consumption
        - Manage stress through meditation or hobbies
        - Get adequate sleep (7-9 hours)
        - Maintain a healthy weight

        🩺 **Medical Care:**
        - Regular check-ups with healthcare provider
        - Monitor blood pressure and cholesterol
        - Take prescribed medications as directed
        - Know your family history

        ⚠️ **Disclaimer:** Always consult with healthcare professionals for personalized advice.
        """)

    with tab4:
        st.markdown("### 📈 Data Visualization & Insights")

        # Feature importance visualization if available
        if hasattr(model, 'feature_importances_'):
            st.subheader("Feature Importance")

            importances = model.feature_importances_
            feature_names = features

            # Create feature importance plot
            fig, ax = plt.subplots(figsize=(10, 6))
            indices = np.argsort(importances)[::-1]

            ax.bar(range(len(importances)), importances[indices])
            ax.set_xticks(range(len(importances)))
            ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
            ax.set_title('Feature Importance in Heart Disease Prediction')
            ax.set_ylabel('Importance Score')

            st.pyplot(fig)
            plt.close()

            # Top features explanation
            st.markdown("**Top Contributing Features:**")
            for i in range(min(5, len(indices))):
                feature_name = feature_names[indices[i]]
                importance = importances[indices[i]]
                st.write(
                    f"{i + 1}. **{feature_name}**: {importance:.3f} - {feature_descriptions.get(feature_name, 'Clinical parameter')}")

        # Model performance visualization
        st.subheader("Model Performance Overview")

        if model_info:
            metrics = ['Accuracy', 'F1 Score', 'AUC Score']
            values = [model_info['test_accuracy'], model_info['test_f1_score'], model_info['test_auc']]

            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.bar(metrics, values, color=['#3498db', '#2ecc71', '#e74c3c'], alpha=0.7)
            ax.set_ylim(0, 1)
            ax.set_ylabel('Score')
            ax.set_title('Model Performance Metrics')

            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')

            st.pyplot(fig)
            plt.close()

        st.markdown("""
        **Interpretation Guide:**
        - **Accuracy:** Overall correctness of predictions
        - **F1 Score:** Balance between precision and recall
        - **AUC Score:** Area under ROC curve (discrimination ability)

        All metrics range from 0 to 1, with higher values indicating better performance.
        """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; margin-top: 2rem;'>
        <p>💻 Developed with Streamlit | 🤖 Powered by Machine Learning | ❤️ For Heart Health Awareness</p>
        <p style='font-size: 0.8rem;'>⚠️ This tool is for educational purposes only and should not replace professional medical consultation.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()