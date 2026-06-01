import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import sys
import os

# Resolve paths relative to this file so the app works regardless of cwd
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'heart.csv')

sys.path.append(os.path.join(BASE_DIR, 'src'))

try:
    from data_analyzer import HeartDiseaseAnalyzer
    from model_trainer import HeartDiseaseModelTrainer
except ImportError:
    st.error("Could not import analysis modules. Please ensure src/ directory contains the required files.")


st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon=" ",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for theme
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

# Custom CSS with modern black theme
def get_custom_css(theme='dark'):
    if theme == 'dark':
        return """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
            
            /* Global Styles */
            * {
                font-family: 'Inter', sans-serif;
            }
            
            /* Main Container */
            .main {
                background-color: #000000;
                color: #D1D1D1;
            }
            
            /* Sidebar */
            [data-testid="stSidebar"] {
                background-color: #121212;
                border-right: 1px solid #1e1e1e;
            }
            
            [data-testid="stSidebar"] .css-1d391kg {
                color: #D1D1D1;
            }
            
            /* Headers */
            .main-header {
                font-size: 3rem;
                font-weight: 700;
                background: linear-gradient(135deg, #00ff88 0%, #00d4ff 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                text-align: center;
                margin-bottom: 2rem;
                letter-spacing: -0.02em;
            }
            
            .sub-header {
                font-size: 1.5rem;
                font-weight: 600;
                color: #ffffff;
                margin-bottom: 1.5rem;
                margin-top: 2rem;
                letter-spacing: -0.01em;
            }
            
            /* Cards */
            .metric-card {
                background: linear-gradient(135deg, #1a1a1a 0%, #121212 100%);
                padding: 1.5rem;
                border-radius: 16px;
                border: 1px solid #1e1e1e;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
                transition: all 0.3s ease;
            }
            
            .metric-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 12px rgba(0, 255, 136, 0.1);
                border-color: #00ff88;
            }
            
            /* Prediction Results */
            .prediction-result {
                font-size: 1.8rem;
                font-weight: 600;
                padding: 2rem;
                border-radius: 16px;
                text-align: center;
                margin: 2rem 0;
                transition: all 0.3s ease;
                animation: fadeIn 0.5s ease-in;
            }
            
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }
            
            .positive-prediction {
                background: linear-gradient(135deg, #1a0a0a 0%, #2a0a0a 100%);
                color: #ff4444;
                border: 2px solid #ff4444;
                box-shadow: 0 0 20px rgba(255, 68, 68, 0.2);
            }
            
            .negative-prediction {
                background: linear-gradient(135deg, #0a1a0a 0%, #0a2a0a 100%);
                color: #00ff88;
                border: 2px solid #00ff88;
                box-shadow: 0 0 20px rgba(0, 255, 136, 0.2);
            }
            
            /* Buttons */
            .stButton > button {
                background: linear-gradient(135deg, #00ff88 0%, #00d4ff 100%);
                color: #000000;
                font-weight: 600;
                border: none;
                border-radius: 12px;
                padding: 0.75rem 2rem;
                font-size: 1rem;
                transition: all 0.3s ease;
                box-shadow: 0 4px 12px rgba(0, 255, 136, 0.3);
            }
            
            .stButton > button:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(0, 255, 136, 0.4);
                background: linear-gradient(135deg, #00ffaa 0%, #00e4ff 100%);
            }
            
            .stButton > button:active {
                transform: translateY(0);
            }
            
            /* Input Fields */
            .stTextInput > div > div > input,
            .stSelectbox > div > div > select,
            .stSlider > div > div > div > div {
                background-color: #1a1a1a;
                color: #D1D1D1;
                border: 1px solid #2a2a2a;
                border-radius: 8px;
                transition: all 0.3s ease;
            }
            
            .stTextInput > div > div > input:focus,
            .stSelectbox > div > div > select:focus {
                border-color: #00ff88;
                box-shadow: 0 0 0 2px rgba(0, 255, 136, 0.1);
            }
            
            /* Slider */
            .stSlider > div > div > div > div {
                background-color: #1a1a1a;
            }
            
            .stSlider > div > div > div > div > div {
                background-color: #00ff88;
            }
            
            /* Slider Value Display */
            .stSlider > div > div > div[data-baseweb="slider"] > div:last-child {
                color: #00ff88 !important;
                font-weight: 600 !important;
                font-size: 1.1rem !important;
            }
            
            /* Slider Label */
            .stSlider > label {
                color: #ffffff !important;
                font-weight: 500 !important;
                font-size: 1rem !important;
                margin-bottom: 0.5rem !important;
            }
            
            /* Metrics */
            [data-testid="stMetricValue"] {
                color: #00ff88;
                font-size: 1.8rem;
                font-weight: 600;
            }
            
            [data-testid="stMetricLabel"] {
                color: #D1D1D1;
                font-weight: 500;
            }
            
            /* Expander */
            .streamlit-expanderHeader {
                background-color: #1a1a1a;
                border-radius: 8px;
                color: #D1D1D1;
                font-weight: 500;
                transition: all 0.3s ease;
            }
            
            .streamlit-expanderHeader:hover {
                background-color: #2a2a2a;
                border-color: #00ff88;
            }
            
            /* Divider */
            hr {
                border-color: #1e1e1e;
                margin: 2rem 0;
            }
            
            /* Info/Warning/Success boxes */
            .stAlert {
                background-color: #1a1a1a;
                border-radius: 12px;
                border-left: 4px solid #00ff88;
                color: #D1D1D1;
            }
            
            /* Selectbox */
            [data-baseweb="select"] {
                background-color: #1a1a1a;
            }
            
            /* Theme Toggle */
            .theme-toggle {
                position: fixed;
                top: 1rem;
                right: 1rem;
                z-index: 999;
                background: linear-gradient(135deg, #1a1a1a 0%, #121212 100%);
                border: 1px solid #2a2a2a;
                border-radius: 12px;
                padding: 0.5rem 1rem;
                cursor: pointer;
                transition: all 0.3s ease;
            }
            
            .theme-toggle:hover {
                border-color: #00ff88;
                box-shadow: 0 0 12px rgba(0, 255, 136, 0.2);
            }
            
            /* Footer */
            .footer {
                text-align: center;
                color: #666;
                padding: 2rem 0;
                font-size: 0.9rem;
                border-top: 1px solid #1e1e1e;
                margin-top: 3rem;
            }
            
            /* Scrollbar */
            ::-webkit-scrollbar {
                width: 10px;
                height: 10px;
            }
            
            ::-webkit-scrollbar-track {
                background: #121212;
            }
            
            ::-webkit-scrollbar-thumb {
                background: #2a2a2a;
                border-radius: 5px;
            }
            
            ::-webkit-scrollbar-thumb:hover {
                background: #00ff88;
            }
        </style>
        """
    else:  # light theme
        return """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
            
            * {
                font-family: 'Inter', sans-serif;
            }
            
            .main {
                background-color: #ffffff;
                color: #1a1a1a;
            }
            
            [data-testid="stSidebar"] {
                background-color: #f5f5f5;
                border-right: 1px solid #e0e0e0;
            }
            
            .main-header {
                font-size: 3rem;
                font-weight: 700;
                background: linear-gradient(135deg, #00aa66 0%, #0099cc 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                text-align: center;
                margin-bottom: 2rem;
            }
            
            .sub-header {
                font-size: 1.5rem;
                font-weight: 600;
                color: #1a1a1a;
                margin-bottom: 1.5rem;
                margin-top: 2rem;
            }
            
            .metric-card {
                background: #ffffff;
                padding: 1.5rem;
                border-radius: 16px;
                border: 1px solid #e0e0e0;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
                transition: all 0.3s ease;
            }
            
            .metric-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0, 170, 102, 0.1);
                border-color: #00aa66;
            }
            
            .prediction-result {
                font-size: 1.8rem;
                font-weight: 600;
                padding: 2rem;
                border-radius: 16px;
                text-align: center;
                margin: 2rem 0;
                transition: all 0.3s ease;
            }
            
            .positive-prediction {
                background: #fff5f5;
                color: #cc0000;
                border: 2px solid #cc0000;
            }
            
            .negative-prediction {
                background: #f0fff5;
                color: #00aa66;
                border: 2px solid #00aa66;
            }
            
            .stButton > button {
                background: linear-gradient(135deg, #00aa66 0%, #0099cc 100%);
                color: #ffffff;
                font-weight: 600;
                border: none;
                border-radius: 12px;
                padding: 0.75rem 2rem;
                transition: all 0.3s ease;
            }
            
            .stButton > button:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(0, 170, 102, 0.3);
            }
            
            [data-testid="stMetricValue"] {
                color: #00aa66;
                font-size: 1.8rem;
                font-weight: 600;
            }
            
            hr {
                border-color: #e0e0e0;
            }
        </style>
        """

st.markdown(get_custom_css(st.session_state.theme), unsafe_allow_html=True)

class HeartDiseaseApp:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.model_info = None
        self.feature_names = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]
        self.feature_descriptions = {
            'age': 'Age (years)',
            'sex': 'Sex (1 = Male, 0 = Female)',
            'cp': 'Chest Pain Type (0-3)',
            'trestbps': 'Resting Blood Pressure (mm Hg)',
            'chol': 'Cholesterol Level (mg/dl)',
            'fbs': 'Fasting Blood Sugar > 120 mg/dl (1 = Yes, 0 = No)',
            'restecg': 'Resting ECG Results (0-2)',
            'thalach': 'Maximum Heart Rate Achieved',
            'exang': 'Exercise Induced Angina (1 = Yes, 0 = No)',
            'oldpeak': 'ST Depression Induced by Exercise',
            'slope': 'Slope of Peak Exercise ST Segment (0-2)',
            'ca': 'Number of Major Vessels (0-3)',
            'thal': 'Thalassemia (1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect)'
        }
    
    def get_chart_theme(self):
        """Get chart theme based on current app theme"""
        if st.session_state.theme == 'dark':
            return {
                'template': 'plotly_dark',
                'paper_bgcolor': '#121212',
                'plot_bgcolor': '#1a1a1a',
                'font_color': '#D1D1D1',
                'grid_color': '#2a2a2a',
                'accent_color': '#00ff88'
            }
        else:
            return {
                'template': 'plotly_white',
                'paper_bgcolor': '#ffffff',
                'plot_bgcolor': '#f8f8f8',
                'font_color': '#1a1a1a',
                'grid_color': '#e0e0e0',
                'accent_color': '#00aa66'
            }
    
    def load_model(self):
        """Load the trained model and scaler"""
        try:
            self.model = joblib.load(os.path.join(MODELS_DIR, 'best_heart_disease_model.pkl'))
            self.scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
            self.model_info = joblib.load(os.path.join(MODELS_DIR, 'model_info.pkl'))
            return True
        except FileNotFoundError:
            return False
    
    def create_input_form(self):
        """Create input form for user data"""
        st.markdown('<div class="sub-header">👤 Patient Information</div>', unsafe_allow_html=True)
        
        # Demographics Section
        st.markdown("##### 📋 Demographics")
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider(" Age (years)", 20, 100, 50, help="Patient's age in years")
        with col2:
            sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male", help="Biological sex")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Vital Signs Section
        st.markdown("##### Vital Signs & Measurements")
        col1, col2 = st.columns(2)
        with col1:
            trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120, help="Blood pressure at rest")
            chol = st.slider("Cholesterol Level (mg/dl)", 100, 600, 200, help="Serum cholesterol")
            thalach = st.slider("Max Heart Rate Achieved", 60, 220, 150, help="Maximum heart rate during exercise")
        
        with col2:
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], 
                             format_func=lambda x: "No" if x == 0 else "Yes",
                             help="Fasting blood sugar greater than 120 mg/dl")
            oldpeak = st.slider("ST Depression (Exercise)", 0.0, 7.0, 1.0, 0.1, 
                              help="ST depression induced by exercise relative to rest")
            ca = st.selectbox(" Number of Major Vessels (0-3)", [0, 1, 2, 3],
                            help="Number of major vessels colored by fluoroscopy")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Clinical Indicators Section
        st.markdown("##### 🏥 Clinical Indicators")
        col1, col2, col3 = st.columns(3)
        with col1:
            cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], 
                            format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"][x],
                            help="Type of chest pain experienced")
        with col2:
            restecg = st.selectbox("Resting ECG", [0, 1, 2],
                                 format_func=lambda x: ["Normal", "ST-T Abnormality", "LV Hypertrophy"][x],
                                 help="Resting electrocardiographic results")
        with col3:
            exang = st.selectbox("Exercise Angina", [0, 1],
                               format_func=lambda x: "No" if x == 0 else "Yes",
                               help="Exercise induced angina")
        
        col1, col2 = st.columns(2)
        with col1:
            slope = st.selectbox("ST Segment Slope", [0, 1, 2],
                               format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x],
                               help="Slope of peak exercise ST segment")
        with col2:
            thal = st.selectbox("Thalassemia", [1, 2, 3],
                              format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect"][x-1],
                              help="Thalassemia blood disorder status")
        
        return [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
    
    def make_prediction(self, input_data):
        """Make prediction using the loaded model"""
        if self.model is None:
            return None, None
        
        # Convert to numpy array and reshape
        input_array = np.array(input_data).reshape(1, -1)
        
        # Scale input — always apply the scaler so inference matches training
        input_scaled = self.scaler.transform(input_array)
        
        # Random Forest is scale-invariant, but we scale consistently for
        # correctness if the model is ever swapped for one that requires it.
        model_name = str(type(self.model).__name__)
        needs_scaling = any(
            name in model_name
            for name in ['Logistic', 'SVC', 'KNeighbors', 'MLP', 'Linear']
        )
        model_input = input_scaled if needs_scaling else input_array
        
        # Make prediction
        prediction = self.model.predict(model_input)[0]
        
        # Get probability if available
        if hasattr(self.model, 'predict_proba'):
            probability = self.model.predict_proba(model_input)[0]
            confidence = max(probability)
        else:
            confidence = None
        
        return prediction, confidence
    
    def display_prediction_result(self, prediction, confidence):
        """Display prediction result with styling"""
        if prediction == 1:
            st.markdown(
                '<div class="prediction-result positive-prediction">'
                '⚠️ High Risk of Heart Disease'
                '</div>', 
                unsafe_allow_html=True
            )
            st.warning("The model indicates a high risk of heart disease. Please consult with a healthcare professional for proper medical evaluation.")
        else:
            st.markdown(
                '<div class="prediction-result negative-prediction">'
                '✅ Low Risk of Heart Disease'
                '</div>', 
                unsafe_allow_html=True
            )
            st.success("The model indicates a low risk of heart disease. Continue maintaining a healthy lifestyle!")
        
        if confidence:
            st.info(f"Model Confidence: {confidence:.2%}")

def main():
    """Main application function"""
    
    # Theme toggle in sidebar
    st.sidebar.markdown("### Settings")
    theme_option = st.sidebar.radio(
        "Theme",
        ["Dark Mode", "Light Mode"],
        index=0 if st.session_state.theme == 'dark' else 1
    )
    
    new_theme = 'dark' if theme_option == "Dark Mode" else 'light'
    if new_theme != st.session_state.theme:
        st.session_state.theme = new_theme
        st.rerun()
    
    st.markdown('<div class="main-header">Heart Disease Prediction</div>', unsafe_allow_html=True)
    
    app = HeartDiseaseApp()
    
    # Sidebar Navigation - Simplified
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Navigation")
    page = st.sidebar.selectbox(
        "Choose a page", 
        ["Prediction", "Data Analysis"],
        label_visibility="collapsed"
    )
    
    if page == "Prediction":
        st.markdown('<div class="sub-header">Make a Prediction</div>', unsafe_allow_html=True)
        st.markdown('<p style="color: #888; margin-bottom: 2rem;">Enter patient information below to assess cardiovascular risk.</p>', unsafe_allow_html=True)
        
        # Load model
        if not app.load_model():
            st.error("Model not found! Please train the model first by running the model_trainer.py script.")
            st.info("Run the following command in your terminal: `python src/model_trainer.py`")
            return
        
        # Create input form
        input_data = app.create_input_form()
        
        # Prediction button
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Analyze Risk", type="primary", use_container_width=True):
            prediction, confidence = app.make_prediction(input_data)
            
            if prediction is not None:
                st.markdown("---")
                app.display_prediction_result(prediction, confidence)
    
    elif page == "Data Analysis":
        st.markdown('<div class="sub-header">Dataset Analysis</div>', unsafe_allow_html=True)
        st.markdown('<p style="color: #888; margin-bottom: 2rem;">Explore the heart disease dataset with interactive visualizations.</p>', unsafe_allow_html=True)
        
        # Check if data file exists
        if os.path.exists(DATA_PATH):
            analyzer = HeartDiseaseAnalyzer(DATA_PATH)
            theme = app.get_chart_theme()
            
            if analyzer.load_data() is not None:
                # Basic info
                with st.expander("Dataset Overview", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Samples", analyzer.data.shape[0])
                    with col2:
                        st.metric("Features", analyzer.data.shape[1])
                    with col3:
                        st.metric("Target Classes", 2)
                    
                    # Target distribution
                    target_counts = analyzer.data['target'].value_counts()
                    fig = px.pie(values=target_counts.values, 
                               names=['No Disease', 'Heart Disease'],
                               title="Target Distribution",
                               template=theme['template'],
                               color_discrete_sequence=['#00ff88', '#ff4444'] if st.session_state.theme == 'dark' else ['#00aa66', '#cc0000'])
                    
                    fig.update_layout(
                        paper_bgcolor=theme['paper_bgcolor'],
                        font=dict(color=theme['font_color'], family='Inter'),
                        title_font=dict(size=20, color=theme['font_color']),
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Correlation heatmap
                with st.expander("Feature Correlations"):
                    corr_matrix = analyzer.data.corr()
                    fig = px.imshow(corr_matrix, 
                                  title="Feature Correlation Matrix",
                                  template=theme['template'],
                                  color_continuous_scale='RdYlGn' if st.session_state.theme == 'dark' else 'RdBu')
                    
                    fig.update_layout(
                        paper_bgcolor=theme['paper_bgcolor'],
                        font=dict(color=theme['font_color'], family='Inter'),
                        title_font=dict(size=20, color=theme['font_color']),
                        height=600
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Feature distributions
                with st.expander("Feature Distributions"):
                    selected_feature = st.selectbox("Select feature to analyze", 
                                                   analyzer.data.columns[:-1])
                    
                    fig = px.histogram(analyzer.data, x=selected_feature, color='target',
                                     title=f'{selected_feature} Distribution by Heart Disease Status',
                                     template=theme['template'],
                                     color_discrete_sequence=['#00ff88', '#ff4444'] if st.session_state.theme == 'dark' else ['#00aa66', '#cc0000'])
                    
                    fig.update_layout(
                        paper_bgcolor=theme['paper_bgcolor'],
                        plot_bgcolor=theme['plot_bgcolor'],
                        font=dict(color=theme['font_color'], family='Inter'),
                        title_font=dict(size=20, color=theme['font_color']),
                        xaxis=dict(gridcolor=theme['grid_color']),
                        yaxis=dict(gridcolor=theme['grid_color']),
                        height=450
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Dataset not found! Please ensure 'heart.csv' is in the data/ folder.")
    
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div class='footer'>" 
        "<p>Heart Disease Prediction System</p>"
        "<p style='font-size: 0.8rem; margin-top: 0.5rem;'>Built with using Streamlit & Machine Learning</p>"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
