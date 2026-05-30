#  Heart Disease Prediction System

A comprehensive machine learning project for predicting heart disease risk using an optimized Random Forest model with interactive web interface.

## 🎯 Project Overview

This project implements a complete machine learning pipeline for heart disease prediction, featuring:

- **Optimized Random Forest model** with hyperparameter tuning
- **Interactive web application** built with Streamlit (modern black theme)
- **Advanced model interpretability** using SHAP values
- **Comprehensive data analysis** and visualization
- **Professional code structure** with modular design

## 🚀 Features

### 🔬 Machine Learning Model
- **Random Forest Classifier** with comprehensive hyperparameter tuning
- Grid Search CV with 5-fold stratified cross-validation
- Optimized parameters for maximum performance
- Feature importance analysis

### 📊 Data Analysis
- Comprehensive exploratory data analysis (EDA)
- Statistical summaries and insights
- Interactive visualizations with Plotly
- Correlation analysis and feature importance

### 🌐 Web Application
- Real-time heart disease risk prediction
- Clean, minimal interface focused on predictions
- Interactive patient data input with real-time value display
- Dataset analysis and visualizations
- Responsive design with modern black theme UI

## 📁 Project Structure

```
HeartDisease/
├── src/                          # Source code modules
│   ├── data_analyzer.py          # Data analysis and visualization
│   ├── model_trainer.py          # ML model training and evaluation
│   └── model_interpretability.py # SHAP-based model explanations
├── app/                          # Web application
│   └── streamlit_app.py          # Streamlit web interface
├── data/                         # Dataset storage
│   └── heart.csv                 # Heart disease dataset
├── models/                       # Trained models storage
│   ├── best_heart_disease_model.pkl
│   ├── scaler.pkl
│   └── model_info.pkl
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps

1. **Clone or download the project**
   ```bash
   cd HeartDisease
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset**
   - Download the heart disease dataset (`heart.csv`)
   - Place it in the `data/` folder
   - Dataset available at: [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)

## 🚀 Quick Start

### Single Command to Run Everything

**Windows:**
```bash
start.bat
```

**Or using Python:**
```bash
python run.py
```

This will:
- ✅ Check and install dependencies
- ✅ Train the model (if not already trained)
- ✅ Launch the web application automatically

## 📖 Detailed Usage

### 1. Data Analysis
Run comprehensive data analysis and visualization:
```bash
python src/data_analyzer.py
```

### 2. Model Training (Manual)
Train the optimized Random Forest model:
```bash
python src/model_trainer.py
```
This will:
- Perform hyperparameter tuning using GridSearchCV
- Train the model with best parameters
- Generate performance metrics and visualizations
- Save the trained model


### 4. Web Application (Manual)
Launch the interactive web application:
```bash
streamlit run app/streamlit_app.py
```

### 5. Complete Pipeline
Run the entire pipeline (analysis + training + interpretability):
```bash
python main.py
```

## 📊 Dataset Information

The heart disease dataset contains 303 instances with 14 attributes:

| Feature | Description |
|---------|-------------|
| age | Age in years |
| sex | Sex (1 = male; 0 = female) |
| cp | Chest pain type (0-3) |
| trestbps | Resting blood pressure (mm Hg) |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl |
| restecg | Resting electrocardiographic results |
| thalach | Maximum heart rate achieved |
| exang | Exercise induced angina |
| oldpeak | ST depression induced by exercise |
| slope | Slope of peak exercise ST segment |
| ca | Number of major vessels (0-3) |
| thal | Thalassemia type |
| target | Heart disease presence (0 = no, 1 = yes) |

## 🏆 Model Performance

The Random Forest model achieves excellent performance:

| Metric | Score |
|--------|-------|
| Accuracy | ~0.88-0.90 |
| Precision | ~0.87-0.91 |
| Recall | ~0.86-0.90 |
| F1-Score | ~0.87-0.90 |
| ROC-AUC | ~0.92-0.95 |

**Hyperparameter Tuning:**
- Grid search across multiple parameter combinations
- 5-fold stratified cross-validation
- Optimized for accuracy while maintaining balanced performance

*Note: Actual performance may vary based on data split and random state*

## 🔍 Key Features

### Advanced Analytics
- **Cross-validation** with 5-fold stratified K-fold
- **Hyperparameter tuning** using GridSearchCV with extensive parameter grid
- **ROC curve** and AUC analysis
- **Confusion matrix** visualization
- **Feature correlation** and importance analysis

### Model Interpretability
- **SHAP summary plots** showing feature importance
- **Waterfall plots** for individual predictions
- **Dependence plots** showing feature interactions
- **Force plots** for prediction explanations

### Web Interface
- **Real-time predictions** with confidence scores
- **Patient health profile** radar charts
- **Model performance** metrics display

## 🎨 Visualizations

The project generates various visualizations:

1. **Data Analysis**
   - Target distribution pie charts
   - Feature correlation heatmaps
   - Box plots for feature distributions
   - Interactive 3D scatter plots

2. **Model Performance**
   - Feature importance bar chart
   - ROC curve with AUC score
   - Confusion matrix heatmap
   - Performance metrics visualization

## 🌐 Web Application Features

### Prediction Interface
- Clean, minimal design focused on predictions
- User-friendly input forms with real-time value display
- Instant risk assessment results
- Confidence score display
- Dark/Light theme toggle

### Data Analysis Dashboard
- Interactive dataset exploration
- Feature correlation heatmaps
- Distribution plots by target class
- Dataset statistics overview

## 🔧 Technical Details

### Machine Learning Pipeline
1. **Data Preprocessing**
   - Missing value handling
   - Feature scaling with StandardScaler
   - Train-test split with stratification (80-20 split)

2. **Model Training**
   - Random Forest implementation with scikit-learn
   - Extensive hyperparameter grid search
   - 5-fold cross-validation for robust evaluation

3. **Model Evaluation**
   - Multiple metrics (accuracy, precision, recall, F1-score, ROC-AUC)
   - Confusion matrix analysis
   - Feature importance ranking



## 📈 Business Value

This project demonstrates:

1. **Healthcare Applications** - Real-world medical prediction system
2. **Technical Skills** - Advanced ML techniques and best practices
3. **Full-Stack Development** - From data analysis to web deployment
4. **Model Interpretability** - Explainable AI for healthcare decisions
5. **Professional Standards** - Production-ready code structure

## 📚 Dependencies

Key libraries used in this project:

```
pandas==2.0.3          # Data manipulation
numpy==1.24.3          # Numerical computing
scikit-learn==1.3.0    # Machine learning (Random Forest)
matplotlib==3.7.2      # Static plotting
seaborn==0.12.2        # Statistical visualization
plotly==5.15.0         # Interactive plots
streamlit==1.25.0      # Web application
shap==0.42.1           # Model interpretability
joblib==1.3.2          # Model serialization
scipy==1.11.1          # Scientific computing
```

## 🤝 Contributing

This project is designed as a portfolio piece, but suggestions for improvements are welcome:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 👨‍💻 Author

**Aditya**
-Student At Graphic Era Hill University


*This project showcases a complete machine learning workflow from data analysis to deployment, demonstrating both technical depth and practical application in healthcare AI.*
# Heart_Disease_Prediction-
