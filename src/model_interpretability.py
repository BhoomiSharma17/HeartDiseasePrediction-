"""
Model Interpretability Module using SHAP and LIME
Provides explanations for model predictions and feature importance analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class ModelInterpreter:
    """Advanced model interpretability using SHAP values"""
    
    def __init__(self, model_path, scaler_path, data_path):
        """Initialize with model, scaler, and data paths"""
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.data_path = data_path
        self.model = None
        self.scaler = None
        self.df = None
        self.X_test = None
        self.y_test = None
        self.explainer = None
        self.shap_values = None
        
        self.feature_names = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]
        
        self.feature_descriptions = {
            'age': 'Age (years)',
            'sex': 'Sex (1=Male, 0=Female)',
            'cp': 'Chest Pain Type',
            'trestbps': 'Resting Blood Pressure',
            'chol': 'Cholesterol Level',
            'fbs': 'Fasting Blood Sugar > 120',
            'restecg': 'Resting ECG Results',
            'thalach': 'Max Heart Rate',
            'exang': 'Exercise Induced Angina',
            'oldpeak': 'ST Depression',
            'slope': 'ST Segment Slope',
            'ca': 'Major Vessels Count',
            'thal': 'Thalassemia Type'
        }
    
    def load_components(self):
        """Load model, scaler, and data"""
        try:
            # Load model and scaler
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            
            # Load and prepare data
            self.df = pd.read_csv(self.data_path)
            X = self.df.drop('target', axis=1)
            y = self.df['target']
            
            # Split data (same as training)
            _, self.X_test, _, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            print("✅ Model, scaler, and data loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading components: {str(e)}")
            return False
    
    def initialize_explainer(self):
        """Initialize SHAP explainer based on model type"""
        if self.model is None:
            print("Please load model first!")
            return False
        
        try:
            model_name = str(type(self.model).__name__)
            
            # Prepare data for explainer
            if any(name in model_name for name in ['Logistic', 'SVC', 'KNeighbors', 'MLP']):
                # Use scaled data for models that need scaling
                X_background = self.scaler.transform(self.X_test.iloc[:100])  # Sample for background
                X_explain = self.scaler.transform(self.X_test)
            else:

                X_background = self.X_test.iloc[:100].values
                X_explain = self.X_test.values
            
            if 'RandomForest' in model_name or 'XGB' in model_name or 'LightGBM' in model_name:
                self.explainer = shap.TreeExplainer(self.model)
                self.shap_values = self.explainer.shap_values(X_explain)
                
                # For binary classification, get positive class SHAP values
                if isinstance(self.shap_values, list):
                    self.shap_values = self.shap_values[1]
                    
            else:
                # Use KernelExplainer for other models
                self.explainer = shap.KernelExplainer(self.model.predict_proba, X_background)
                self.shap_values = self.explainer.shap_values(X_explain[:50])  # Limit for speed
                
                # For binary classification, get positive class SHAP values
                if isinstance(self.shap_values, list):
                    self.shap_values = self.shap_values[1]
                
                # Update X_test to match SHAP values dimensions
                self.X_test = self.X_test.iloc[:50]
            
            print(f"✅ SHAP explainer initialized for {model_name}")
            return True
            
        except Exception as e:
            print(f"❌ Error initializing explainer: {str(e)}")
            return False
    
    def plot_feature_importance(self):
        """Plot SHAP feature importance"""
        if self.shap_values is None:
            print("Please initialize explainer first!")
            return
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(self.shap_values, self.X_test, 
                         feature_names=self.feature_names, show=False)
        plt.title('SHAP Feature Importance Summary', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance_bar(self):
        """Plot SHAP feature importance as bar chart"""
        if self.shap_values is None:
            return
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(self.shap_values, self.X_test, 
                         feature_names=self.feature_names, 
                         plot_type="bar", show=False)
        plt.title('SHAP Feature Importance (Bar Chart)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def plot_waterfall_explanation(self, sample_idx=0):
        """Plot waterfall explanation for a specific prediction"""
        if self.shap_values is None:
            return
        
        try:
            # Create waterfall plot
            shap.waterfall_plot(
                shap.Explanation(
                    values=self.shap_values[sample_idx],
                    base_values=self.explainer.expected_value if hasattr(self.explainer, 'expected_value') else 0,
                    data=self.X_test.iloc[sample_idx].values,
                    feature_names=self.feature_names
                ),
                show=False
            )
            plt.title(f'SHAP Waterfall Plot - Sample {sample_idx}', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error creating waterfall plot: {str(e)}")
    
    def plot_force_plot(self, sample_idx=0):
        """Create force plot for individual prediction"""
        if self.shap_values is None:
            return
        
        try:
            expected_value = self.explainer.expected_value if hasattr(self.explainer, 'expected_value') else 0
            
            # Create force plot
            shap.force_plot(
                expected_value,
                self.shap_values[sample_idx],
                self.X_test.iloc[sample_idx],
                feature_names=self.feature_names,
                matplotlib=True,
                show=False
            )
            plt.title(f'SHAP Force Plot - Sample {sample_idx}', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error creating force plot: {str(e)}")
    
    def plot_dependence_plots(self, top_features=4):
        """Plot SHAP dependence plots for top features"""
        if self.shap_values is None:
            return
        
        # Get top features by mean absolute SHAP value
        mean_shap = np.abs(self.shap_values).mean(axis=0)
        top_indices = np.argsort(mean_shap)[-top_features:]
        top_feature_names = [self.feature_names[i] for i in top_indices]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('SHAP Dependence Plots - Top Features', fontsize=16, fontweight='bold')
        
        for i, (feature_idx, feature_name) in enumerate(zip(top_indices, top_feature_names)):
            row, col = i // 2, i % 2
            
            try:
                shap.dependence_plot(
                    feature_idx, self.shap_values, self.X_test,
                    feature_names=self.feature_names,
                    ax=axes[row, col], show=False
                )
                axes[row, col].set_title(f'{feature_name} Dependence')
                
            except Exception as e:
                axes[row, col].text(0.5, 0.5, f'Error: {str(e)}', 
                                   transform=axes[row, col].transAxes, ha='center')
        
        plt.tight_layout()
        plt.show()
    
    def analyze_individual_prediction(self, sample_data, sample_idx=None):
        """Analyze individual prediction with detailed explanation"""
        if self.model is None:
            return
        
        # Convert sample data to appropriate format
        if isinstance(sample_data, list):
            sample_array = np.array(sample_data).reshape(1, -1)
            sample_df = pd.DataFrame([sample_data], columns=self.feature_names)
        else:
            sample_array = sample_data.reshape(1, -1)
            sample_df = pd.DataFrame(sample_array, columns=self.feature_names)
        
        # Scale if necessary
        model_name = str(type(self.model).__name__)
        if any(name in model_name for name in ['Logistic', 'SVC', 'KNeighbors', 'MLP']):
            sample_scaled = self.scaler.transform(sample_array)
            prediction = self.model.predict(sample_scaled)[0]
            probability = self.model.predict_proba(sample_scaled)[0]
        else:
            prediction = self.model.predict(sample_array)[0]
            probability = self.model.predict_proba(sample_array)[0] if hasattr(self.model, 'predict_proba') else None
        
        print("\n" + "="*60)
        print("🔍 INDIVIDUAL PREDICTION ANALYSIS")
        print("="*60)
        
        print(f"\n📊 Prediction: {'Heart Disease' if prediction == 1 else 'No Heart Disease'}")
        if probability is not None:
            print(f"🎯 Confidence: {max(probability):.3f}")
            print(f"   No Disease: {probability[0]:.3f}")
            print(f"   Heart Disease: {probability[1]:.3f}")
        
        print(f"\n📋 Patient Information:")
        for feature, value in zip(self.feature_names, sample_data):
            description = self.feature_descriptions.get(feature, feature)
            print(f"   {description}: {value}")
        
        # Get SHAP explanation if available
        if self.explainer is not None:
            try:
                if any(name in model_name for name in ['Logistic', 'SVC', 'KNeighbors', 'MLP']):
                    shap_vals = self.explainer.shap_values(sample_scaled)
                else:
                    shap_vals = self.explainer.shap_values(sample_array)
                
                if isinstance(shap_vals, list):
                    shap_vals = shap_vals[1]  # Positive class
                
                # Show top contributing features
                feature_contributions = list(zip(self.feature_names, shap_vals[0]))
                feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
                
                print(f"\n🎯 Top Feature Contributions (SHAP values):")
                for i, (feature, contribution) in enumerate(feature_contributions[:5]):
                    direction = "increases" if contribution > 0 else "decreases"
                    print(f"   {i+1}. {self.feature_descriptions[feature]}: {contribution:.3f} ({direction} risk)")
                
            except Exception as e:
                print(f"   Could not generate SHAP explanation: {str(e)}")
        
        return prediction, probability
    
    def generate_global_insights(self):
        """Generate global model insights"""
        if self.shap_values is None:
            return
        
        print("\n" + "="*60)
        print("🌍 GLOBAL MODEL INSIGHTS")
        print("="*60)
        
        # Feature importance ranking
        mean_shap = np.abs(self.shap_values).mean(axis=0)
        feature_importance = list(zip(self.feature_names, mean_shap))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n📊 Feature Importance Ranking:")
        for i, (feature, importance) in enumerate(feature_importance):
            print(f"   {i+1:2d}. {self.feature_descriptions[feature]:25s}: {importance:.4f}")
        
        # Positive vs negative contributions
        positive_contrib = np.sum(self.shap_values > 0, axis=0)
        negative_contrib = np.sum(self.shap_values < 0, axis=0)
        total_samples = self.shap_values.shape[0]
        
        print(f"\n📈 Feature Impact Patterns:")
        for i, feature in enumerate(self.feature_names):
            pos_pct = (positive_contrib[i] / total_samples) * 100
            neg_pct = (negative_contrib[i] / total_samples) * 100
            print(f"   {self.feature_descriptions[feature]:25s}: {pos_pct:5.1f}% increase risk, {neg_pct:5.1f}% decrease risk")
        
        return feature_importance
    
    def create_comprehensive_report(self, output_file='model_interpretability_report.txt'):
        """Create comprehensive interpretability report"""
        if self.shap_values is None:
            return
        
        with open(output_file, 'w') as f:
            f.write("HEART DISEASE PREDICTION MODEL - INTERPRETABILITY REPORT\n")
            f.write("="*60 + "\n\n")
            
            # Model information
            f.write("MODEL INFORMATION:\n")
            f.write(f"Model Type: {type(self.model).__name__}\n")
            f.write(f"Dataset Size: {self.df.shape[0]} samples, {self.df.shape[1]-1} features\n")
            f.write(f"Test Set Size: {self.X_test.shape[0]} samples\n\n")
            
            # Feature importance
            mean_shap = np.abs(self.shap_values).mean(axis=0)
            feature_importance = list(zip(self.feature_names, mean_shap))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            f.write("FEATURE IMPORTANCE RANKING:\n")
            for i, (feature, importance) in enumerate(feature_importance):
                f.write(f"{i+1:2d}. {self.feature_descriptions[feature]:25s}: {importance:.4f}\n")
            
            f.write("\nFEATURE DESCRIPTIONS:\n")
            for feature, description in self.feature_descriptions.items():
                f.write(f"{feature:12s}: {description}\n")
            
            # Statistical insights
            f.write("\nSTATISTICAL INSIGHTS:\n")
            for i, feature in enumerate(self.feature_names):
                pos_contrib = np.sum(self.shap_values[:, i] > 0)
                neg_contrib = np.sum(self.shap_values[:, i] < 0)
                total = self.shap_values.shape[0]
                
                f.write(f"{self.feature_descriptions[feature]:25s}:\n")
                f.write(f"  - Increases risk in {pos_contrib/total*100:5.1f}% of cases\n")
                f.write(f"  - Decreases risk in {neg_contrib/total*100:5.1f}% of cases\n")
                f.write(f"  - Mean absolute impact: {mean_shap[i]:.4f}\n\n")
        
        print(f"✅ Comprehensive report saved to {output_file}")

def main():
    """Main function to run interpretability analysis"""
    # Initialize interpreter
    interpreter = ModelInterpreter(
        'models/best_heart_disease_model.pkl',
        'models/scaler.pkl',
        'data/heart.csv'
    )
    
    # Load components and initialize explainer
    if interpreter.load_components():
        if interpreter.initialize_explainer():
            # Create visualizations
            interpreter.plot_feature_importance()
            interpreter.plot_feature_importance_bar()
            interpreter.plot_dependence_plots()
            
            # Generate insights
            interpreter.generate_global_insights()
            
            # Analyze sample predictions
            print("\n" + "="*60)
            print("🔍 SAMPLE PREDICTION ANALYSIS")
            print("="*60)
            
            # Analyze a few test samples
            for i in range(min(3, len(interpreter.X_test))):
                sample_data = interpreter.X_test.iloc[i].values
                interpreter.analyze_individual_prediction(sample_data, i)
                interpreter.plot_waterfall_explanation(i)
            
            # Create comprehensive report
            interpreter.create_comprehensive_report()

if __name__ == "__main__":
    main()
