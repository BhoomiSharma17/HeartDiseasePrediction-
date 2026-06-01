"""
Heart Disease Data Analysis and Visualization Module
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class HeartDiseaseAnalyzer:
    """Comprehensive data analysis and visualization for heart disease dataset"""
    
    def __init__(self, data_path):
        """Initialize with dataset path"""
        self.data_path = data_path
        self.data = None
        self.feature_descriptions = {
            'age': 'Age in years',
            'sex': 'Sex (1 = male; 0 = female)',
            'cp': 'Chest pain type (0-3)',
            'trestbps': 'Resting blood pressure (mm Hg)',
            'chol': 'Serum cholesterol (mg/dl)',
            'fbs': 'Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)',
            'restecg': 'Resting electrocardiographic results (0-2)',
            'thalach': 'Maximum heart rate achieved',
            'exang': 'Exercise induced angina (1 = yes; 0 = no)',
            'oldpeak': 'ST depression induced by exercise',
            'slope': 'Slope of peak exercise ST segment (0-2)',
            'ca': 'Number of major vessels colored by fluoroscopy (0-3)',
            'thal': 'Thalassemia (1 = normal; 2 = fixed defect; 3 = reversable defect)',
            'target': 'Heart disease presence (1 = disease; 0 = no disease)'
        }
        try:
            self.load_data()
        except Exception as e:
            print(f"Warning: Could not load data during initialization: {e}")

    
    def display_basic_info(self):
     return self.basic_info()
    
    def load_data(self):
        """Load and perform initial data inspection"""
        try:
            self.data = pd.read_csv(self.data_path)
            print("✅ Data loaded successfully!")
            print(f"Dataset shape: {self.data.shape}")
            return self.data
        except FileNotFoundError:
            print("❌ Heart disease dataset not found. Please ensure 'heart.csv' is in the data folder.")
            return None
    
    def basic_info(self):
        """Display basic dataset information"""
        if self.data is None:
            print("Please load data first using load_data()")
            return
        
        print("\n" + "="*60)
        print("📊 DATASET OVERVIEW")
        print("="*60)
        
        print(f"Dataset Shape: {self.data.shape}")
        print(f"Features: {self.data.shape[1]}")
        print(f"Samples: {self.data.shape[0]}")
        
        print("\n📋 Feature Information:")
        for col, desc in self.feature_descriptions.items():
            if col in self.data.columns:
                print(f"  • {col}: {desc}")
        
        print("\n🔍 Data Types:")
        print(self.data.dtypes)
        
        print("\n❓ Missing Values:")
        missing = self.data.isnull().sum()
        if missing.sum() == 0:
            print("  ✅ No missing values found!")
        else:
            print(missing[missing > 0])
        
        print("\n📈 Target Distribution:")
        target_counts = self.data['target'].value_counts()
        print(f"  • No Disease (0): {target_counts[0]} ({target_counts[0]/len(self.data)*100:.1f}%)")
        print(f"  • Disease (1): {target_counts[1]} ({target_counts[1]/len(self.data)*100:.1f}%)")
    
    def statistical_summary(self):
        """Generate comprehensive statistical summary"""
        if self.data is None:
            return
        
        print("\n" + "="*60)
        print("📊 STATISTICAL SUMMARY")
        print("="*60)
        
        # Overall statistics
        print("\n🔢 Descriptive Statistics:")
        print(self.data.describe().round(2))
        
        # Statistics by target
        print("\n🎯 Statistics by Heart Disease Status:")
        for target in [0, 1]:
            status = "No Disease" if target == 0 else "Heart Disease"
            print(f"\n{status}:")
            subset = self.data[self.data['target'] == target]
            print(subset.describe().round(2))
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        if self.data is None:
            return
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Target Distribution
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Heart Disease Dataset - Overview Analysis', fontsize=16, fontweight='bold')
        
        # Target distribution
        target_counts = self.data['target'].value_counts()
        axes[0,0].pie(target_counts.values, labels=['No Disease', 'Heart Disease'], 
                     autopct='%1.1f%%', startangle=90, colors=['lightcoral', 'lightblue'])
        axes[0,0].set_title('Target Distribution')
        
        # Age distribution by target
        sns.histplot(data=self.data, x='age', hue='target', bins=20, ax=axes[0,1])
        axes[0,1].set_title('Age Distribution by Heart Disease Status')
        
        # Chest pain types
        cp_counts = self.data.groupby(['cp', 'target']).size().unstack()
        cp_counts.plot(kind='bar', ax=axes[1,0], color=['lightcoral', 'lightblue'])
        axes[1,0].set_title('Chest Pain Types by Heart Disease')
        axes[1,0].set_xlabel('Chest Pain Type')
        axes[1,0].legend(['No Disease', 'Heart Disease'])
        
        # Max heart rate vs age
        scatter = axes[1,1].scatter(self.data['age'], self.data['thalach'], 
                                   c=self.data['target'], alpha=0.6, cmap='coolwarm')
        axes[1,1].set_xlabel('Age')
        axes[1,1].set_ylabel('Max Heart Rate')
        axes[1,1].set_title('Max Heart Rate vs Age')
        plt.colorbar(scatter, ax=axes[1,1])
        
        plt.tight_layout()
        plt.show()
        
        # 2. Correlation Heatmap
        plt.figure(figsize=(12, 10))
        correlation_matrix = self.data.corr()
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, fmt='.2f')
        plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # 3. Feature distributions
        numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Feature Distributions by Heart Disease Status', fontsize=16, fontweight='bold')
        
        for i, feature in enumerate(numeric_features):
            row, col = i // 3, i % 3
            sns.boxplot(data=self.data, x='target', y=feature, ax=axes[row, col])
            axes[row, col].set_title(f'{feature.capitalize()} Distribution')
            axes[row, col].set_xlabel('Heart Disease (0=No, 1=Yes)')
        
        # Remove empty subplot
        axes[1, 2].remove()
        
        plt.tight_layout()
        plt.show()
    
    def create_interactive_plots(self):
        """Create interactive plotly visualizations"""
        if self.data is None:
            return
        
        # 1. Interactive correlation heatmap
        corr_matrix = self.data.corr()
        fig_heatmap = px.imshow(corr_matrix, 
                               title="Interactive Correlation Heatmap",
                               color_continuous_scale='RdBu',
                               aspect="auto")
        fig_heatmap.show()
        
        # 2. 3D scatter plot
        fig_3d = px.scatter_3d(self.data, x='age', y='thalach', z='chol',
                              color='target', 
                              title="3D Scatter: Age vs Max Heart Rate vs Cholesterol",
                              labels={'target': 'Heart Disease'})
        fig_3d.show()
        
        # 3. Interactive feature distributions
        numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        
        for col in numeric_cols:
            fig = px.histogram(self.data, x=col, color='target', 
                             title=f'{col.capitalize()} Distribution by Heart Disease Status',
                             marginal="box")
            fig.show()
    
    def feature_importance_analysis(self):
        """Analyze feature importance using correlation with target"""
        if self.data is None:
            return
        
        # Calculate correlation with target
        correlations = self.data.corr()['target'].abs().sort_values(ascending=False)
        correlations = correlations.drop('target')  # Remove target itself
        
        print("\n" + "="*60)
        print("🎯 FEATURE IMPORTANCE ANALYSIS")
        print("="*60)
        
        print("\n📊 Features ranked by correlation with target:")
        for i, (feature, corr) in enumerate(correlations.items(), 1):
            print(f"{i:2d}. {feature:12s}: {corr:.3f}")
        
        # Visualize feature importance
        plt.figure(figsize=(10, 8))
        correlations.plot(kind='barh', color='skyblue')
        plt.title('Feature Importance (Correlation with Target)', fontweight='bold')
        plt.xlabel('Absolute Correlation with Heart Disease')
        plt.tight_layout()
        plt.show()
        
        return correlations
    
    def generate_insights(self):
        """Generate key insights from the data"""
        if self.data is None:
            return
        
        print("\n" + "="*60)
        print("💡 KEY INSIGHTS")
        print("="*60)
        
        # Age insights
        avg_age_disease = self.data[self.data['target'] == 1]['age'].mean()
        avg_age_no_disease = self.data[self.data['target'] == 0]['age'].mean()
        print(f"\n👥 Age Analysis:")
        print(f"  • Average age with heart disease: {avg_age_disease:.1f} years")
        print(f"  • Average age without heart disease: {avg_age_no_disease:.1f} years")
        
        # Gender insights
        gender_disease = pd.crosstab(self.data['sex'], self.data['target'], normalize='index') * 100
        print(f"\n⚥ Gender Analysis:")
        print(f"  • Males with heart disease: {gender_disease.loc[1, 1]:.1f}%")
        print(f"  • Females with heart disease: {gender_disease.loc[0, 1]:.1f}%")
        
        # Chest pain insights
        cp_disease = self.data.groupby('cp')['target'].mean() * 100
        print(f"\n💔 Chest Pain Analysis:")
        for cp_type, percentage in cp_disease.items():
            print(f"  • Chest pain type {cp_type}: {percentage:.1f}% have heart disease")
        
        # Heart rate insights
        hr_disease = self.data[self.data['target'] == 1]['thalach'].mean()
        hr_no_disease = self.data[self.data['target'] == 0]['thalach'].mean()
        print(f"\n❤️ Heart Rate Analysis:")
        print(f"  • Average max heart rate with disease: {hr_disease:.1f} bpm")
        print(f"  • Average max heart rate without disease: {hr_no_disease:.1f} bpm")
        
        return {
            'age_with_disease': avg_age_disease,
            'age_without_disease': avg_age_no_disease,
            'gender_analysis': gender_disease,
            'chest_pain_analysis': cp_disease,
            'heart_rate_with_disease': hr_disease,
            'heart_rate_without_disease': hr_no_disease
        }

def main():
    """Main function to run the analysis"""
    # Initialize analyzer
    analyzer = HeartDiseaseAnalyzer('data/heart.csv')
    
    # Load and analyze data
    if analyzer.load_data() is not None:
        analyzer.basic_info()
        analyzer.statistical_summary()
        analyzer.create_visualizations()
        analyzer.create_interactive_plots()
        analyzer.feature_importance_analysis()
        analyzer.generate_insights()
    
if __name__ == "__main__":
    main()
