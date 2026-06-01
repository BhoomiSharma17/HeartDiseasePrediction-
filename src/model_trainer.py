"""
Optimized Random Forest Model Training Module for Heart Disease Prediction
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           roc_auc_score, confusion_matrix, classification_report, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

class HeartDiseaseModelTrainer:
    """Optimized Random Forest model training for heart disease prediction"""
    
    def __init__(self, data_path):
        """Initialize with dataset path"""
        self.data_path = data_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.model = None
        self.best_params = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        try:
            self.load_and_prepare_data()
        except Exception as e:
            print(f"Warning: Could not load data during initialization: {e}")
          
    def load_and_prepare_data(self):
        """Load and prepare data for training"""
        try:
            self.data = pd.read_csv(self.data_path)
            print("✅ Data loaded successfully!")
            print(f"Dataset shape: {self.data.shape}")
            
            # Separate features and target
            X = self.data.drop('target', axis=1)
            y = self.data['target']
            
            # Split the data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale the features
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)
            self.X_test_scaled = self.scaler.transform(self.X_test)
            
            print(f"Training set size: {self.X_train.shape[0]}")
            print(f"Test set size: {self.X_test.shape[0]}")
            
            return True
            
        except FileNotFoundError:
            print("❌ Dataset not found. Please ensure 'heart.csv' is in the data folder.")
            return False
    
    def train_model_with_tuning(self):
        """Train Random Forest model with hyperparameter tuning"""
        if self.X_train is None:
            print("Please load and prepare data first!")
            return
        
        print("\n" + "="*60)
        print("🚀 TRAINING RANDOM FOREST MODEL")
        print("="*60)
        
        # Define hyperparameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2'],
            'bootstrap': [True, False]
        }
        
        print("\n🔧 Performing hyperparameter tuning...")
        print("This may take a few minutes...\n")
        
        # Initialize Random Forest
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        # Perform grid search with cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid_search = GridSearchCV(
            rf, param_grid, cv=cv, scoring='accuracy', 
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        # Get best model
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        
        print("\n✅ Training completed!")
        print(f"\n📊 Best Parameters:")
        for param, value in self.best_params.items():
            print(f"   {param}: {value}")
        print(f"\n🎯 Best Cross-Validation Score: {grid_search.best_score_:.4f}")
    
    def evaluate_model(self):
        """Evaluate the trained model on test data"""
        if self.model is None:
            print("Please train the model first!")
            return
        
        print("\n" + "="*60)
        print("📊 MODEL EVALUATION")
        print("="*60)
        
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        
        print(f"\n✅ Test Set Performance:")
        print(f"   Accuracy:  {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1-Score:  {f1:.4f}")
        print(f"   ROC-AUC:   {roc_auc:.4f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    
    def plot_feature_importance(self):
        """Plot feature importance from the trained model"""
        if self.model is None:
            print("Please train the model first!")
            return
        
        feature_names = self.data.drop('target', axis=1).columns
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.title('Feature Importance - Random Forest', fontsize=16, fontweight='bold')
        plt.bar(range(len(importances)), importances[indices], color='#00ff88', alpha=0.7)
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.xlabel('Features', fontsize=12)
        plt.ylabel('Importance Score', fontsize=12)
        plt.tight_layout()
        plt.grid(axis='y', alpha=0.3)
        plt.show()
        
        print("\n📊 Top 5 Most Important Features:")
        for i in range(min(5, len(importances))):
            print(f"   {i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
    
    def plot_confusion_matrix(self, predictions):
        """Plot confusion matrix for the model"""
        cm = confusion_matrix(self.y_test, predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                   xticklabels=['No Disease', 'Heart Disease'],
                   yticklabels=['No Disease', 'Heart Disease'],
                   cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix - Random Forest', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curve(self, probabilities):
        """Plot ROC curve for the model"""
        fpr, tpr, thresholds = roc_curve(self.y_test, probabilities)
        roc_auc = roc_auc_score(self.y_test, probabilities)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='#00ff88', lw=2, label=f'Random Forest (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve - Random Forest', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def generate_classification_report(self, predictions):
        """Generate detailed classification report"""
        print("\n" + "="*60)
        print("📋 CLASSIFICATION REPORT")
        print("="*60)
        print(classification_report(self.y_test, predictions, 
                                  target_names=['No Disease', 'Heart Disease']))
    
    def save_model(self):
        """Save the trained model and scaler"""
        if self.model is None:
            print("No model trained yet!")
            return
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Save model and scaler
        joblib.dump(self.model, 'models/best_heart_disease_model.pkl')
        joblib.dump(self.scaler, 'models/scaler.pkl')
        
        # Evaluate and save model info
        results = self.evaluate_model()
        
        model_info = {
            'model_name': 'Random Forest',
            'accuracy': results['accuracy'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1_score': results['f1_score'],
            'roc_auc': results['roc_auc'],
            'best_params': self.best_params
        }
        
        joblib.dump(model_info, 'models/model_info.pkl')
        
        print("\n" + "="*60)
        print("💾 MODEL SAVED SUCCESSFULLY")
        print("="*60)
        print(f"Model: Random Forest")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Location: models/best_heart_disease_model.pkl")
        
        return results
    
    def create_visualizations(self, results):
        """Create all visualizations for the model"""
        print("\n📊 Generating visualizations...")
        
        # Feature importance
        self.plot_feature_importance()
        
        # Confusion matrix
        self.plot_confusion_matrix(results['predictions'])
        
        # ROC curve
        self.plot_roc_curve(results['probabilities'])
        
        print("✅ All visualizations generated!")
    
    

def main():
    """Main function to run model training"""
    print("\n" + "="*60)
    print("HEART DISEASE PREDICTION - MODEL TRAINING")
    print("="*60)
    
    # Initialize trainer
    trainer = HeartDiseaseModelTrainer('data/heart.csv')
    
    # Load and prepare data
    if trainer.load_and_prepare_data():
        # Train model with hyperparameter tuning
        trainer.train_model_with_tuning()
        
        # Evaluate model
        results = trainer.evaluate_model()
        
        # Generate classification report
        trainer.generate_classification_report(results['predictions'])
        
        # Create visualizations
        trainer.create_visualizations(results)
        
        # Save model
        trainer.save_model()
        
        print("\n" + "="*60)
        print("✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)

if __name__ == "__main__":
    main()
