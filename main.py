import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_analyzer import HeartDiseaseAnalyzer
from model_trainer import HeartDiseaseModelTrainer
from model_interpretability import ModelInterpreter

def main():
    """Execute the complete heart disease prediction pipeline"""
    
    print(" HEART DISEASE PREDICTION SYSTEM")
    print("=" * 50)
    print("Complete Machine Learning Pipeline")
    print("=" * 50)
    
    # Check if data file exists
    data_path = 'data/heart.csv'
    if not os.path.exists(data_path):
        print(" Dataset not found!")
        print("Please download 'heart.csv' and place it in the 'data/' folder")
        print("Dataset available at: https://archive.ics.uci.edu/ml/datasets/Heart+Disease")
        return
    
    try:
    
        print("\n STEP 1: DATA ANALYSIS")
        print("-" * 30)
        analyzer = HeartDiseaseAnalyzer(data_path)
        
        if analyzer.load_data() is not None:
            analyzer.basic_info()
            analyzer.statistical_summary()
            analyzer.create_visualizations()
            analyzer.feature_importance_analysis()
            analyzer.generate_insights()
            print("✅ Data analysis completed!")
        
        
        print("\n STEP 2: MODEL TRAINING")
        print("-" * 30)
        trainer = HeartDiseaseModelTrainer(data_path)
        
        if trainer.load_and_prepare_data():
            
            trainer.train_model_with_tuning()
            
            results = trainer.evaluate_model()
            
            if results is not None and isinstance(results, dict) and 'predictions' in results:
                trainer.generate_classification_report(results['predictions'])
                trainer.create_visualizations(results)
                trainer.save_model()
                print(" Model training completed! Model: Random Forest")
            else:
                print(" No evaluation results available. Skipping report generation and visualization.")
                return
        
        
        print("\n🔍 STEP 3: MODEL INTERPRETABILITY")
        print("-" * 30)
        
        if os.path.exists('models/best_heart_disease_model.pkl'):
            interpreter = ModelInterpreter(
                'models/best_heart_disease_model.pkl',
                'models/scaler.pkl',
                data_path
            )
            
            if interpreter.load_components():
                if interpreter.initialize_explainer():
                    interpreter.plot_feature_importance()
                    interpreter.plot_feature_importance_bar()
                    interpreter.plot_dependence_plots()
                    interpreter.generate_global_insights()
                    interpreter.create_comprehensive_report()
                    print(" Model interpretability analysis completed!")
        
        # Final Summary
        print("\n PROJECT COMPLETION SUMMARY")
        print("=" * 50)
        print(" Data analysis and visualization")
        print(" Random Forest model trained with hyperparameter tuning")
        print(" Model saved with performance metrics")
        print(" Model interpretability analysis with SHAP")
        print(" Comprehensive documentation generated")
        print("\n Next Steps:")
        print("   • Run 'streamlit run app/streamlit_app.py' for web interface")
        print("   • Check 'models/' folder for saved models")
        print("   • Review 'model_interpretability_report.txt' for insights")
        
    except Exception as e:
        print(f" Error during execution: {str(e)}")
        print("Please check the error message and ensure all dependencies are installed.")

if __name__ == "__main__":
    main()
