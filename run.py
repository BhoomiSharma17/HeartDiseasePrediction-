"""
Single Command Startup Script for Heart Disease Prediction System
Run this script to start the Streamlit web application
"""

import os
import sys
import subprocess

def main():
    """Start the Heart Disease Prediction web application"""
    
    print("=" * 60)
    print("❤️  HEART DISEASE PREDICTION SYSTEM")
    print("=" * 60)
    print("\n🚀 Starting web application...\n")
    
    # Check if streamlit is installed
    try:
        import streamlit
    except ImportError:
        print("❌ Streamlit not found!")
        print("📦 Installing dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Check if model exists
    model_path = os.path.join("models", "best_heart_disease_model.pkl")
    if not os.path.exists(model_path):
        print("⚠️  Model not found!")
        print("🔧 Training model first...\n")
        
        # Add src to path
        sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
        
        try:
            from model_trainer import HeartDiseaseModelTrainer
            
            trainer = HeartDiseaseModelTrainer('data/heart.csv')
            if trainer.load_and_prepare_data():
                trainer.train_model_with_tuning()
                results = trainer.evaluate_model()
                trainer.generate_classification_report(results['predictions'])
                trainer.save_model()
                print("\n✅ Model trained and saved!")
        except Exception as e:
            print(f"❌ Error training model: {str(e)}")
            print("Please run: python src/model_trainer.py")
            return
    
    # Start Streamlit app
    print("\n🌐 Launching web application...")
    print("=" * 60)
    
    app_path = os.path.join("app", "streamlit_app.py")
    subprocess.run([sys.executable, "-m", "streamlit", "run", app_path])

if __name__ == "__main__":
    main()
