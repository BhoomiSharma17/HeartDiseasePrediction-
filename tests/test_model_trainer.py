"""
Unit tests for model_trainer.py
"""
import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import sys
import os
import tempfile

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model_trainer import HeartDiseaseModelTrainer


class TestHeartDiseaseModelTrainer:
    """Test cases for HeartDiseaseModelTrainer class"""
    
    @pytest.fixture
    def sample_csv(self, tmp_path):
        """Create a temporary CSV file with sample data for testing"""
        np.random.seed(42)
        data = {
            'age': np.random.randint(30, 80, 100),
            'sex': np.random.randint(0, 2, 100),
            'cp': np.random.randint(0, 4, 100),
            'trestbps': np.random.randint(90, 200, 100),
            'chol': np.random.randint(100, 400, 100),
            'fbs': np.random.randint(0, 2, 100),
            'restecg': np.random.randint(0, 3, 100),
            'thalach': np.random.randint(60, 200, 100),
            'exang': np.random.randint(0, 2, 100),
            'oldpeak': np.random.uniform(0, 6, 100),
            'slope': np.random.randint(0, 3, 100),
            'ca': np.random.randint(0, 4, 100),
            'thal': np.random.randint(0, 4, 100),
            'target': np.random.randint(0, 2, 100)
        }
        csv_path = tmp_path / "heart_test.csv"
        pd.DataFrame(data).to_csv(csv_path, index=False)
        return str(csv_path)
    
    @pytest.fixture
    def trainer(self, sample_csv):
        """Create trainer instance with sample CSV path"""
        return HeartDiseaseModelTrainer(sample_csv)
    
    def test_initialization(self, trainer):
        """Test if trainer initializes correctly"""
        assert trainer.data is not None
        assert len(trainer.data) == 100
        assert trainer.model is None
    
    def test_data_split(self, trainer):
        """Test if data is split correctly"""
        assert trainer.X_train is not None
        assert trainer.X_test is not None
        assert trainer.y_train is not None
        assert trainer.y_test is not None
        assert len(trainer.X_train) + len(trainer.X_test) == len(trainer.data)
    
    def test_model_training(self, trainer):
        """Test if model trains without errors"""
        # Use a simple model for testing (no hyperparameter tuning)
        trainer.model = RandomForestClassifier(n_estimators=10, random_state=42)
        trainer.model.fit(trainer.X_train_scaled, trainer.y_train)
        
        assert trainer.model is not None
        assert hasattr(trainer.model, 'predict')
    
    def test_model_evaluation(self, trainer):
        """Test if model evaluation returns correct metrics"""
        # Train a simple model first
        trainer.model = RandomForestClassifier(n_estimators=10, random_state=42)
        trainer.model.fit(trainer.X_train_scaled, trainer.y_train)
        
        results = trainer.evaluate_model()
        
        assert 'accuracy' in results
        assert 'precision' in results
        assert 'recall' in results
        assert 'f1_score' in results
        assert 'roc_auc' in results
        
        # Check if metrics are in valid range [0, 1]
        for metric, value in results.items():
            if metric not in ('predictions', 'probabilities'):
                assert 0 <= value <= 1, f"{metric} should be between 0 and 1"
    
    def test_feature_scaling(self, trainer):
        """Test if features are scaled correctly"""
        assert trainer.X_train_scaled is not None
        assert trainer.X_test_scaled is not None
        
        # Check if scaled data has mean close to 0 and std close to 1
        mean = np.mean(trainer.X_train_scaled, axis=0)
        std = np.std(trainer.X_train_scaled, axis=0)
        
        assert np.allclose(mean, 0, atol=1e-7)
        assert np.allclose(std, 1, atol=1e-7)


def test_data_loading():
    """Test if data can be loaded from CSV"""
    # This test assumes the data file exists
    data_path = 'data/heart.csv'
    if os.path.exists(data_path):
        data = pd.read_csv(data_path)
        assert len(data) > 0
        assert 'target' in data.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
