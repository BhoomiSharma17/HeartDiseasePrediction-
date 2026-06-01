"""
Unit tests for data_analyzer.py
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_analyzer import HeartDiseaseAnalyzer


class TestHeartDiseaseAnalyzer:
    """Test cases for HeartDiseaseAnalyzer class"""
    
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
    def analyzer(self, sample_csv):
        """Create analyzer instance with sample CSV path"""
        return HeartDiseaseAnalyzer(sample_csv)
    
    def test_initialization(self, analyzer):
        """Test if analyzer initializes correctly"""
        assert analyzer.data is not None
        assert len(analyzer.data) == 100
    
    def test_basic_info(self, analyzer, capsys):
        """Test if basic info is displayed"""
        analyzer.display_basic_info()
        captured = capsys.readouterr()
        assert "Dataset Shape" in captured.out or len(captured.out) > 0
    
    def test_statistical_summary(self, analyzer):
        """Test if statistical summary is generated"""
        summary = analyzer.data.describe()
        assert summary is not None
        assert len(summary) > 0
    
    def test_correlation_analysis(self, analyzer):
        """Test if correlation matrix is generated"""
        corr_matrix = analyzer.data.corr()
        assert corr_matrix is not None
        assert corr_matrix.shape[0] == corr_matrix.shape[1]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
