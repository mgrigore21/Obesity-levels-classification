import pytest
from ml import ClassificationPipeline

class TestClassificationPipeline:
    @pytest.fixture
    def pipeline(self):
        return ClassificationPipeline("data", "ObesityDataSet.csv")

    def test_load_and_preprocess_data(self, pipeline):
        pipeline.load_and_preprocess_data()
        assert pipeline.x_train is not None
        assert pipeline.x_test is not None
        assert pipeline.y_train is not None
        assert pipeline.y_test is not None