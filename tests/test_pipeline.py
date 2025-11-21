from tvm_cost_model.training.pipeline import TrainingPipeline


def test_pipeline_predict_returns_prediction():
    pipeline = TrainingPipeline()
    pipeline.fit(["tir"], [0.0])
    prediction = pipeline.predict("tir")
    assert prediction.score == 0.0
