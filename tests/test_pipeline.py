# from tvm_cost_model.data.dataset_builder import MeasurementRecord
# from tvm_cost_model.training.pipeline import TrainingConfig, TrainingPipeline


# def test_pipeline_predict_returns_prediction():
#     pipeline = TrainingPipeline()
#     pipeline.fit(["tir"], [0.0])
#     prediction = pipeline.predict("tir")
#     assert isinstance(prediction.score, float)


# def test_pipeline_trains_on_measurements():
#     measurements = [
#         MeasurementRecord(
#             operator="test",
#             schedule_json="{}",
#             original_tir="for i in range(4): c[i] = a[i]\n",
#             scheduled_tir="for i in range(4): c[i] = a[i]\n",
#             workload_shape={"A": (4,)},
#             runtime_ms=1.0,
#             hardware_id="cpu",
#         ),
#         MeasurementRecord(
#             operator="test",
#             schedule_json="{}",
#             original_tir="for i in range(4): c[i] = a[i]\n",
#             scheduled_tir="for i in range(4): c[i] = a[i]\n",
#             workload_shape={"A": (4,)},
#             runtime_ms=3.0,
#             hardware_id="cpu",
#         ),
#     ]
#     pipeline = TrainingPipeline(TrainingConfig(epochs=1, max_pairs=2, batch_size=1))
#     pair_count = pipeline.fit_measurements(measurements)
#     assert pair_count > 0
#     prediction = pipeline.predict(measurements[0].scheduled_tir)
#     assert isinstance(prediction.score, float)
