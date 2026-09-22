import unittest

from watch_stage_a import classify, safe_json


class MonitorTests(unittest.TestCase):
    def running(self):
        return {"pipeline": {"status": "RUNNING_STAGE_A"}, "training": {"status": "STAGE_A_TRAINING"},
                "pipeline_process": {"matches": True}, "worker_process": {"matches": True},
                "seconds_since_activity": 20, "last_training_row": {"image_loss": 0.01},
                "disk_free_GiB": 300, "gpu": {"temperature_C": 78}}

    def test_healthy_active_training(self):
        self.assertEqual(classify(self.running()), ("STAGE_A_RUNNING", []))

    def test_calibration_does_not_count_as_stall(self):
        snapshot = self.running()
        snapshot["seconds_since_activity"] = 90
        self.assertEqual(classify(snapshot)[1], [])

    def test_stall_and_pid_reuse_are_reported(self):
        snapshot = self.running()
        snapshot["seconds_since_activity"] = 700
        self.assertTrue(classify(snapshot)[1])
        snapshot["worker_process"]["matches"] = False
        self.assertIn("stage_A_worker_missing_or_pid_reused", classify(snapshot)[1])

    def test_resource_yield_is_not_a_crash(self):
        snapshot = self.running()
        snapshot["training"]["status"] = "YIELDED_TO_OTHER_AUTHORIZED_GPU_TASK"
        snapshot["worker_process"]["matches"] = False
        self.assertEqual(classify(snapshot), ("WAITING_FOR_AUTHORIZED_RESOURCES", []))

    def test_completion_is_not_full_experiment_success(self):
        snapshot = self.running()
        snapshot["completion"] = {"stage_B_eligible": True}
        snapshot["worker_process"]["matches"] = False
        self.assertEqual(classify(snapshot), ("STAGE_A_COMPLETE_B_NOT_STARTED", []))

    def test_nonfinite_loss_and_immutable_source(self):
        snapshot = self.running()
        snapshot["last_training_row"]["image_loss"] = float("nan")
        snapshot["changed_sources"] = ["configuration"]
        alerts = classify(snapshot)[1]
        self.assertIn("nonfinite_image_loss", alerts)
        self.assertIn("frozen_source_or_config_changed", alerts)
        self.assertEqual(safe_json({"value": float("nan")}), {"value": "nan"})


if __name__ == "__main__":
    unittest.main()
