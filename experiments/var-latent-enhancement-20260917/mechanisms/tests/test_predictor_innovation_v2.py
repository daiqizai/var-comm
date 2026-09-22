import ast
import json
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]
SOURCE = (HERE / "src/latent_mechanisms/predictor_innovation.py").read_text()


def test_v2_config_declares_matched_conditions():
    config = json.loads((HERE / "predictor_innovation_v2_config.json").read_text())
    assert "P(Fb_TX)" in config["tx_conditions"]
    assert "P(Fb_RX)" in config["rx_start"]
    assert config["new_holdout"] is False
    assert "strict_false_weight_loading" in config["forbidden"]


def test_v2_has_strict_loading_and_independent_optimizer_restore():
    tree = ast.parse(SOURCE)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert any(isinstance(node.func, ast.Attribute) and node.func.attr == "load_state_dict"
               and any(keyword.arg == "strict" and isinstance(keyword.value, ast.Constant)
                       and keyword.value.value is True for keyword in node.keywords) for node in calls)
    assert SOURCE.count("copy.deepcopy(parent_opt)") == 2
    assert "matched_tx_shared_P_rx_shared_P_two_arm_training" in SOURCE
    assert "receive_training_residual_sample" in SOURCE
    assert "residual=batch['F']-(ptx if innovation else batch['Fb_TX'])" in SOURCE
    assert "selected_checkpoint_evaluation" in SOURCE
    assert "phase='selected_evaluation'" in SOURCE
    assert "torch.load(r['checkpoint'" in SOURCE


def test_v2_does_not_use_rx_status_as_predictor_condition():
    # P's only status literal is the registered canonical condition.  Actual
    # rx_status is passed only into the communication receiver and loss gate.
    assert "shared_predictor(predictor,batch['Fb_RX']" in SOURCE
    assert "shared_predictor(predictor,batch['Fb_TX']" in SOURCE
    assert "shared_predictor(predictor,batch['rx_status']" not in SOURCE
