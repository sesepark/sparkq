"""CPU-only checks for the wrapper around LeRobot's inference server."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


def load_wrapper():
    policy_server = types.ModuleType("lerobot.async_inference.policy_server")

    class PolicyServer:
        _get_action_chunk = lambda self, obs: None
        _predict_action_chunk = lambda self, obs: None
        _reset_server = lambda self: None
        _obs_sanity_checks = lambda self, obs, previous: "original"

    policy_server.PolicyServer = PolicyServer
    policy_server.SUPPORTED_POLICIES = []
    policy_server.get_policy_class = lambda name: None
    policy_server.serve = lambda: None
    lerobot = types.ModuleType("lerobot")
    async_inference = types.ModuleType("lerobot.async_inference")
    async_inference.policy_server = policy_server
    lerobot.async_inference = async_inference
    modules = {
        "lerobot": lerobot,
        "lerobot.async_inference": async_inference,
        "lerobot.async_inference.policy_server": policy_server,
    }
    path = Path(__file__).parent / "bin" / "soarm_policy_server.py"
    spec = importlib.util.spec_from_file_location("soarm_policy_server_under_test", path)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


class FastWAMObservationTest(unittest.TestCase):
    def test_new_fastwam_timestep_reaches_inference_even_when_state_is_similar(self):
        wrapper = load_wrapper()

        class Observation:
            def __init__(self, step):
                self.step = step

            def get_timestep(self):
                return self.step

        server = types.SimpleNamespace(policy_type="fastwam")
        previous = Observation(3)
        self.assertTrue(wrapper._patched_obs_sanity_checks(server, Observation(4), previous))
        self.assertFalse(wrapper._patched_obs_sanity_checks(server, Observation(3), previous))
        self.assertEqual(
            wrapper._patched_obs_sanity_checks(
                types.SimpleNamespace(policy_type="pi05"), Observation(4), previous
            ),
            "original",
        )
        wrapper.main()
        self.assertIs(
            wrapper.policy_server.PolicyServer._obs_sanity_checks,
            wrapper._patched_obs_sanity_checks,
        )


if __name__ == "__main__":
    unittest.main()
