from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from typing import Any, Dict
from skyrl_gym.envs.lcb.livecodebench import compute_score
import json


class LCBEnv(BaseTextEnv):
    """
    Environment for LiveCodeBench execution environment.
    """

    def __init__(
        self,
        env_config: Any = None,
        extras: Dict[str, Any] = {},
    ):
        super().__init__()

        assert "reward_spec" in extras, "reward_spec field is required"
        assert "ground_truth" in extras["reward_spec"], "ground_truth is required in reward_spec field"
        self.tests = json.loads(extras["reward_spec"]["ground_truth"])
        self._sandbox_profiling: Dict[str, Any] = {}

    def _get_reward(self, action: str) -> float:
        _, reward, _ = compute_score(action, self.tests)
        return reward

    def step(self, action: str) -> BaseTextEnvStepOutput:
        done = True
        parsed_code, reward, self._sandbox_profiling = compute_score(action, self.tests)

        return BaseTextEnvStepOutput(observations=[], reward=reward, done=done, metadata={"parsed_code": parsed_code})

    def get_metrics(self) -> Dict[str, Any]:
        return {"sandbox_profiling": self._sandbox_profiling}
