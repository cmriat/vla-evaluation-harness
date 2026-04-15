"""RoboTwin 2.0 benchmark — dual-arm manipulation on SAPIEN/CuRobo.

Ported from the existing ``vla_evaluation_harness`` implementation
shipped in the ``robotwin`` Docker image.

Non-obvious behaviors:
    - **Expert check**: ``get_tasks()`` optionally runs the oracle planner
      per seed to verify solvability (``skip_expert_check=False``).
    - **Lazy init**: Heavy imports happen on first use, not at construction.
    - **14D action**: dual-arm qpos; 16D inputs are trimmed to 14D.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
from typing import Any

import numpy as np

from vla_eval.benchmarks.base import StepBenchmark, StepResult
from vla_eval.types import Action, EpisodeResult, Observation, Task

logger = logging.getLogger(__name__)

ROBOTWIN_ROOT = "/app/RoboTwin"

# ---------------------------------------------------------------------------
# Multiprocessing helpers for parallel expert check
# ---------------------------------------------------------------------------
_worker_state: dict[str, Any] = {}


def _worker_init(
    task_name: str,
    args: dict[str, Any],
    instruction_type: str,
    test_num: int,
) -> None:
    """Initialise a worker process: add RoboTwin to path, create env."""
    for p in [ROBOTWIN_ROOT, f"{ROBOTWIN_ROOT}/policy", f"{ROBOTWIN_ROOT}/description/utils"]:
        if p not in sys.path:
            sys.path.insert(0, p)
    os.chdir(ROBOTWIN_ROOT)

    envs_module = importlib.import_module(f"envs.{task_name}")
    env_class = getattr(envs_module, task_name)

    _worker_state["env"] = env_class()
    _worker_state["args"] = args
    _worker_state["task_name"] = task_name
    _worker_state["instruction_type"] = instruction_type
    _worker_state["test_num"] = test_num


def _check_seed(seed: int) -> dict[str, Any] | None:
    """Check one seed via oracle planner.  Returns a task dict or ``None``."""
    env = _worker_state["env"]
    args = _worker_state["args"]
    task_name = _worker_state["task_name"]

    try:
        # now_ep_num=0: in parallel mode we don't know the final episode index yet;
        # it is reassigned after collecting and sorting all valid seeds.
        env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **args)
        episode_info = env.play_once()
        env.close_env()
        if env.plan_success and env.check_success():
            from generate_episode_instructions import generate_episode_descriptions

            results = generate_episode_descriptions(
                task_name,
                [episode_info["info"]],
                _worker_state["test_num"],
            )
            rng = np.random.RandomState(seed)
            instruction = rng.choice(results[0][_worker_state["instruction_type"]])
            return {"name": task_name, "suite": "robotwin", "seed": seed, "instruction": str(instruction)}
        return None
    except Exception as exc:
        logger.warning("Expert check worker failed for seed %d: %s", seed, exc)
        try:
            env.close_env()
        except Exception:
            pass
        return None


class RoboTwinBenchmark(StepBenchmark):
    """RoboTwin dual-arm manipulation benchmark (SAPIEN/CuRobo).

    Args:
        task_name: RoboTwin task (e.g. ``"grab_roller"``).
        task_config: Config name under ``task_config/`` (default ``"demo_clean"``).
        seed: Base seed index.  Starting seed = ``100000 * (1 + seed)``.
        instruction_type: Instruction variant (``"seen"`` or ``"unseen"``).
        test_num: Number of valid episodes to evaluate.
        skip_expert_check: If ``True``, skip oracle planner verification in
            ``get_tasks()`` (useful for quick smoke tests).
    """

    def __init__(
        self,
        task_name: str,
        task_config: str = "demo_clean",
        seed: int = 0,
        instruction_type: str = "seen",
        test_num: int = 100,
        skip_expert_check: bool = False,
    ) -> None:
        import re

        super().__init__()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", task_name):
            raise ValueError(f"Invalid task_name: {task_name!r}")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", task_config):
            raise ValueError(f"Invalid task_config: {task_config!r}")
        self.task_name = task_name
        self.task_config = task_config
        self.seed = seed
        self.instruction_type = instruction_type
        self.test_num = test_num
        self.skip_expert_check = skip_expert_check
        self._env: Any = None
        self._env_class: Any = None
        self._args: dict[str, Any] | None = None

    # -----------------------------------------------------------------
    # Lazy init
    # -----------------------------------------------------------------

    def _init_robotwin(self) -> None:
        """Add RoboTwin paths, load YAML configs, resolve embodiment."""
        if self._args is not None:
            return

        for p in [ROBOTWIN_ROOT, f"{ROBOTWIN_ROOT}/policy", f"{ROBOTWIN_ROOT}/description/utils"]:
            if p not in sys.path:
                sys.path.insert(0, p)

        os.chdir(ROBOTWIN_ROOT)
        import yaml

        config_path = os.path.join(
            ROBOTWIN_ROOT,
            "task_config",
            f"{self.task_config}.yml",
        )
        with open(config_path) as f:
            args: dict[str, Any] = yaml.safe_load(f)

        args["task_name"] = self.task_name
        args["task_config"] = self.task_config

        from envs import CONFIGS_PATH

        embodiment_type = args.get("embodiment")
        with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml")) as f:
            _embodiment_types = yaml.safe_load(f)

        def _get_file(etype: str) -> str:
            return _embodiment_types[etype]["file_path"]

        if len(embodiment_type) == 1:
            args["left_robot_file"] = _get_file(embodiment_type[0])
            args["right_robot_file"] = _get_file(embodiment_type[0])
            args["dual_arm_embodied"] = True
        elif len(embodiment_type) == 3:
            args["left_robot_file"] = _get_file(embodiment_type[0])
            args["right_robot_file"] = _get_file(embodiment_type[1])
            args["embodiment_dis"] = embodiment_type[2]
            args["dual_arm_embodied"] = False

        def _get_config(robot_file: str) -> dict:
            with open(os.path.join(robot_file, "config.yml")) as f:
                return yaml.safe_load(f)

        args["left_embodiment_config"] = _get_config(args["left_robot_file"])
        args["right_embodiment_config"] = _get_config(args["right_robot_file"])

        with open(os.path.join(CONFIGS_PATH, "_camera_config.yml")) as f:
            _camera_config = yaml.safe_load(f)

        hcam = args["camera"]["head_camera_type"]
        args["head_camera_h"] = _camera_config[hcam]["h"]
        args["head_camera_w"] = _camera_config[hcam]["w"]
        args["eval_mode"] = True

        self._args = args
        envs_module = importlib.import_module(f"envs.{self.task_name}")
        self._env_class = getattr(envs_module, self.task_name)
        logger.info("RoboTwin initialised: task=%s", self.task_name)

    def _create_env(self) -> Any:
        assert self._env_class is not None
        return self._env_class()

    def cleanup(self) -> None:
        if self._env is not None:
            try:
                self._env.close_env(clear_cache=True)
            except Exception:
                pass
            self._env = None

    # -----------------------------------------------------------------
    # StepBenchmark interface
    # -----------------------------------------------------------------

    def get_tasks(self, num_workers: int = 1) -> list[Task]:
        self._init_robotwin()
        assert self._args is not None
        st_seed = 100000 * (1 + self.seed)

        if self.skip_expert_check:
            return [
                {
                    "name": self.task_name,
                    "suite": "robotwin",
                    "seed": st_seed + i,
                    "episode_idx": i,
                    "instruction": f"Perform the {self.task_name} task.",
                }
                for i in range(self.test_num)
            ]

        if num_workers > 1:
            return self._get_tasks_parallel(st_seed, num_workers)

        # Full expert check — run oracle planner per seed (single process)
        from generate_episode_instructions import generate_episode_descriptions

        env = self._create_env()
        tasks: list[Task] = []
        now_seed = st_seed
        max_seed = st_seed + self.test_num * 20
        episode_idx = 0
        logger.info("Running expert checks from seed %d ...", st_seed)

        while len(tasks) < self.test_num:
            if now_seed >= max_seed:
                raise RuntimeError(
                    f"Exhausted {self.test_num * 20} seeds without finding {self.test_num} valid episodes "
                    f"(found {len(tasks)}). Task '{self.task_name}' may be broken."
                )
            try:
                env.setup_demo(
                    now_ep_num=episode_idx,
                    seed=now_seed,
                    is_test=True,
                    **self._args,
                )
                episode_info = env.play_once()
                env.close_env()
                if env.plan_success and env.check_success():
                    results = generate_episode_descriptions(
                        self.task_name,
                        [episode_info["info"]],
                        self.test_num,
                    )
                    rng = np.random.RandomState(now_seed)
                    instruction = rng.choice(
                        results[0][self.instruction_type],
                    )
                    tasks.append(
                        {
                            "name": self.task_name,
                            "suite": "robotwin",
                            "seed": now_seed,
                            "episode_idx": episode_idx,
                            "instruction": instruction,
                        }
                    )
                    episode_idx += 1
            except Exception as e:
                logger.warning("Expert check failed for seed %d: %s", now_seed, e)
                try:
                    env.close_env()
                except Exception:
                    pass
            now_seed += 1
        return tasks

    def _get_tasks_parallel(self, st_seed: int, num_workers: int) -> list[Task]:
        """Run expert checks in parallel using a process pool.

        Uses ``spawn`` start method to avoid inheriting parent GPU state.
        Seeds are processed in finite batches to avoid unbounded memory from
        feeding an infinite iterator into ``imap_unordered``.
        """
        from multiprocessing import get_context

        logger.info(
            "Running parallel expert checks from seed %d with %d workers ...",
            st_seed,
            num_workers,
        )

        ctx = get_context("spawn")
        tasks: list[Task] = []
        next_seed = st_seed
        max_seed = st_seed + self.test_num * 20
        batch_size = num_workers * 10

        with ctx.Pool(
            num_workers,
            initializer=_worker_init,
            initargs=(self.task_name, self._args, self.instruction_type, self.test_num),
        ) as pool:
            while len(tasks) < self.test_num:
                if next_seed >= max_seed:
                    raise RuntimeError(
                        f"Exhausted {self.test_num * 20} seeds without finding {self.test_num} valid episodes "
                        f"(found {len(tasks)}). Task '{self.task_name}' may be broken."
                    )
                seeds = range(next_seed, min(next_seed + batch_size, max_seed))
                next_seed += batch_size
                for result in pool.imap_unordered(_check_seed, seeds, chunksize=1):
                    if result is not None:
                        tasks.append(result)
                        logger.info("Expert check: %d/%d valid seeds found", len(tasks), self.test_num)
                        if len(tasks) >= self.test_num:
                            break

        # Sort by seed for deterministic ordering, trim to exact count, assign episode_idx
        tasks.sort(key=lambda t: t["seed"])
        tasks = tasks[: self.test_num]
        for i, t in enumerate(tasks):
            t["episode_idx"] = i

        return tasks

    def reset(self, task: Task) -> Any:
        self._init_robotwin()
        assert self._args is not None

        if self._env is not None:
            try:
                self._env.close_env(clear_cache=True)
            except Exception as e:
                logger.warning("Failed to close previous RoboTwin env: %s", e)
            self._env = None

        self._env = self._create_env()
        self._env.setup_demo(
            now_ep_num=task.get("episode_idx", 0),
            seed=task["seed"],
            is_test=True,
            **self._args,
        )
        self._env.set_instruction(instruction=task["instruction"])
        raw_obs = self._env.get_obs()
        return raw_obs

    def step(self, action: Action) -> StepResult:
        raw = action.get("actions", action.get("action"))
        act = np.asarray(raw, dtype=np.float64).flatten()
        assert act.shape[-1] == 14, f"Action dimension mismatch: got {act.shape[-1]}, expected 14"
        if len(act) > 14:
            act = act[:14]
        elif len(act) < 14:
            act = np.pad(act, (0, 14 - len(act)))

        self._env.take_action(act, action_type="qpos")
        raw_obs = self._env.get_obs()
        success = bool(self._env.eval_success)
        done = success or (self._env.take_action_cnt >= self._env.step_lim)
        return StepResult(obs=raw_obs, reward=1.0 if success else 0.0, done=done, info={"success": success})

    def make_obs(self, raw_obs: Any, task: Task) -> Observation:
        return {
            "images": {
                "head_camera": raw_obs["observation"]["head_camera"]["rgb"],
                "left_camera": raw_obs["observation"]["left_camera"]["rgb"],
                "right_camera": raw_obs["observation"]["right_camera"]["rgb"],
            },
            "task_description": raw_obs.get(
                "language",
                task.get("instruction", ""),
            ),
            "joint_state": np.array(raw_obs["joint_action"]["vector"]),
        }

    def check_done(self, step_result: StepResult) -> bool:
        return step_result.done

    def get_step_result(self, step_result: StepResult) -> EpisodeResult:
        return {"success": step_result.info.get("success", False)}

    def get_metadata(self) -> dict[str, Any]:
        return {"max_steps": 400, "task_name": self.task_name}
