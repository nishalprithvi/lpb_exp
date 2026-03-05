from typing import Optional

import cv2
import gym
import numpy as np
from gym import spaces
from robomimic.envs.env_robosuite import EnvRobosuite

from diffusion_policy.common.robomimic_util import get_robomimic_model_file


class RobomimicImageWrapper(gym.Env):
    def __init__(
        self,
        env: EnvRobosuite,
        shape_meta: dict,
        init_state: Optional[np.ndarray] = None,
        render_obs_key='agentview_image',
    ):
        self.env = env
        self.render_obs_key = render_obs_key
        self.init_state = init_state
        self.seed_state_map = dict()
        self._seed = None
        self.shape_meta = shape_meta
        self.render_cache = None
        self.has_reset_before = False

        print('env._env_name:', self.env._env_name)
        self.model_file = get_robomimic_model_file(self.env._env_name)

        action_shape = shape_meta['action']['shape']
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=action_shape,
            dtype=np.float32,
        )

        observation_space = spaces.Dict()
        for key, value in shape_meta['obs'].items():
            if key == 'language':
                continue

            shape = value['shape']
            min_value, max_value = -1, 1
            if key.endswith('image') or key.endswith('rgb'):
                min_value, max_value = 0, 1
            elif key.endswith('depth'):
                min_value, max_value = 0, 1
            elif key.endswith('quat'):
                min_value, max_value = -1, 1
            elif key.endswith('qpos'):
                min_value, max_value = -1, 1
            elif key.endswith('pos'):
                min_value, max_value = -1, 1
            elif key.endswith('ori'):
                min_value, max_value = -1, 1
            elif key.endswith('states'):
                min_value, max_value = -1, 1
            else:
                raise RuntimeError(f'Unsupported type {key}')

            observation_space[key] = spaces.Box(
                low=min_value,
                high=max_value,
                shape=shape,
                dtype=np.float32,
            )
        self.observation_space = observation_space

    def _process_obs(self, key: str, value):
        arr = np.asarray(value)
        if key.endswith('rgb') or key.endswith('image') or key.endswith('depth'):
            # Convert HWC -> CHW if needed.
            if arr.ndim == 3 and arr.shape[-1] in (1, 3) and arr.shape[0] not in (1, 3):
                arr = np.moveaxis(arr, -1, 0)

            arr = arr.astype(np.float32)
            if arr.max(initial=0.0) > 1.0:
                arr = arr / 255.0

            # Match expected image shape for encoder (e.g. 3x128x128).
            expected = tuple(self.observation_space[key].shape)
            if arr.ndim == 3 and tuple(arr.shape) != expected and len(expected) == 3:
                c_exp, h_exp, w_exp = expected
                if arr.shape[0] in (1, 3):
                    hwc = np.moveaxis(arr, 0, -1)
                else:
                    hwc = arr
                hwc = cv2.resize(hwc, (w_exp, h_exp), interpolation=cv2.INTER_AREA)
                if hwc.ndim == 2:
                    hwc = hwc[..., None]
                if hwc.shape[-1] != c_exp:
                    if c_exp == 1:
                        hwc = hwc[..., :1]
                    elif c_exp == 3 and hwc.shape[-1] == 1:
                        hwc = np.repeat(hwc, 3, axis=-1)
                arr = np.moveaxis(hwc, -1, 0)

        else:
            arr = arr.astype(np.float32)
        return arr

    def get_observation(self, raw_obs=None):
        if raw_obs is None:
            raw_obs = self.env.get_observation()

        render_candidates = [
            self.render_obs_key,
            'agentview_image',
            'agentview_rgb',
            'robot0_eye_in_hand_image',
            'robot0_eye_in_hand_rgb',
        ]
        render_key = next((k for k in render_candidates if k in raw_obs), None)
        if render_key is None:
            raise KeyError(
                f"Missing render key. Tried {render_candidates}. Available raw obs keys: {list(raw_obs.keys())}"
            )
        self.render_cache = raw_obs[render_key]

        obs = dict()
        for key in self.observation_space.keys():
            if key in raw_obs:
                obs[key] = self._process_obs(key, raw_obs[key])
                continue

            if key == 'agentview_rgb':
                candidates = ['agentview_image', 'robot0_eye_in_hand_image', 'agentview_rgb', 'robot0_eye_in_hand_rgb']
            elif key in ('eye_in_hand_rgb', 'robot0_eye_in_hand_rgb'):
                candidates = ['robot0_eye_in_hand_image', 'robot0_eye_in_hand_rgb']
            elif key == 'joint_states':
                candidates = ['robot0_joint_pos']
            elif key == 'ee_pos':
                candidates = ['robot0_eef_pos']
            elif key == 'ee_ori':
                candidates = ['robot0_eef_quat']
            else:
                candidates = []

            alt = next((k for k in candidates if k in raw_obs), None)
            if alt is not None:
                obs[key] = self._process_obs(key, raw_obs[alt])
                continue

            raise KeyError(
                f"Missing observation key '{key}'. Tried aliases {candidates}. Available raw obs keys: {list(raw_obs.keys())}"
            )
        return obs

    def seed(self, seed=None):
        np.random.seed(seed=seed)
        self._seed = seed

    def get_flattened_state(self):
        return self.env.env.sim.get_state().flatten()

    def get_success_label(self):
        return self.env.env._check_success()

    def get_check_tool_on_frame(self):
        return self.env.env._check_tool_on_frame()

    def get_check_frame_assembled(self):
        return self.env.env._check_frame_assembled()

    def get_trash_in_trash_bin(self):
        return self.env.env.transport.trash_in_trash_bin

    def get_payload_in_target_bin(self):
        return self.env.env.transport.payload_in_target_bin

    def set_init_state(self, state):
        self.init_state = state

    def reset(self):
        if self.init_state is not None:
            if not self.has_reset_before:
                self.env.reset()
                self.has_reset_before = True

            if self.model_file is None:
                raw_obs = self.env.reset_to({'states': self.init_state})
            else:
                raw_obs = self.env.reset_to({'states': self.init_state, 'model': self.model_file})
        elif self._seed is not None:
            seed = self._seed
            if seed in self.seed_state_map:
                if self.model_file is None:
                    raw_obs = self.env.reset_to({'states': self.seed_state_map[seed]})
                else:
                    raw_obs = self.env.reset_to({'states': self.seed_state_map[seed], 'model': self.model_file})
            else:
                np.random.seed(seed=seed)
                raw_obs = self.env.reset()
                state = self.env.get_state()['states']
                self.seed_state_map[seed] = state
                if self.model_file is not None:
                    raw_obs = self.env.reset_to({'states': state, 'model': self.model_file})
            self._seed = None
        else:
            raw_obs = self.env.reset()
            state = self.env.get_state()['states']
            if self.model_file is not None:
                raw_obs = self.env.reset_to({'states': state, 'model': self.model_file})

        obs = self.get_observation(raw_obs)
        self.idx = 0
        return obs

    def step(self, action):
        raw_obs, reward, done, info = self.env.step(action)
        self.idx += 1
        obs = self.get_observation(raw_obs)
        return obs, reward, done, info

    def render(self, mode='rgb_array'):
        if self.render_cache is None:
            raise RuntimeError('Must run reset or step before render.')

        arr = np.asarray(self.render_cache)

        # Collapse temporal/batch dimensions if present.
        while arr.ndim > 3:
            arr = arr[-1]

        # Convert to HWC.
        if arr.ndim == 3:
            if arr.shape[0] in (1, 3, 4):
                arr = np.moveaxis(arr, 0, -1)
            elif arr.shape[-1] in (1, 3, 4):
                pass
            else:
                # Fallback: use first channel as grayscale.
                arr = arr[..., :1]
        elif arr.ndim == 2:
            arr = arr[..., None]
        else:
            raise RuntimeError(f'Unexpected render array shape: {arr.shape}')

        arr = arr.astype(np.float32)
        if arr.max(initial=0.0) <= 1.0:
            arr = arr * 255.0
        img = np.clip(arr, 0, 255).astype(np.uint8)
        return img
