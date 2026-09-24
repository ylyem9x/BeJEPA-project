from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch

from humanoidverse.simulator.base_simulator.base_simulator import BaseSimulator
from humanoidverse.utils.torch_utils import quat_rotate, quat_rotate_inverse


class MuJoCo(BaseSimulator):
    def __init__(self, config, device):
        super().__init__(config, device)
        self.simulator_config = config.simulator.config
        self.robot_cfg = config.robot
        self.domain_rand_config = config.domain_rand
        self.device = device
        self.renderer = None
        if config.save_rendering_dir is not None:
            self.save_rendering_dir = Path(config.save_rendering_dir)
        self.render_width = 400
        self.render_height = 400

    def setup(self):
        hv_root = Path(__file__).parents[2]
        self.model_path = str(hv_root / "data/robots/g1/scene_29dof_freebase_mujoco.xml")
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        self.sim_dt = 1 / self.simulator_config.sim.fps
        self.default_dof_frictionloss = self.model.dof_frictionloss.copy()
        self.default_body_mass = self.model.body_mass.copy()
        self.default_geom_friction = self.model.geom_friction.copy()
        self.model.opt.timestep = 1 / self.simulator_config.sim.fps
        self.viewer = None
        self.episodic_domain_randomization(None)

    def episodic_domain_randomization(self, env_ids):
        self.model.dof_frictionloss[:] = self.default_dof_frictionloss
        self.model.body_mass[:] = self.default_body_mass
        self.model.geom_friction[:] = self.default_geom_friction
        if self.domain_rand_config.get("randomize_link_mass", False):
            dmass = np.random.uniform(
                low=self.domain_rand_config["link_mass_range"][0],
                high=self.domain_rand_config["link_mass_range"][1],
                size=(self.model.nbody,),
            )
            self.model.body_mass[:] = self.model.body_mass * dmass
        if self.domain_rand_config.get("randomize_friction", False):
            njnt = self.model.njnt
            frictionloss = self.model.dof_frictionloss[6:] * np.random.uniform(
                low=abs(self.domain_rand_config["friction_range"][0]),
                high=abs(self.domain_rand_config["friction_range"][1]),
                size=(njnt - 1,),
            )
            self.model.dof_frictionloss[6:] = frictionloss
            friction = self.model.geom_friction * np.random.uniform(
                low=abs(self.domain_rand_config["friction_range"][0]),
                high=abs(self.domain_rand_config["friction_range"][1]),
                size=(self.model.ngeom, 1),
            )
            self.model.geom_friction[:] = friction
        if self.domain_rand_config.get("randomize_base_com", False):
            self.torso_id = self.model.body("torso_link").id
            assert self.torso_id > -1
            x_uniform_range = self.domain_rand_config["base_com_range"]["x"]
            assert self.domain_rand_config["base_com_range"]["y"] == x_uniform_range
            assert self.domain_rand_config["base_com_range"]["z"] == x_uniform_range
            dmass_torso = np.random.uniform(low=x_uniform_range[0], high=x_uniform_range[1])
            self.model.body_mass[self.torso_id] = self.model.body_mass[self.torso_id] + dmass_torso

    def setup_terrain(self, mesh_type):
        if mesh_type not in (None, "plane"):
            raise ValueError("MuJoCo playback supports plane terrain only")

    def load_assets(self):
        self.num_dof = self.model.nv - 6
        self.dof_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(self.model.njnt)][
            1:
        ]
        self.body_names = list(self.robot_cfg.body_names)
        self.body_id = np.array([self.model.body(name).id for name in self.body_names])
        self.num_bodies = len(self.body_names)
        assert self.num_dof == len(self.robot_cfg.dof_names), "Number of DOFs must match the config."
        assert self.dof_names == self.robot_cfg.dof_names, "DOF names must match the config."

    def create_envs(self, num_envs, env_origins, base_init_state):
        if num_envs != 1:
            raise ValueError("MuJoCo playback requires num_envs=1")
        self.num_envs = 1
        self.env_config = self.config
        self.env_origins = env_origins
        self.envs = [self.data]
        self.robot_handles = []
        self.base_init_state = base_init_state
        self.data.qpos[0:3] = base_init_state[0:3].cpu()
        self.data.qpos[3:7] = base_init_state[3:7][..., [3, 0, 1, 2]].cpu()
        self.data.qvel[0:6] = base_init_state[7:13].cpu()
        self._body_list = self.body_names
        return (self.envs, self.robot_handles)

    def prepare_sim(self):
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_step(self.model, self.data)
        self.refresh_sim_tensors()

    def refresh_sim_tensors(self):
        mujoco.mj_forward(self.model, self.data)

    def apply_torques_at_dof(self, torques):
        if isinstance(torques, torch.Tensor):
            torques = torques.cpu().numpy()
        self.data.ctrl[6:] = torques

    def set_actor_root_state_tensor(self, set_env_ids, root_states):
        if isinstance(root_states, torch.Tensor):
            root_states = root_states.clone()
            root_states[:, 10:13] = quat_rotate_inverse(root_states[:, 3:7], root_states[:, 10:13], w_last=True)
            root_states = root_states.cpu().numpy()
        self.data.qpos[0:3] = root_states[0, 0:3]
        self.data.qpos[3:7] = root_states[0, 3:7][..., [3, 0, 1, 2]]
        self.data.qvel[0:3] = root_states[0, 7:10]
        self.data.qvel[3:6] = root_states[0, 10:13]

    def set_dof_state_tensor(self, set_env_ids, dof_states):
        if isinstance(dof_states, torch.Tensor):
            dof_states = dof_states.cpu().numpy()
        self.data.qpos[7:] = dof_states[0, :, 0]
        self.data.qvel[6:] = dof_states[0, :, 1]

    def apply_rigid_body_force_at_pos_tensor(self, force_tensor, force, pos):
        self.data.xfrc_applied[:, 0:3] = force_tensor

    def simulate_at_each_physics_step(self):
        mujoco.mj_step(self.model, self.data)
        self.refresh_sim_tensors()
        if self.viewer is not None:
            self.viewer.sync()

    def get_dof_limits_properties(self):
        self.hard_dof_pos_limits = torch.zeros(
            self.num_dof, 2, dtype=torch.float, device=self.sim_device, requires_grad=False
        )
        self.dof_pos_limits = torch.zeros(
            self.num_dof, 2, dtype=torch.float, device=self.sim_device, requires_grad=False
        )
        self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.sim_device, requires_grad=False)
        self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.sim_device, requires_grad=False)
        for i in range(self.num_dof):
            self.hard_dof_pos_limits[i, 0] = self.robot_cfg.dof_pos_lower_limit_list[i]
            self.hard_dof_pos_limits[i, 1] = self.robot_cfg.dof_pos_upper_limit_list[i]
            self.dof_pos_limits[i, 0] = self.robot_cfg.dof_pos_lower_limit_list[i]
            self.dof_pos_limits[i, 1] = self.robot_cfg.dof_pos_upper_limit_list[i]
            self.dof_vel_limits[i] = self.robot_cfg.dof_vel_limit_list[i]
            self.torque_limits[i] = self.robot_cfg.dof_effort_limit_list[i]
            m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
            r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
            self.dof_pos_limits[i, 0] = m - 0.5 * r * self.config.rewards.reward_limit.soft_dof_pos_limit
            self.dof_pos_limits[i, 1] = m + 0.5 * r * self.config.rewards.reward_limit.soft_dof_pos_limit
        return (self.dof_pos_limits, self.dof_vel_limits, self.torque_limits)

    def find_rigid_body_indice(self, body_name):
        if body_name not in self.body_names:
            raise ValueError(f"Rigid body '{body_name}' not found in the model.")
        return self.body_names.index(body_name)

    def setup_viewer(self):
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def render(self, sync_frame_time=True):
        if self.viewer is not None:
            self.viewer.sync()
            return None
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, width=self.render_width, height=self.render_height)
        self.renderer.update_scene(self.data, camera="track")
        return self.renderer.render()

    def _tensor(self, value):
        return torch.tensor(np.asarray(value)[None], device=self.device, dtype=torch.float32)

    @property
    def dof_state(self):
        return torch.cat([self.dof_pos[..., None], self.dof_vel[..., None]], dim=-1)

    @property
    def robot_root_states(self):
        base_quat = self.base_quat
        qvel_tensor = self._tensor(self.data.qvel[:6])
        return torch.cat(
            [
                self._tensor(self.data.qpos[:3]),
                base_quat,
                qvel_tensor[:, 0:3],
                quat_rotate(base_quat, qvel_tensor[:, 3:6], w_last=True),
            ],
            dim=-1,
        )

    @property
    def contact_forces(self):
        return self._tensor(self.data.cfrc_ext[self.body_id, 3:6])

    @property
    def base_quat(self):
        return self._tensor(self.data.qpos[3:7])[..., [1, 2, 3, 0]]

    @property
    def all_root_states(self):
        return self.robot_root_states

    @property
    def dof_pos(self):
        return self._tensor(self.data.qpos[7:])

    @property
    def dof_vel(self):
        return self._tensor(self.data.qvel[6:])

    @property
    def _rigid_body_pos(self):
        return self._tensor(self.data.xpos[self.body_id])

    @property
    def _rigid_body_rot(self):
        return self._tensor(self.data.xquat[self.body_id])[..., [1, 2, 3, 0]]

    @property
    def _rigid_body_vel(self):
        return self._tensor(self.data.cvel[self.body_id, 3:6])

    @property
    def _rigid_body_ang_vel(self):
        return self._tensor(self.data.cvel[self.body_id, 0:3])
