import builtins
import copy
import os
from typing import Optional

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import ViewerCfg
from isaaclab.envs.ui import ViewportCameraController
from isaaclab.managers import EventManager, SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import (
    ContactSensor,
    ContactSensorCfg,
    RayCaster,
    RayCasterCfg,
    TiledCamera,
    TiledCameraCfg,
    patterns,
)
from isaaclab.sim import PhysxCfg, SimulationCfg, SimulationContext
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils.timer import Timer
from loguru import logger

from humanoidverse.simulator.base_simulator.base_simulator import BaseSimulator
from humanoidverse.simulator.isaacsim.event_cfg import EventCfg
from humanoidverse.simulator.isaacsim.events import randomize_body_com


class IsaacSim(BaseSimulator):
    def __init__(self, config, device):
        super().__init__(config, device)
        self.simulator_config = config.simulator.config
        self.robot_config = config.robot
        self.env_config = config
        self.terrain_config = config.terrain
        self.domain_rand_config = config.domain_rand
        sim_config: SimulationCfg = SimulationCfg(
            dt=1.0 / self.simulator_config.sim.fps,
            render_interval=self.simulator_config.sim.render_interval,
            device=self.sim_device,
            physx=PhysxCfg(
                bounce_threshold_velocity=self.simulator_config.sim.physx.bounce_threshold_velocity,
                solver_type=self.simulator_config.sim.physx.solver_type,
                max_position_iteration_count=self.simulator_config.sim.physx.num_position_iterations,
                max_velocity_iteration_count=self.simulator_config.sim.physx.num_velocity_iterations,
            ),
        )
        if SimulationContext.instance() is None:
            self.sim: SimulationContext = SimulationContext(sim_config)
        else:
            raise RuntimeError("Simulation context already exists. Cannot create a new one.")
        self.sim.set_camera_view([2.0, 0.0, 2.5], [-0.5, 0.0, 0.5])
        logger.info("IsaacSim initialized.")
        logger.info("[INFO]: Base environment:")
        logger.info(f"\tEnvironment device    : {self.sim_device}")
        logger.info(f"\tPhysics step-size     : {1.0 / self.simulator_config.sim.fps}")
        logger.info(
            f"\tRendering step-size   : {1.0 / self.simulator_config.sim.fps * self.simulator_config.sim.substeps}"
        )
        if self.simulator_config.sim.render_interval < self.simulator_config.sim.control_decimation:
            msg = f"The render interval ({self.simulator_config.sim.render_interval}) is smaller than the decimation ({self.simulator_config.sim.control_decimation}). Multiple render calls will happen for each environment step.If this is not intended, set the render interval to be equal to the decimation."
            logger.warning(msg)
        scene_config: InteractiveSceneCfg = InteractiveSceneCfg(
            num_envs=self.simulator_config.scene.num_envs,
            env_spacing=self.simulator_config.scene.env_spacing,
            replicate_physics=self.simulator_config.scene.replicate_physics,
        )
        with Timer("[INFO]: Time taken for scene creation", "scene_creation"):
            self.scene = InteractiveScene(scene_config)
            self._setup_scene()
        print("[INFO]: Scene manager: ", self.scene)
        viewer_config: ViewerCfg = ViewerCfg()
        if self.sim.render_mode >= self.sim.RenderMode.PARTIAL_RENDERING:
            self.viewport_camera_controller = ViewportCameraController(self, viewer_config)
        else:
            self.viewport_camera_controller = None
        if builtins.ISAAC_LAUNCHED_FROM_TERMINAL is False:
            logger.info("Starting the simulation. This may take a few seconds. Please wait...")
            with Timer("[INFO]: Time taken for simulation start", "simulation_start"):
                self.sim.reset()
        self.default_coms = self._robot.root_physx_view.get_coms().clone()
        self.base_com_bias = torch.zeros((self.simulator_config.scene.num_envs, 3), dtype=torch.float, device="cpu")
        self.event_types = set()
        self.events_cfg = EventCfg()
        if self.domain_rand_config.get("randomize_link_mass", False):
            self.events_cfg.scale_body_mass = EventTerm(
                func=mdp.randomize_rigid_body_mass,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                    "mass_distribution_params": tuple(self.domain_rand_config["link_mass_range"]),
                    "operation": "scale",
                },
            )
            self.event_types.add("startup")
        if self.domain_rand_config.get("randomize_friction", False):
            self.events_cfg.random_joint_friction = EventTerm(
                func=mdp.randomize_rigid_body_material,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                    "static_friction_range": self.domain_rand_config["friction_range"],
                    "dynamic_friction_range": self.domain_rand_config["friction_range"],
                    "restitution_range": (0.0, 0.0),
                    "num_buckets": 1024,
                },
            )
            self.event_types.add("startup")
        if self.domain_rand_config.get("randomize_base_com", False):
            self.events_cfg.random_base_com = EventTerm(
                func=randomize_body_com,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=["torso_link"]),
                    "distribution_params": (
                        torch.tensor(
                            [
                                float(self.domain_rand_config["base_com_range"]["x"][0]),
                                float(self.domain_rand_config["base_com_range"]["y"][0]),
                                float(self.domain_rand_config["base_com_range"]["z"][0]),
                            ]
                        ),
                        torch.tensor(
                            [
                                float(self.domain_rand_config["base_com_range"]["x"][1]),
                                float(self.domain_rand_config["base_com_range"]["y"][1]),
                                float(self.domain_rand_config["base_com_range"]["z"][1]),
                            ]
                        ),
                    ),
                    "operation": "add",
                    "distribution": "uniform",
                    "num_envs": self.simulator_config.scene.num_envs,
                },
            )
            self.event_types.add("startup")
        if self.domain_rand_config.get("push_robots", False):
            self.events_cfg.push_robots = EventTerm(
                func=mdp.push_by_setting_velocity,
                mode="on_push_by_setting_velocity",
                params={
                    "velocity_range": {
                        "x": (-self.domain_rand_config["max_push_vel_xy"], self.domain_rand_config["max_push_vel_xy"]),
                        "y": (-self.domain_rand_config["max_push_vel_xy"], self.domain_rand_config["max_push_vel_xy"]),
                        "roll": (
                            -self.domain_rand_config["max_push_ang_vel"],
                            self.domain_rand_config["max_push_ang_vel"],
                        ),
                        "pitch": (
                            -self.domain_rand_config["max_push_ang_vel"],
                            self.domain_rand_config["max_push_ang_vel"],
                        ),
                        "yaw": (
                            -self.domain_rand_config["max_push_ang_vel"],
                            self.domain_rand_config["max_push_ang_vel"],
                        ),
                    }
                },
            )
            self.event_types.add("on_push_by_setting_velocity")
        self.event_manager = EventManager(self.events_cfg, self)
        print("[INFO] Event Manager: ", self.event_manager)
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")
        if "cuda" in self.sim_device:
            torch.cuda.set_device(self.sim_device)
        self._sim_step_counter = 0
        logger.info("Completed setting up the environment...")

    def _setup_scene(self):
        asset_root = self.robot_config.asset.asset_root
        asset_path = self.robot_config.asset.usd_file
        asset_abs_path = os.path.abspath(os.path.join(asset_root, asset_path))
        assert os.path.isfile(asset_abs_path)
        spawn = sim_utils.UsdFileCfg(
            usd_path=asset_abs_path,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=not bool(self.env_config.robot.asset.self_collisions),
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
        )
        default_joint_angles = copy.deepcopy(self.robot_config.init_state.default_joint_angles)
        init_state = ArticulationCfg.InitialStateCfg(
            pos=tuple(self.robot_config.init_state.pos),
            joint_pos={joint_name: joint_angle for (joint_name, joint_angle) in default_joint_angles.items()},
            joint_vel={".*": 0.0},
        )
        dof_names_list = copy.deepcopy(self.robot_config.dof_names)
        dof_effort_limit_list = self.robot_config.dof_effort_limit_list
        dof_vel_limit_list = self.robot_config.dof_vel_limit_list
        dof_armature_list = self.robot_config.dof_armature_list
        dof_joint_friction_list = self.robot_config.dof_joint_friction_list
        kp_list = []
        kd_list = []
        stiffness_dict = self.robot_config.control.stiffness
        damping_dict = self.robot_config.control.damping
        for i in range(len(dof_names_list)):
            dof_names_i_without_joint = dof_names_list[i].replace("_joint", "")
            for key in stiffness_dict.keys():
                if key in dof_names_i_without_joint:
                    kp_list.append(stiffness_dict[key])
                    kd_list.append(damping_dict[key])
                    print(f"key: {key}, kp: {stiffness_dict[key]}, kd: {damping_dict[key]}")
        stiffness_dict = {".*" + key + ".*": value for (key, value) in stiffness_dict.items()}
        damping_dict = {".*" + key + ".*": value for (key, value) in damping_dict.items()}
        actuators = dict()
        actuators["all"] = ImplicitActuatorCfg(
            joint_names_expr=[dof_names_list[i] for i in range(len(dof_names_list))],
            effort_limit_sim={dof_names_list[i]: dof_effort_limit_list[i] for i in range(len(dof_names_list))},
            velocity_limit_sim={dof_names_list[i]: dof_vel_limit_list[i] for i in range(len(dof_names_list))},
            stiffness=stiffness_dict,
            damping=damping_dict,
            armature={dof_names_list[i]: dof_armature_list[i] for i in range(len(dof_names_list))},
            friction={dof_names_list[i]: dof_joint_friction_list[i] for i in range(len(dof_names_list))},
        )
        robot_articulation_config = ArticulationCfg(
            prim_path="/World/envs/env_.*/Robot",
            spawn=spawn,
            init_state=init_state,
            actuators=actuators,
            soft_joint_pos_limit_factor=0.9,
        )
        contact_sensor_config: ContactSensorCfg = ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
        )
        height_scanner_config = RayCasterCfg(
            prim_path="/World/envs/env_.*/Robot/pelvis",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
            attach_yaw_only=True,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.05, 0.05]),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        if self.terrain_config.mesh_type == "heightfield" or self.terrain_config.mesh_type == "trimesh":
            sub_terrains = {}
            terrain_types = self.terrain_config.terrain_types
            terrain_proportions = self.terrain_config.terrain_proportions
            for terrain_type, proportion in zip(terrain_types, terrain_proportions):
                if proportion > 0:
                    if terrain_type == "flat":
                        sub_terrains[terrain_type] = terrain_gen.MeshPlaneTerrainCfg(proportion=proportion)
                    elif terrain_type == "rough":
                        sub_terrains[terrain_type] = terrain_gen.HfRandomUniformTerrainCfg(
                            proportion=proportion, noise_range=(0.02, 0.1), noise_step=0.02, border_width=0.25
                        )
                    elif terrain_type == "low_obst":
                        sub_terrains[terrain_type] = terrain_gen.MeshRandomGridTerrainCfg(
                            proportion=proportion, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
                        )
            terrain_generator_config = TerrainGeneratorCfg(
                curriculum=self.terrain_config.curriculum,
                size=(self.terrain_config.terrain_length, self.terrain_config.terrain_width),
                border_width=self.terrain_config.border_size,
                num_rows=self.terrain_config.num_rows,
                num_cols=self.terrain_config.num_cols,
                horizontal_scale=self.terrain_config.horizontal_scale,
                vertical_scale=self.terrain_config.vertical_scale,
                slope_threshold=self.terrain_config.slope_treshold,
                use_cache=False,
                sub_terrains=sub_terrains,
            )
            terrain_config = TerrainImporterCfg(
                prim_path="/World/ground",
                terrain_type="generator",
                terrain_generator=terrain_generator_config,
                max_init_terrain_level=9,
                collision_group=-1,
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply",
                    static_friction=self.terrain_config.static_friction,
                    dynamic_friction=self.terrain_config.dynamic_friction,
                ),
                visual_material=sim_utils.MdlFileCfg(
                    mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl", project_uvw=True
                ),
                debug_vis=False,
            )
            terrain_config.num_envs = self.scene.cfg.num_envs
        else:
            terrain_config = TerrainImporterCfg(
                prim_path="/World/ground",
                terrain_type="plane",
                collision_group=-1,
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply",
                    static_friction=self.terrain_config.static_friction,
                    dynamic_friction=self.terrain_config.dynamic_friction,
                    restitution=0.0,
                ),
                debug_vis=False,
            )
            terrain_config.num_envs = self.scene.cfg.num_envs
            terrain_config.env_spacing = self.scene.cfg.env_spacing
        self._robot = Articulation(robot_articulation_config)
        self.scene.articulations["robot"] = self._robot
        self.contact_sensor = ContactSensor(contact_sensor_config)
        self.scene.sensors["contact_sensor"] = self.contact_sensor
        self._height_scanner = RayCaster(height_scanner_config)
        self.scene.sensors["height_scanner"] = self._height_scanner
        self.terrain = terrain_config.class_type(terrain_config)
        self.terrain.env_origins = self.terrain.terrain_origins
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[terrain_config.prim_path])
        light_config1 = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.98, 0.95, 0.88))
        light_config1.func("/World/DomeLight", light_config1, translation=(1, 0, 10))
        self.vis_spheres = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/goal_marker",
                markers={
                    "sphere": sim_utils.SphereCfg(
                        radius=0.05, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0))
                    )
                },
            )
        )
        if self.config.simulator.get("enable_cameras", False):
            self.setup_rendering_cameras()

    def setup_keyboard(self):
        from isaaclab.devices.keyboard.se2_keyboard import Se2Keyboard

        self.keyboard_interface = Se2Keyboard()

    def add_keyboard_callback(self, key, callback):
        self.keyboard_interface.add_callback(key, callback)

    def set_headless(self, headless):
        super().set_headless(headless)
        if not self.headless:
            from isaacsim.util.debug_draw import _debug_draw

            self.draw = _debug_draw.acquire_debug_draw_interface()
        else:
            self.draw = None

    def setup(self):
        self.sim_dt = 1.0 / self.simulator_config.sim.fps
        if not self.headless:
            self.setup_keyboard()

    def setup_terrain(self, mesh_type):
        pass

    def trigger_on_push_by_setting_velocity_events(
        self, env_ids: torch.Tensor | None = None, global_env_step_count: Optional[int] = None
    ):
        self.event_manager.apply(
            env_ids=env_ids, mode="on_push_by_setting_velocity", global_env_step_count=global_env_step_count
        )

    def load_assets(self):
        """
        save self.num_dofs, self.num_bodies, self.dof_names, self.body_names in simulator class
        """
        dof_names_list = copy.deepcopy(self.robot_config.dof_names)
        (self.dof_ids, self.dof_names) = self._robot.find_joints(dof_names_list, preserve_order=True)
        (self.body_ids, self.body_names) = self._robot.find_bodies(self.robot_config.body_names, preserve_order=True)
        self.contact_to_body_idx = [self.contact_sensor.body_names.index(body_name) for body_name in self.body_names]
        self._body_list = self.body_names.copy()
        self.num_dof = len(self.dof_ids)
        self.num_bodies = len(self.body_ids)
        if self.dof_ids != list(range(self.num_dof)):
            logger.warning(
                "The order of the joint_names in the robot_config does not match the order of the joint_ids in IsaacSim."
            )
        assert self.num_dof == len(self.robot_config.dof_names), "Number of DOFs must be equal to number of actions"
        assert self.num_bodies == len(self.robot_config.body_names), (
            "Number of bodies must be equal to number of body names"
        )
        assert self.dof_names == self.robot_config.dof_names, "DOF names must match the config"
        assert self.body_names == self.robot_config.body_names, "Body names must match the config"

    def create_envs(self, num_envs, env_origins, base_init_state):
        self.num_envs = num_envs
        self.env_origins = env_origins
        self.base_init_state = base_init_state
        return (self.scene, self._robot)

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
            self.hard_dof_pos_limits[i, 0] = self.robot_config.dof_pos_lower_limit_list[i]
            self.hard_dof_pos_limits[i, 1] = self.robot_config.dof_pos_upper_limit_list[i]
            self.dof_pos_limits[i, 0] = self.robot_config.dof_pos_lower_limit_list[i]
            self.dof_pos_limits[i, 1] = self.robot_config.dof_pos_upper_limit_list[i]
            self.dof_vel_limits[i] = self.robot_config.dof_vel_limit_list[i]
            self.torque_limits[i] = self.robot_config.dof_effort_limit_list[i]
            m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
            r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
            self.dof_pos_limits[i, 0] = m - 0.5 * r * self.env_config.rewards.reward_limit.soft_dof_pos_limit
            self.dof_pos_limits[i, 1] = m + 0.5 * r * self.env_config.rewards.reward_limit.soft_dof_pos_limit
        return (self.dof_pos_limits, self.dof_vel_limits, self.torque_limits)

    def find_rigid_body_indice(self, body_name):
        (indices, names) = self._robot.find_bodies(body_name)
        indices = [self.body_ids.index(i) for i in indices]
        if len(indices) == 0:
            logger.warning(f"Body {body_name} not found in the contact sensor.")
            return None
        elif len(indices) == 1:
            return indices[0]
        else:
            logger.warning(f"Multiple bodies found for {body_name}.")
            return indices

    def prepare_sim(self):
        self.refresh_sim_tensors()

    @property
    def dof_state(self):
        return torch.cat([self.dof_pos[..., None], self.dof_vel[..., None]], dim=-1)

    def refresh_sim_tensors(self):
        pass

    @property
    def contact_forces(self):
        return self.contact_sensor.data.net_forces_w[:, self.contact_to_body_idx, :]

    @property
    def base_quat(self):
        return self._robot.data.root_state_w[:, [4, 5, 6, 3]]

    @property
    def all_root_states(self):
        return self._robot.data.root_state_w[:, [0, 1, 2, 4, 5, 6, 3, 7, 8, 9, 10, 11, 12]]

    @property
    def robot_root_states(self):
        return self._robot.data.root_state_w[:, [0, 1, 2, 4, 5, 6, 3, 7, 8, 9, 10, 11, 12]]

    @property
    def dof_pos(self):
        return self._robot.data.joint_pos[:, self.dof_ids]

    @property
    def dof_vel(self):
        return self._robot.data.joint_vel[:, self.dof_ids]

    @property
    def _rigid_body_pos(self):
        return self._robot.data.body_pos_w[:, self.body_ids, :]

    @property
    def _rigid_body_rot(self):
        return self._robot.data.body_quat_w[:, self.body_ids][:, :, [1, 2, 3, 0]]

    @property
    def _rigid_body_vel(self):
        return self._robot.data.body_lin_vel_w[:, self.body_ids, :]

    @property
    def _rigid_body_ang_vel(self):
        return self._robot.data.body_ang_vel_w[:, self.body_ids, :]

    @property
    def _rigid_body_dr_masses(self):
        masses = self._robot.root_physx_view.get_masses().to(self.sim_device)
        return masses[:, self.body_ids]

    @property
    def _rigid_body_dr_friction(self):
        frictions = self._robot.root_physx_view.get_material_properties()[:, self.body_ids, :].to(self.sim_device)
        return frictions.reshape(frictions.shape[0], frictions.shape[1] * 3)

    @property
    def _rigid_body_dr_coms(self):
        coms = self._robot.root_physx_view.get_coms()[:, self.body_ids, :].to(self.sim_device)
        return coms.reshape(coms.shape[0], coms.shape[1] * coms.shape[2])

    def apply_torques_at_dof(self, torques):
        self._robot.set_joint_effort_target(torques, joint_ids=self.dof_ids)

    def set_actor_root_state_tensor(self, set_env_ids, root_states):
        self._robot.write_root_pose_to_sim(root_states[set_env_ids, :7], set_env_ids)
        self._robot.write_root_velocity_to_sim(root_states[set_env_ids, 7:], set_env_ids)

    def set_dof_state_tensor(self, set_env_ids, dof_states):
        (dof_pos, dof_vel) = (dof_states[set_env_ids, :, 0], dof_states[set_env_ids, :, 1])
        self._robot.write_joint_state_to_sim(dof_pos, dof_vel, self.dof_ids, set_env_ids)

    def simulate_at_each_physics_step(self):
        self._sim_step_counter += 1
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        if self._sim_step_counter % self.simulator_config.sim.render_interval == 0 and is_rendering:
            self.sim.render()
        self.scene.update(dt=1.0 / self.simulator_config.sim.fps)

    def setup_viewer(self):
        self.viewer = self.viewport_camera_controller

    def render(self, sync_frame_time=True):
        pass

    def clear_lines(self):
        self.draw.clear_lines()
        self.draw.clear_points()

    def draw_sphere(self, pos, radius, color, env_id, pos_id):
        point_list = [(pos[0].item(), pos[1].item(), pos[2].item())]
        color_list = [(color[0], color[1], color[2], 1.0)]
        sizes = [20]
        self.draw.draw_points(point_list, color_list, sizes)

    def draw_spheres_batch(self, pos, rot=None, scales=None):
        self.vis_spheres.visualize(pos, rot, scales)

    def draw_line(self, start_point, end_point, color, env_id):
        start_point_list = [(start_point.x.item(), start_point.y.item(), start_point.z.item())]
        end_point_list = [(end_point.x.item(), end_point.y.item(), end_point.z.item())]
        color_list = [(color.x, color.y, color.z, 1.0)]
        sizes = [1]
        self.draw.draw_lines(start_point_list, end_point_list, color_list, sizes)

    def setup_rendering_cameras(self):
        self.rendering_cam = TiledCamera(
            TiledCameraCfg(
                prim_path="/World/envs/env_.*/Camera_ego",
                offset=TiledCameraCfg.OffsetCfg(pos=(0, 0, 0), rot=(1, 0, 0, 0), convention="world"),
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=5.0, focus_distance=50.0, horizontal_aperture=5, clipping_range=(0.1, 20.0)
                ),
                colorize_semantic_segmentation=False,
                width=512,
                height=512,
            )
        )
        self.scene.sensors["rendering_cam"] = self.rendering_cam
