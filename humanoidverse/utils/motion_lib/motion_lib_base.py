import gc
import glob
import os.path as osp
import random
from enum import Enum
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.multiprocessing as mp
from easydict import EasyDict
from loguru import logger
from rich.progress import track
from scipy.spatial.transform import Rotation as sRot

from humanoidverse.utils.motion_dataset import load_motion_dataset
from humanoidverse.utils.motion_lib.skeleton import SkeletonTree
from humanoidverse.utils.torch_utils import calc_heading_quat_inv, quat_to_exp_map, slerp


class FixHeightMode(Enum):
    no_fix = 0
    full_fix = 1
    ankle_fix = 2


class MotionlibMode(Enum):
    file = 1
    directory = 2


def to_torch(tensor):
    if torch.is_tensor(tensor):
        return tensor
    else:
        return torch.from_numpy(tensor)


class MotionLibBase:
    def __init__(self, motion_lib_cfg, num_envs, device):
        self.m_cfg = motion_lib_cfg
        self._sim_fps = 1 / self.m_cfg.get("step_dt", 1 / 50)
        self.all_motions_loaded = False
        self.num_envs = num_envs
        self._device = device
        self.mesh_parsers = None
        self.has_action = False
        skeleton_file = Path(self.m_cfg.asset.assetRoot) / self.m_cfg.asset.assetFileName
        self.skeleton_tree = SkeletonTree.from_mjcf(skeleton_file)
        logger.info(f"Loaded skeleton from {skeleton_file}")
        logger.info(f"Loading motion data from {self.m_cfg.motion_file}...")
        self.load_data(self.m_cfg.motion_file)
        self.setup_constants(fix_height=motion_lib_cfg.get("fix_height", FixHeightMode.no_fix), multi_thread=True)
        self.smpl_data = None
        smpl_motion_file = motion_lib_cfg.get("smpl_motion_file", None)
        if smpl_motion_file is not None:
            self.smpl_data = joblib.load(smpl_motion_file)
            self.smpl_data = [self.smpl_data[k] for k in self._motion_data_keys]
        return

    def load_data(self, motion_file, min_length=-1, im_eval=False):
        if osp.isfile(motion_file):
            self.mode = MotionlibMode.file
            self._motion_data_load = load_motion_dataset(Path(motion_file))
        else:
            self.mode = MotionlibMode.directory
            self._motion_data_load = glob.glob(osp.join(motion_file, "*.pkl"))
        data_list = self._motion_data_load
        if self.mode == MotionlibMode.file:
            if min_length != -1:
                data_list = {
                    k: v for (k, v) in list(self._motion_data_load.items()) if len(v["pose_quat_global"]) >= min_length
                }
            elif im_eval:
                data_list = {
                    item[0]: item[1]
                    for item in sorted(
                        self._motion_data_load.items(),
                        key=lambda entry: len(entry[1]["pose_quat_global"]),
                        reverse=True,
                    )
                }
            else:
                data_list = self._motion_data_load
            self._motion_data_list = np.array(list(data_list.values()))
            self._motion_data_keys = np.array(list(data_list.keys()))
        else:
            self._motion_data_list = np.array(self._motion_data_load)
            self._motion_data_keys = np.array(self._motion_data_load)
        self._num_unique_motions = len(self._motion_data_list)
        if self.mode == MotionlibMode.directory:
            self._motion_data_load = joblib.load(self._motion_data_load[0])
        logger.info(f"Loaded {self._num_unique_motions} motions")

    def setup_constants(self, fix_height=FixHeightMode.full_fix, multi_thread=True):
        self.fix_height = fix_height
        self.multi_thread = multi_thread
        self._curr_motion_ids = None
        self._termination_history = torch.zeros(self._num_unique_motions).to(self._device)
        self._success_rate = torch.zeros(self._num_unique_motions).to(self._device)
        self._sampling_history = torch.zeros(self._num_unique_motions).to(self._device)
        self._sampling_prob = torch.ones(self._num_unique_motions).to(self._device) / self._num_unique_motions

    def update_soft_sampling_weight(self, failed_keys):
        if len(failed_keys) > 0:
            all_keys = self._motion_data_keys.tolist()
            indexes = [all_keys.index(k) for k in failed_keys]
            self._termination_history[indexes] += 1
            self.update_sampling_prob(self._termination_history)
            print(
                "############################################################ Auto PMCP ############################################################"
            )
            print(f"Training mostly on {len(self._sampling_prob.cpu().nonzero())} seqs ")
            print(self._motion_data_keys[self._sampling_prob.cpu().nonzero()].flatten())
            print(
                "###############################################################################################################################"
            )
        else:
            all_keys = self._motion_data_keys.tolist()
            self._sampling_prob = torch.ones(self._num_unique_motions).to(self._device) / self._num_unique_motions

    def update_sampling_prob(self, termination_history):
        if len(termination_history) == len(self._termination_history) and termination_history.sum() > 0:
            self._sampling_prob[:] = termination_history / termination_history.sum()
            if self._sampling_prob[self._curr_motion_ids].sum() == 0:
                self._sampling_prob[self._curr_motion_ids] += 1e-06
                self._sampling_prob[:] = self._sampling_prob[:] / self._sampling_prob[:].sum()
            self._sampling_batch_prob = (
                self._sampling_prob[self._curr_motion_ids] / self._sampling_prob[self._curr_motion_ids].sum()
            )
            self._termination_history = termination_history
            return True
        else:
            return False

    def update_sampling_weight_by_id(
        self, priorities: list[float], motions_id: list[int], file_name: dict[int, str] | None = None
    ) -> None:
        """
        Update sampling probabilities (_sampling_prob) given motion keys and their priorities.
        This replaces the distribution with the normalized values.
        """
        if len(motions_id) != len(priorities):
            raise ValueError("motions_id and priorities must have the same length")
        if not motions_id:
            raise ValueError("At least one motion priority is required")
        normalized_motion_ids = []
        for motion_id in motions_id:
            if isinstance(motion_id, bool) or not isinstance(motion_id, (int, np.integer)):
                raise ValueError(f"Motion IDs must be integers, got {motion_id!r}")
            normalized_motion_ids.append(int(motion_id))
        if len(set(normalized_motion_ids)) != len(normalized_motion_ids):
            raise ValueError("motions_id must not contain duplicates")
        invalid_motion_ids = [
            motion_id for motion_id in normalized_motion_ids if motion_id < 0 or motion_id >= self._num_unique_motions
        ]
        if invalid_motion_ids:
            raise IndexError(f"Motion IDs {invalid_motion_ids[:10]} are outside [0, {self._num_unique_motions}).")
        priorities_tensor = torch.as_tensor(priorities, dtype=torch.float32, device=self._device)
        if priorities_tensor.ndim != 1 or not torch.isfinite(priorities_tensor).all():
            raise ValueError("Priorities must be a finite one-dimensional sequence")
        if torch.any(priorities_tensor < 0):
            raise ValueError("Priorities must be non-negative")
        total = priorities_tensor.sum()
        if total <= 0:
            raise ValueError("Sum of priorities must be greater than zero")
        normalized_priorities = priorities_tensor / total
        all_keys = self._motion_data_keys.tolist()
        new_sampling_prob = torch.zeros(self._num_unique_motions, device=self._device)
        motion_ids_tensor = torch.tensor(normalized_motion_ids, dtype=torch.long, device=self._device)
        new_sampling_prob[motion_ids_tensor] = normalized_priorities
        for m in normalized_motion_ids:
            if file_name is not None:
                if m not in file_name:
                    raise ValueError(f"Missing file name for motion ID {m}")
                if all_keys[m] != file_name[m]:
                    raise ValueError(f"Motion ID {m} does not match file name {file_name[m]}")
        if new_sampling_prob.sum() == 0:
            print("[Warning] No valid motions updated. Falling back to uniform sampling.")
            new_sampling_prob = torch.ones(self._num_unique_motions, device=self._device) / self._num_unique_motions
        else:
            new_sampling_prob /= new_sampling_prob.sum()
        self._sampling_prob = new_sampling_prob
        if self._curr_motion_ids is not None:
            current_probability = self._sampling_prob[self._curr_motion_ids].sum()
            if current_probability <= 0:
                self._sampling_prob[self._curr_motion_ids] += torch.finfo(self._sampling_prob.dtype).eps
                self._sampling_prob /= self._sampling_prob.sum()
                current_probability = self._sampling_prob[self._curr_motion_ids].sum()
            self._sampling_batch_prob = self._sampling_prob[self._curr_motion_ids] / current_probability

    def get_motion_actions(self, motion_ids, motion_times):
        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]
        (frame_idx0, frame_idx1, blend) = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)
        f0l = frame_idx0 + self.length_starts[motion_ids]
        frame_idx1 + self.length_starts[motion_ids]
        action = self._motion_actions[f0l]
        return action

    def get_motion_state(self, motion_ids, motion_times, offset=None):
        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]
        (frame_idx0, frame_idx1, blend) = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)
        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]
        if "dof_pos" in self.__dict__:
            local_rot0 = self.dof_pos[f0l]
            local_rot1 = self.dof_pos[f1l]
        else:
            local_rot0 = self.lrs[f0l]
            local_rot1 = self.lrs[f1l]
        body_vel0 = self.gvs[f0l]
        body_vel1 = self.gvs[f1l]
        body_ang_vel0 = self.gavs[f0l]
        body_ang_vel1 = self.gavs[f1l]
        rg_pos0 = self.gts[f0l, :]
        rg_pos1 = self.gts[f1l, :]
        dof_vel0 = self.dvs[f0l]
        dof_vel1 = self.dvs[f1l]
        vals = [
            local_rot0,
            local_rot1,
            body_vel0,
            body_vel1,
            body_ang_vel0,
            body_ang_vel1,
            rg_pos0,
            rg_pos1,
            dof_vel0,
            dof_vel1,
        ]
        for v in vals:
            assert v.dtype != torch.float64
        blend = blend.unsqueeze(-1)
        blend_exp = blend.unsqueeze(-1)
        if offset is None:
            rg_pos = (1.0 - blend_exp) * rg_pos0 + blend_exp * rg_pos1
        else:
            rg_pos = (1.0 - blend_exp) * rg_pos0 + blend_exp * rg_pos1 + offset[..., None, :]
        body_vel = (1.0 - blend_exp) * body_vel0 + blend_exp * body_vel1
        body_ang_vel = (1.0 - blend_exp) * body_ang_vel0 + blend_exp * body_ang_vel1
        if "dof_pos" in self.__dict__:
            dof_vel = (1.0 - blend) * dof_vel0 + blend * dof_vel1
            dof_pos = (1.0 - blend) * local_rot0 + blend * local_rot1
        else:
            dof_vel = (1.0 - blend_exp) * dof_vel0 + blend_exp * dof_vel1
            local_rot = slerp(local_rot0, local_rot1, torch.unsqueeze(blend, axis=-1))
            dof_pos = self._local_rotation_to_dof_smpl(local_rot)
        rb_rot0 = self.grs[f0l]
        rb_rot1 = self.grs[f1l]
        rb_rot = slerp(rb_rot0, rb_rot1, blend_exp)
        return_dict = {}
        if "gts_t" in self.__dict__:
            rg_pos_t0 = self.gts_t[f0l]
            rg_pos_t1 = self.gts_t[f1l]
            rg_rot_t0 = self.grs_t[f0l]
            rg_rot_t1 = self.grs_t[f1l]
            body_vel_t0 = self.gvs_t[f0l]
            body_vel_t1 = self.gvs_t[f1l]
            body_ang_vel_t0 = self.gavs_t[f0l]
            body_ang_vel_t1 = self.gavs_t[f1l]
            if offset is None:
                rg_pos_t = (1.0 - blend_exp) * rg_pos_t0 + blend_exp * rg_pos_t1
            else:
                rg_pos_t = (1.0 - blend_exp) * rg_pos_t0 + blend_exp * rg_pos_t1 + offset[..., None, :]
            rg_rot_t = slerp(rg_rot_t0, rg_rot_t1, blend_exp)
            body_vel_t = (1.0 - blend_exp) * body_vel_t0 + blend_exp * body_vel_t1
            body_ang_vel_t = (1.0 - blend_exp) * body_ang_vel_t0 + blend_exp * body_ang_vel_t1
        else:
            rg_pos_t = rg_pos
            rg_rot_t = rb_rot
            body_vel_t = body_vel
            body_ang_vel_t = body_ang_vel
        if self.smpl_data is not None:
            smpl_pose0 = self._motion_smpl_poses[f0l]
            smpl_pose1 = self._motion_smpl_poses[f1l]
            smpl_pose = (1.0 - blend) * smpl_pose0 + blend * smpl_pose1
            return_dict.update({"smpl_pose": smpl_pose.clone()})
        return_dict.update(
            {
                "root_pos": rg_pos[..., 0, :].clone(),
                "root_rot": rb_rot[..., 0, :].clone(),
                "dof_pos": dof_pos.clone(),
                "root_vel": body_vel[..., 0, :].clone(),
                "root_ang_vel": body_ang_vel[..., 0, :].clone(),
                "dof_vel": dof_vel.view(dof_vel.shape[0], -1).clone(),
                "motion_aa": self._motion_aa[f0l].clone(),
                "motion_bodies": self._motion_bodies[motion_ids].clone(),
                "rg_pos": rg_pos.clone(),
                "rb_rot": rb_rot.clone(),
                "body_vel": body_vel.clone(),
                "body_ang_vel": body_ang_vel.clone(),
                "rg_pos_t": rg_pos_t.clone(),
                "rg_rot_t": rg_rot_t.clone(),
                "body_vel_t": body_vel_t.clone(),
                "body_ang_vel_t": body_ang_vel_t.clone(),
            }
        )
        return return_dict

    def load_motions_for_training(self, max_num_seqs=None):
        if max_num_seqs is not None:
            assert max_num_seqs <= self.num_envs
        if self.all_motions_loaded:
            print("All motions already loaded!!! No need to resample.")
            return
        if self.m_cfg.get("override_num_motions_to_load", None) is not None:
            max_num_seqs = self.m_cfg.override_num_motions_to_load
        if max_num_seqs is None:
            max_num_seqs = self._num_unique_motions
            self.all_motions_loaded = True
            self.load_motions(random_sample=False, num_motions_to_load=self._num_unique_motions)
        elif max_num_seqs > self._num_unique_motions:
            self.all_motions_loaded = True
            self.load_motions(random_sample=False, num_motions_to_load=self._num_unique_motions)
        else:
            self.all_motions_loaded = False
            self.load_motions(random_sample=True, num_motions_to_load=max_num_seqs)

    def load_motions_for_evaluation(self, start_idx=0):
        if self.all_motions_loaded:
            print("All motions already loaded!!! No need to resample.")
            return
        if self._num_unique_motions > self.num_envs:
            self.all_motions_loaded = False
            self.load_motions(random_sample=False, num_motions_to_load=self.num_envs, start_idx=start_idx)
        else:
            self.all_motions_loaded = True
            self.load_motions(random_sample=False, num_motions_to_load=self._num_unique_motions, start_idx=start_idx)

    def load_all_motions(self):
        self.all_motions_loaded = True
        self.load_motions(random_sample=False, num_motions_to_load=self._num_unique_motions)

    def load_motions(self, random_sample=True, start_idx=0, max_len=-1, target_heading=None, num_motions_to_load=None):
        if "gts" in self.__dict__:
            del self.gts, self.grs, self.lrs, self.grvs, self.gravs, self.gavs, self.gvs, self.dvs, self.dof_pos
            if "gts_t" in self.__dict__:
                del self.gts_t, self.grs_t, self.gvs_t, self.gavs_t
        motions = []
        _motion_lengths = []
        _motion_fps = []
        _motion_dt = []
        _motion_num_frames = []
        _motion_bodies = []
        _motion_aa = []
        _motion_actions = []
        _motion_smpl_poses = []
        total_len = 0.0
        self.num_joints = len(self.skeleton_tree.node_names)
        if num_motions_to_load is None:
            num_motion_to_load = self.num_envs
        else:
            num_motion_to_load = num_motions_to_load
        if random_sample:
            sample_idxes = torch.multinomial(self._sampling_prob, num_samples=num_motion_to_load, replacement=True).to(
                self._device
            )
        else:
            sample_idxes = torch.clamp(
                torch.arange(num_motion_to_load) + start_idx, max=self._num_unique_motions - 1
            ).to(self._device)
        self._curr_motion_ids = sample_idxes
        self.curr_motion_keys = (
            [self._motion_data_keys[sample_idxes.cpu()]]
            if sample_idxes.numel() == 1
            else self._motion_data_keys[sample_idxes.cpu()].tolist()
        )
        self._sampling_batch_prob = (
            self._sampling_prob[self._curr_motion_ids] / self._sampling_prob[self._curr_motion_ids].sum()
        )
        logger.info(f"Loading {num_motion_to_load} motions...")
        logger.info(f"Sampling motion: {sample_idxes[:10]}, ....")
        logger.info(f"Current motion keys: {self.curr_motion_keys[:10]}, ....")
        motion_data_list = self._motion_data_list[sample_idxes.cpu().numpy()]
        if self.smpl_data is not None:
            smpl_data_list = [self.smpl_data[idx] for idx in sample_idxes.cpu().numpy()]
        else:
            smpl_data_list = None
        torch.set_num_threads(1)
        num_jobs = min(mp.cpu_count(), 8)
        if num_jobs <= 16 or not self.multi_thread:
            num_jobs = 1
        res_acc = {}
        jobs = motion_data_list
        chunk = np.ceil(len(jobs) / num_jobs).astype(int)
        ids = np.arange(len(jobs))
        jobs = [
            (ids[i : i + chunk], jobs[i : i + chunk], smpl_data_list, self.fix_height, target_heading, max_len)
            for i in range(0, len(jobs), chunk)
        ]
        job_args = [jobs[i] for i in range(len(jobs))]
        if len(job_args) == 1:
            res_acc.update(self.load_motion_with_skeleton(*job_args[0], None, 0))
        else:
            manager = mp.Manager()
            queue = manager.Queue()
            for i in range(1, len(job_args)):
                worker_args = (*job_args[i], queue, i)
                worker = mp.Process(target=self.load_motion_with_skeleton, args=worker_args)
                worker.start()
            res_acc.update(self.load_motion_with_skeleton(*job_args[0], queue, 0))
            for i in track(range(len(job_args) - 1), "Gathering results..."):
                res_acc.update(queue.get())
        for f in track(range(len(res_acc)), description="Processing motions..."):
            (motion_file_data, curr_motion) = res_acc[f]
            motion_fps = curr_motion.fps
            curr_dt = 1.0 / motion_fps
            num_frames = curr_motion.global_rotation.shape[0]
            curr_len = 1.0 / motion_fps * (num_frames - 1)
            if "beta" in motion_file_data:
                _motion_aa.append(motion_file_data["pose_aa"].reshape(-1, self.num_joints * 3))
                _motion_bodies.append(curr_motion.gender_beta)
            else:
                _motion_aa.append(np.zeros((num_frames, self.num_joints * 3)))
                _motion_bodies.append(torch.zeros(17))
            _motion_fps.append(motion_fps)
            _motion_dt.append(curr_dt)
            _motion_num_frames.append(num_frames)
            motions.append(curr_motion)
            _motion_lengths.append(curr_len)
            if self.has_action:
                _motion_actions.append(curr_motion.action)
            if self.smpl_data is not None:
                _motion_smpl_poses.append(curr_motion["smpl_pose"])
            del curr_motion
        self._motion_lengths = torch.tensor(_motion_lengths, device=self._device, dtype=torch.float32)
        self._motion_fps = torch.tensor(_motion_fps, device=self._device, dtype=torch.float32)
        self._motion_bodies = torch.stack(_motion_bodies).to(self._device).type(torch.float32)
        self._motion_aa = torch.tensor(np.concatenate(_motion_aa), device=self._device, dtype=torch.float32)
        if self.smpl_data is not None:
            self._motion_smpl_poses = torch.cat(_motion_smpl_poses, dim=0).float().to(self._device)
        self._motion_dt = torch.tensor(_motion_dt, device=self._device, dtype=torch.float32)
        self._motion_num_frames = torch.tensor(_motion_num_frames, device=self._device)
        if self.has_action:
            self._motion_actions = torch.cat(_motion_actions, dim=0).float().to(self._device)
        self._num_motions = len(motions)
        self.gts = torch.cat([m.global_translation for m in motions], dim=0).float().to(self._device)
        self.grs = torch.cat([m.global_rotation for m in motions], dim=0).float().to(self._device)
        self.lrs = torch.cat([m.local_rotation for m in motions], dim=0).float().to(self._device)
        self.grvs = torch.cat([m.global_root_velocity for m in motions], dim=0).float().to(self._device)
        self.gravs = torch.cat([m.global_root_angular_velocity for m in motions], dim=0).float().to(self._device)
        self.gavs = torch.cat([m.global_angular_velocity for m in motions], dim=0).float().to(self._device)
        self.gvs = torch.cat([m.global_velocity for m in motions], dim=0).float().to(self._device)
        self.dvs = torch.cat([m.dof_vels for m in motions], dim=0).float().to(self._device)
        if "global_translation_extend" in motions[0].__dict__:
            self.gts_t = torch.cat([m.global_translation_extend for m in motions], dim=0).float().to(self._device)
            self.grs_t = torch.cat([m.global_rotation_extend for m in motions], dim=0).float().to(self._device)
            self.gvs_t = torch.cat([m.global_velocity_extend for m in motions], dim=0).float().to(self._device)
            self.gavs_t = torch.cat([m.global_angular_velocity_extend for m in motions], dim=0).float().to(self._device)
        if "dof_pos" in motions[0].__dict__:
            self.dof_pos = torch.cat([m.dof_pos for m in motions], dim=0).float().to(self._device)
        lengths = self._motion_num_frames
        lengths_shifted = lengths.roll(1)
        lengths_shifted[0] = 0
        self.length_starts = lengths_shifted.cumsum(0)
        self.motion_ids = torch.arange(len(motions), dtype=torch.long, device=self._device)
        motions[0]
        self.num_bodies = self.num_joints
        num_motions = self.num_motions()
        total_len = self.get_total_length()
        logger.info(
            f"Loaded {num_motions:d} motions with a total length of {total_len:.3f}s and {self.gts.shape[0]} frames."
        )
        del (
            motions,
            _motion_lengths,
            _motion_fps,
            _motion_dt,
            _motion_num_frames,
            _motion_bodies,
            _motion_aa,
            _motion_actions,
        )
        torch.cuda.empty_cache()
        gc.collect()

    def fix_trans_height(self, pose_aa, trans, fix_height_mode):
        if fix_height_mode == FixHeightMode.no_fix:
            return (trans, 0)
        with torch.no_grad():
            mesh_obj = self.mesh_parsers.mesh_fk(pose_aa[None, :1], trans[None, :1])
            height_diff = np.asarray(mesh_obj.vertices)[..., 2].min()
            trans[..., 2] -= height_diff
            return (trans, height_diff)

    def load_motion_with_skeleton(
        self, ids, motion_data_list, smpl_data_list, fix_height, target_heading, max_len, queue, pid
    ):
        res = {}
        if pid == 0:
            pbar = track(range(len(ids)), description="Loading motions...")
        else:
            pbar = range(len(ids))
        for f in pbar:
            curr_id = ids[f]
            curr_file = motion_data_list[f]
            if not isinstance(curr_file, dict) and osp.isfile(curr_file):
                key = motion_data_list[f].split("/")[-1].split(".")[0]
                curr_file = joblib.load(curr_file)[key]
            seq_len = curr_file["root_trans_offset"].shape[0]
            if max_len == -1 or seq_len < max_len:
                (start, end) = (0, seq_len)
            else:
                start = random.randint(0, seq_len - max_len)
                end = start + max_len
            trans = to_torch(curr_file["root_trans_offset"]).clone()[start:end]
            pose_aa = to_torch(curr_file["pose_aa"][start:end]).clone()
            if "action" in curr_file.keys():
                self.has_action = True
            dt = 1 / curr_file["fps"]
            (B, J, N) = pose_aa.shape
            if target_heading is not None:
                start_root_rot = sRot.from_rotvec(pose_aa[0, 0])
                heading_inv_rot = sRot.from_quat(
                    calc_heading_quat_inv(torch.from_numpy(start_root_rot.as_quat()[None,]), w_last=True)
                )
                heading_delta = sRot.from_quat(target_heading) * heading_inv_rot
                pose_aa[:, 0] = torch.tensor((heading_delta * sRot.from_rotvec(pose_aa[:, 0])).as_rotvec())
                trans = torch.matmul(trans, torch.from_numpy(heading_delta.as_matrix().squeeze().T))
            if self.mesh_parsers is not None:
                (trans, trans_fix) = self.fix_trans_height(pose_aa, trans, fix_height_mode=fix_height)
                curr_motion = self.mesh_parsers.fk_batch(pose_aa[None,], trans[None,], return_full=True, dt=dt)
                if self.smpl_data is not None:
                    skip = int(smpl_data_list[f]["fps"] // 30)
                    curr_motion["smpl_pose"] = (
                        torch.tensor(smpl_data_list[f]["pose_aa"][::skip][start:end]).float().to(self._device)
                    )
                    assert curr_motion["smpl_pose"].shape[0] == pose_aa.shape[0]
                curr_motion = EasyDict({k: v.squeeze() if torch.is_tensor(v) else v for (k, v) in curr_motion.items()})
                if self.has_action:
                    curr_motion.action = to_torch(curr_file["action"]).clone()[start:end]
                res[curr_id] = (curr_file, curr_motion)
            else:
                logger.error("No mesh parser found")
        if queue is not None:
            queue.put(res)
        else:
            return res

    def num_motions(self):
        return self._num_motions

    def get_total_length(self):
        return sum(self._motion_lengths)

    def get_motion_num_steps(self, motion_ids=None):
        if motion_ids is None:
            return (self._motion_num_frames * self._sim_fps / self._motion_fps).ceil().int()
        else:
            return (self._motion_num_frames[motion_ids] * self._sim_fps / self._motion_fps[motion_ids]).ceil().int()

    def sample_time(self, motion_ids, truncate_time=None):
        len(motion_ids)
        phase = torch.rand(motion_ids.shape, device=self._device)
        motion_len = self._motion_lengths[motion_ids]
        if truncate_time is not None:
            assert truncate_time >= 0.0
            motion_len -= truncate_time
        motion_time = phase * motion_len
        return motion_time.to(self._device)

    def sample_motions(self, n):
        motion_ids = torch.multinomial(self._sampling_batch_prob, num_samples=n, replacement=True).to(self._device)
        return motion_ids

    def get_motion_length(self, motion_ids=None):
        if motion_ids is None:
            return self._motion_lengths
        else:
            return self._motion_lengths[motion_ids]

    def _calc_frame_blend(self, time, len, num_frames, dt):
        time = time.clone()
        phase = time / len
        phase = torch.clip(phase, 0.0, 1.0)
        time[time < 0] = 0
        frame_idx0 = (phase * (num_frames - 1)).long()
        frame_idx1 = torch.min(frame_idx0 + 1, num_frames - 1)
        blend = torch.clip((time - frame_idx0 * dt) / dt, 0.0, 1.0)
        return (frame_idx0, frame_idx1, blend)

    def _get_num_bodies(self):
        return self.num_bodies

    def _local_rotation_to_dof_smpl(self, local_rot):
        (B, J, _) = local_rot.shape
        dof_pos = quat_to_exp_map(local_rot[:, 1:])
        return dof_pos.reshape(B, -1)
