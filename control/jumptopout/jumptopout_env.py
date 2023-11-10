import pydart2 as pydart
from SkateUtils.DartMotionEdit import DartSkelMotion
import numpy as np
from math import exp, pi, log, acos, sqrt
from PyCommon.modules.Math import mmMath as mm
from random import random, randrange
import gym
import gym.spaces
from gym.utils import seeding
import glob
import json
import os
import copy
import itertools


def exp_reward_term(w, exp_w, v):
    norm_sq = sum(_v * _v for _v in v)
    return w * exp(-exp_w * norm_sq/len(v))


def read_json(json_path):
    with open(json_path) as f:
        data = json.load(f)
    kps = []
    for people in data['people']:
        kp = np.array(people['pose_keypoints_2d']).reshape(-1, 3)
        kps.append(kp)
    return kps


def get_contact_pos_avg(skel_body_id, box_body_id, collision_result):
    contact_pos_avg = np.zeros(3)
    contact_pos_cnt = 0
    for contact in collision_result.contacts:
        if contact.skel_id1 == 0 and contact.skel_id2 == 1 and contact.bodynode_id1 == box_body_id and contact.bodynode_id2 == skel_body_id:
            contact_pos_avg += contact.point
            contact_pos_cnt += 1
        if contact.skel_id2 == 0 and contact.skel_id1 == 1 and contact.bodynode_id2 == box_body_id and contact.bodynode_id1 == skel_body_id:
            contact_pos_avg += contact.point
            contact_pos_cnt += 1
    contact_pos_avg /= contact_pos_cnt

    return contact_pos_avg


def to_local(base_body, pos):
    transform = base_body.T
    return np.dot(transform[:3, :3].T, pos - transform[:3, 3])


class SkateDartEnv(gym.Env):
    def __init__(self, env_config={}):
        if "pydart_init" in env_config.keys() and env_config["pydart_init"]:
            pydart.init()
        if "file_dir" in env_config.keys():
            self.base_dir = env_config["file_dir"]
        else:
            self.base_dir = os.path.dirname(os.path.abspath(__file__))

        self.world = pydart.World(1./360., self.base_dir + '/../../data/skel/human_mass_limited_dof_v2.skel')
        self.world.control_skel = self.world.skeletons[1]
        self.skel = self.world.skeletons[1]
        self.skel.body('h_neck').set_collidable(False)
        self.skel.body('h_scapula_left').set_collidable(False)
        self.skel.body('h_scapula_right').set_collidable(False)
        self.skel.body('h_forearm_left').set_collidable(False)
        self.skel.body('h_forearm_right').set_collidable(False)
        self.skel.joint('j_shin_left').set_position_limit_enforced()
        self.skel.joint('j_shin_right').set_position_limit_enforced()
        self.Kp, self.Kd = 600., 49.

        self.ref_world = pydart.World(1./360., self.base_dir + '/../../data/skel/human_mass_limited_dof_v2.skel')
        self.ref_skel = self.ref_world.skeletons[1]
        self.ref_motion = DartSkelMotion()
        self.ref_motion.load(self.base_dir + '/jumptopout.skmo')

        for _i in range(5):
            temp_q = np.asarray(self.ref_motion.get_q(_i))
            temp_q[:3] = mm.logSO3(np.dot(mm.exp(np.array([0., -1.9, 0.])), mm.exp(temp_q[:3])))
            temp_q[4] += 0.02
            temp_q[8] += 0.1
            self.ref_motion.qs[_i][:] = temp_q

        self.ref_motion.refine_dqs(self.ref_skel)
        self.step_per_frame = 15

        self.height_hat_list = [1.2, 0.001]

        # draw box
        self.box_size = []
        self.box_size.append([2.0, self.height_hat_list[0], 0.1])
        self.box_size.append([0.3, self.height_hat_list[1], 0.3])
        self.box0_size = self.box_size[0]
        self.box1_size = self.box_size[1]

        # add box([name], [size], [color])
        self.world.skeletons[0].add_box("box0", self.box0_size, [0.6, 0.6, 0.6])
        self.world.skeletons[0].add_box("box1", self.box1_size, [0.7, 0.7, 0.7])

        self.box_pos = []
        self.box_pos.append(np.array([0., 0.5 * self.box0_size[1], 1.3]))
        self.box_pos.append(np.array([0.2, 0.5 * self.box1_size[1], 2.6]))

        self.box_rot = []
        self.box_rot.append(mm.logSO3(mm.rotY(mm.deg2Rad(0.))))
        self.box_rot.append(mm.logSO3(mm.rotY(mm.deg2Rad(0.))))

        self.world.skeletons[0].body("box1").set_collidable(False)

        self.rsi = True

        self.w_p = 0.35
        self.w_v = 0.1
        self.w_up = 0.2
        self.w_fc = 0.25
        self.w_torque = 0.1
        self.w_root_ori = 0.2

        self.w_par = 0.3

        self.w_exp_h = 0.1
        self.w_h = 0.5

        self.exp_p = 2. * 6.
        self.exp_v = 0.1 * 6.
        self.exp_fc = 1. * 2.
        self.exp_up = 5.
        self.exp_torque = 1.
        self.exp_root_ori = 1.

        self.exp_par = 1.

        self.exp_exp_h = 5.
        self.exp_h = 15.

        self.body_num = self.skel.num_bodynodes()
        self.reward_bodies = [body for body in self.skel.bodynodes]
        self.reward_boxes = [body for body in self.world.skeletons[0].bodynodes[1:]]
        self.idx_e = [self.skel.bodynode_index('h_hand_left'), self.skel.bodynode_index('h_hand_right'),
                      self.skel.bodynode_index('h_heel_left'), self.skel.bodynode_index('h_heel_right')]

        self.body_e = list(map(self.skel.body, self.idx_e))
        self.ref_body_e = list(map(self.ref_skel.body, self.idx_e))
        self.motion_len = len(self.ref_motion)
        self.motion_time = len(self.ref_motion) / self.ref_motion.fps

        self.current_frame = 0
        self.count_frame = 0
        self.max_frame = 30*10

        state_num = len(self.state())
        action_num = self.skel.num_dofs() - 6

        state_high = np.array([np.finfo(np.float32).max] * state_num, dtype=np.float32)
        action_high = np.array([pi*10.] * action_num, dtype=np.float32)

        self.action_space = gym.spaces.Box(-action_high, action_high)
        self.observation_space = gym.spaces.Box(-state_high, state_high)

        self.viewer = None

        self.ext_force = np.zeros(3)
        self.ext_force_duration = 0.

        self.p_fc = None
        self.p_fc_hat = None
        self.p_fc_mask = [-1, -1, -1, -1]
        self.foot_contact_violation = False
        self.contact_violation_num = 0
        self.is_foot_contact_same_list = []

        self.multi_body_collision = False
        self.contact_info = []

        self.up_angle_diff = 0.

        self.up_angle_list = []

        # ground
        self.both_f_contact_start_frame1 = 0
        self.both_f_contact_end_frame1 = 20

        # box0
        self.both_h_contact_start_frame1 = 25
        self.both_h_contact_end_frame1 = 43

        self.lf_contact_start_frame1 = 28
        self.lf_contact_end_frame1 = 35

        self.both_f_contact_start_frame2 = 46
        self.both_f_contact_end_frame2 = 51

        self.rf_contact_start_frame1 = 52
        self.rf_contact_end_frame1 = 57

        # ground
        self.lf_contact_start_frame2 = 61
        self.lf_contact_end_frame2 = 66

        self.rf_contact_start_frame2 = 70
        self.rf_contact_end_frame2 = 74

        self.lf_contact_start_frame3 = 81
        self.lf_contact_end_frame3 = 85

        self.rf_contact_start_frame3 = 90
        self.rf_contact_end_frame3 = 102

        self.lf_contact_start_frame4 = 99
        self.lf_contact_end_frame4 = 102

        self.step_torques = []

        file_name = "jumptopout"
        json_path = self.base_dir + '/../../data/openpose/' + file_name + '/'

        prev_keypoint_backup = None
        prev_up_angle = None
        for json_file in sorted(glob.glob(json_path + "*.json")):
            keypoint = read_json(json_file)

            if len(keypoint) == 0:
                keypoint = prev_keypoint_backup

            head_pos = np.array([keypoint[0][1][0], keypoint[0][1][1]])
            midhip_pos = np.array([keypoint[0][8][0], keypoint[0][8][1]])

            openpose_up_vec = midhip_pos - head_pos

            y_vec = np.array([0, 1])

            up_angle = acos(np.dot(openpose_up_vec, y_vec) / np.linalg.norm(openpose_up_vec))

            if np.isnan(up_angle):
                up_angle = prev_up_angle

            self.up_angle_list.append(up_angle)

            prev_up_angle = up_angle
            prev_keypoint_backup = keypoint

    def state(self):
        pelvis = self.skel.body(0)
        p_pelvis = pelvis.world_transform()[:3, 3]
        R_pelvis = pelvis.world_transform()[:3, :3]

        phase = self.ref_motion.get_frame_looped(self.current_frame)/self.motion_len
        state = [phase]

        p = np.array([np.dot(R_pelvis.T, body.to_world() - p_pelvis) for body in self.skel.bodynodes[1:]]).flatten()
        R = np.array([mm.rot2quat(np.dot(R_pelvis.T, body.world_transform()[:3, :3])) for body in self.skel.bodynodes]).flatten()
        R[:4] = np.asarray(mm.rot2quat(R_pelvis))
        v = np.array([np.dot(R_pelvis.T, body.world_linear_velocity()) for body in self.skel.bodynodes]).flatten()
        w = np.array([np.dot(R_pelvis.T, body.world_angular_velocity())/20. for body in self.skel.bodynodes]).flatten()

        state.extend(p)
        state.extend(np.array([p_pelvis[1]]))
        state.extend(R)
        state.extend(v)
        state.extend(w)

        box_p = np.array([(body.to_world([0, 0.5 * self.box_size[box_idx][1], 0]) - p_pelvis)/10. for box_idx, body in
                          enumerate(self.reward_boxes)]).flatten()

        state.extend(box_p)

        return np.asarray(state).flatten()

    def reward(self):
        self.ref_skel.set_positions(self.ref_motion.get_q(self.current_frame))
        self.ref_skel.set_velocities(self.ref_motion.get_dq(self.current_frame))

        r_p = exp_reward_term(self.w_p, self.exp_p, self.skel.position_differences(self.skel.q, self.ref_skel.q)[6:])
        r_v = exp_reward_term(self.w_v, self.exp_v, self.skel.velocity_differences(self.skel.dq, self.ref_skel.dq)[6:])

        # foot contact reward
        fc_diff = np.clip(np.abs(self.p_fc - self.p_fc_hat), 0., 1.)
        for fc_idx in range(len(self.p_fc)):
            if self.p_fc_hat[fc_idx] == 100:
                fc_diff[fc_idx] = 0.
        r_fc = exp_reward_term(self.w_fc, self.exp_fc, fc_diff)

        # load input up angle data
        # up_angle_hat: given input obtained from video
        up_angle_hat = self.up_angle_list[self.current_frame]

        head_pos = self.skel.joint('j_head').position_in_world_frame()
        pelvis_pos = self.skel.joint('j_pelvis').position_in_world_frame()

        sim_up_vec = head_pos - pelvis_pos
        y_axis = np.array([0, 1., 0])

        up_angle = acos(np.dot(sim_up_vec, y_axis) / np.linalg.norm(sim_up_vec))

        self.up_angle_diff = abs(up_angle - up_angle_hat)
        r_up = exp_reward_term(self.w_up, self.exp_up, [self.up_angle_diff])

        # minimize joint torque
        torque = sum(self.step_torques)/len(self.step_torques)
        r_torque = exp_reward_term(self.w_torque, self.exp_torque, torque)

        reward = r_p + r_v + r_fc + r_up + r_torque

        # pelvis orientation
        pelvis_ori_diff = []
        pelvis_hint_ranges = [
            (self.both_h_contact_start_frame1, self.both_f_contact_end_frame2, 0),
        ]
        for pelvis_hint_range in pelvis_hint_ranges:
            pelvis_box_idx = pelvis_hint_range[2]
            if pelvis_hint_range[0] - 2 <= self.current_frame <= pelvis_hint_range[1]:
                orientation_body_list = ['h_pelvis', 'h_abdomen', 'h_spine']
                pelvis_ori_diff.extend([1. - np.dot(self.skel.body(body_name).world_transform()[:3, 0], self.world.skeletons[0].body(pelvis_box_idx+1).world_transform()[:3, 0]) for body_name in orientation_body_list])

        if len(pelvis_ori_diff) > 0:
            r_root_ori = exp_reward_term(self.w_root_ori, self.exp_root_ori, np.asarray(pelvis_ori_diff))
            reward = (1. - self.w_root_ori) * reward + r_root_ori

        # parabola hint
        parabola_diff =[]
        parabola_hint_ranges = [
            (self.both_f_contact_end_frame1, self.both_h_contact_start_frame1, ['h_hand_left', 'h_hand_right'], 0),
            (self.rf_contact_end_frame1, self.lf_contact_start_frame2, ['h_heel_left'], 1),
        ]
        for parabola_hint_range in parabola_hint_ranges:
            parabola_box_idx = parabola_hint_range[3]
            if parabola_hint_range[0] < self.current_frame < parabola_hint_range[1]:
                dt = (parabola_hint_range[1] - self.current_frame) / self.ref_motion.fps
                exp_com_pos = self.skel.com() + dt * self.skel.com_velocity() + 0.5 * dt * dt * self.world.gravity()
                exp_diff_com_to_contact = self.box_pos[parabola_box_idx] + self.height_hat_list[parabola_box_idx] * mm.unitY()/2. - exp_com_pos
                norm_exp_diff_com_to_contact = np.linalg.norm(exp_diff_com_to_contact)
                self.ref_skel.set_positions(self.ref_motion.get_q(parabola_hint_range[1]))
                pos_diff_ref_com_to_ref_contact = \
                    [norm_exp_diff_com_to_contact - np.linalg.norm(self.ref_skel.body(body_name).to_world() - self.ref_skel.com()) for body_name in parabola_hint_range[2]]
                self.ref_skel.set_positions(self.ref_motion.get_q(self.current_frame))
                self.ref_skel.set_velocities(self.ref_motion.get_dq(self.current_frame))
                parabola_diff.extend(pos_diff_ref_com_to_ref_contact)

        if len(parabola_diff) > 0:
            r_par = exp_reward_term(self.w_par, self.exp_par, np.asarray(parabola_diff))
            reward = (1. - self.w_par) * reward + r_par

        hand_margin = 0.15

        foot_margin = 0.0249

        lowest_contact_margin = 0.0

        def get_lowest_body_point(body: pydart.BodyNode):
            lowest_value = 100.
            lowest_point = np.zeros(3)
            for perm in itertools.product([-1, 1], repeat=3):
                point = body.to_world(np.multiply(np.asarray(body.shapenodes[0].shape.size())/2., np.asarray(perm)))
                if point[1] < lowest_value:
                    lowest_point = point
                    lowest_value = point[1]

            return lowest_point

        # height hint
        exp_contact_body_height_diff = []
        if self.both_h_contact_start_frame1 - 2 <= self.current_frame < self.both_h_contact_start_frame1:
            left_frame = self.both_h_contact_start_frame1 - self.current_frame
            exp_contact_body_height_diff.append(get_lowest_body_point(self.skel.body('h_hand_left'))[1] - self.height_hat_list[0] - 0.1 * left_frame - lowest_contact_margin)
            exp_contact_body_height_diff.append(get_lowest_body_point(self.skel.body('h_hand_right'))[1] - self.height_hat_list[0] - 0.1 * left_frame - lowest_contact_margin)
            exp_contact_body_height_diff.append(get_lowest_body_point(self.skel.body('h_hand_left'))[1] - get_lowest_body_point(self.skel.body('h_hand_right'))[1])

        if self.both_f_contact_start_frame2 - 2 <= self.current_frame < self.both_f_contact_start_frame2:
            left_frame = self.both_f_contact_start_frame2 - self.current_frame
            exp_contact_body_height_diff.append(get_lowest_body_point(self.skel.body('h_heel_left'))[1] - self.height_hat_list[0] - 0.2 * left_frame - lowest_contact_margin)
            exp_contact_body_height_diff.append(get_lowest_body_point(self.skel.body('h_heel_right'))[1] - self.height_hat_list[0] - 0.2 * left_frame - lowest_contact_margin)
            exp_contact_body_height_diff.append(get_lowest_body_point(self.skel.body('h_heel_left'))[1] - get_lowest_body_point(self.skel.body('h_heel_right'))[1])

        if self.lf_contact_start_frame2 - 2 <= self.current_frame < self.lf_contact_start_frame2:
            left_frame = self.lf_contact_start_frame2 - self.current_frame
            exp_contact_body_height_diff.append(get_lowest_body_point(self.skel.body('h_heel_left'))[1] - 0.1 * left_frame - lowest_contact_margin)
        if self.rf_contact_start_frame2 - 2 <= self.current_frame < self.rf_contact_start_frame2:
            left_frame = self.rf_contact_start_frame2 - self.current_frame
            exp_contact_body_height_diff.append(get_lowest_body_point(self.skel.body('h_heel_right'))[1] - 0.1 * left_frame - lowest_contact_margin)
        if self.lf_contact_start_frame3 - 2 <= self.current_frame < self.lf_contact_start_frame3:
            left_frame = self.lf_contact_start_frame3 - self.current_frame
            exp_contact_body_height_diff.append(get_lowest_body_point(self.skel.body('h_heel_left'))[1] - 0.1 * left_frame - lowest_contact_margin)
        if self.rf_contact_start_frame3 - 2 <= self.current_frame < self.rf_contact_start_frame3:
            left_frame = self.rf_contact_start_frame3 - self.current_frame
            exp_contact_body_height_diff.append(get_lowest_body_point(self.skel.body('h_heel_right'))[1] - 0.1 * left_frame - lowest_contact_margin)
        if self.lf_contact_start_frame4 - 2 <= self.current_frame < self.lf_contact_start_frame4:
            left_frame = self.lf_contact_start_frame4 - self.current_frame
            exp_contact_body_height_diff.append(get_lowest_body_point(self.skel.body('h_heel_left'))[1] - 0.1 * left_frame - lowest_contact_margin)

        if len(exp_contact_body_height_diff) > 0:
            r_exp_h = exp_reward_term(self.w_exp_h, self.exp_exp_h, np.asarray(exp_contact_body_height_diff))
            reward = (1.-self.w_exp_h) * reward + r_exp_h

        contact_body_height_diff = []
        if self.both_f_contact_start_frame1 <= self.current_frame <= self.both_f_contact_end_frame1:
            exp_contact_body_height_diff.append(get_lowest_body_point(self.skel.body('h_heel_left'))[1] - lowest_contact_margin)
            exp_contact_body_height_diff.append(get_lowest_body_point(self.skel.body('h_heel_right'))[1] - lowest_contact_margin)
            exp_contact_body_height_diff.append(get_lowest_body_point(self.skel.body('h_heel_left'))[1] - get_lowest_body_point(self.skel.body('h_heel_right'))[1])

        height_hint_ranges = [
            (self.both_h_contact_start_frame1, self.both_h_contact_end_frame1, ['h_hand_left', 'h_hand_right'], 0, (True, True, True), (0., 0., 0.)),
            (self.both_f_contact_start_frame2, self.both_f_contact_end_frame2, ['h_heel_left', 'h_heel_right'], 0, (True, True, True), (0., 0., 0.1+self.box0_size[2]/2.)),
            (self.lf_contact_start_frame2, self.lf_contact_end_frame2, ['h_heel_left'], 1, (True, True, True), (0., 0., 0.)),
        ]
        for height_hint_range in height_hint_ranges:
            height_box_idx = height_hint_range[3]
            if height_hint_range[0] <= self.current_frame <= height_hint_range[1]:
                for height_hint_body_name in height_hint_range[2]:
                    local_pos = to_local(self.world.skeletons[0].body(height_box_idx+1), self.skel.body(height_hint_body_name).to_world())
                    if height_hint_range[4][0]:
                        contact_body_height_diff.append(max(0., abs(local_pos[0]) - self.box_size[height_box_idx][0]/4.))
                    if height_hint_range[4][1]:
                        if 'hand' in height_hint_body_name:
                            contact_body_height_diff.append(local_pos[1] - self.box_size[height_box_idx][1]/2. - hand_margin)
                        elif 'heel' in height_hint_body_name:
                            contact_body_height_diff.append(local_pos[1] - self.box_size[height_box_idx][1]/2. - foot_margin)
                    if height_hint_range[4][2]:
                        contact_body_height_diff.append(local_pos[2] - height_hint_range[5][2])
                if len(height_hint_range[2]) == 2:
                    contact_body_height_diff.append(get_lowest_body_point(self.skel.body(height_hint_range[2][0]))[1] - get_lowest_body_point(self.skel.body(height_hint_range[2][1]))[1])

        if self.lf_contact_start_frame3 <= self.current_frame <= self.lf_contact_end_frame3 or \
            self.lf_contact_start_frame4 <= self.current_frame <= self.lf_contact_end_frame4:
            exp_contact_body_height_diff.append(get_lowest_body_point(self.skel.body('h_heel_left'))[1] - lowest_contact_margin)

        if self.lf_contact_start_frame4 <= self.current_frame <= self.lf_contact_end_frame4:
            exp_contact_body_height_diff.append(get_lowest_body_point(self.skel.body('h_heel_left'))[1] - get_lowest_body_point(self.skel.body('h_heel_right'))[1])

        if self.rf_contact_start_frame2 <= self.current_frame <= self.rf_contact_end_frame2 or \
                self.rf_contact_start_frame3 <= self.current_frame <= self.rf_contact_end_frame3:
            exp_contact_body_height_diff.append(get_lowest_body_point(self.skel.body('h_heel_right'))[1] - lowest_contact_margin)

        if len(contact_body_height_diff) > 0:
            r_h = exp_reward_term(self.w_h, self.exp_h, np.asarray(contact_body_height_diff))
            reward = (1.-self.w_h) * reward + r_h

        return reward

    def is_done(self):
        if self.multi_body_collision:
            # print("multiple body collision with a box occurs!")
            return True

        if self.foot_contact_violation:
            # print('not follow the contact hint too long', self.current_frame)
            return True

        if self.skel.com()[1] < 0.3:
            # print('fallen')
            return True

        if self.up_angle_diff > mm.deg2Rad(45.):
            # print('up angle is too far from openpose result')
            return True

        elif self.skel.body('h_head') in self.world.collision_result.contacted_bodies:
            # print('head collision')
            return True

        elif True in np.isnan(np.asarray(self.skel.q)) or True in np.isnan(np.asarray(self.skel.dq)):
            # print('nan')
            return True

        elif self.ref_motion.has_loop and self.count_frame >= self.max_frame:
            # print('timeout1')
            return True
        elif not self.ref_motion.has_loop and self.current_frame == self.motion_len - 1:
            # print('timeout2')
            return True
        return False

    def step(self, _action):
        action = np.hstack((np.zeros(6), _action/10.))

        next_frame = self.current_frame + 1
        self.ref_skel.set_positions(self.ref_motion.get_q(next_frame))
        self.ref_skel.set_velocities(self.ref_motion.get_dq(next_frame))

        h = self.world.time_step()
        q_des = self.ref_skel.q + action

        self.step_torques = []
        for _ in range(self.step_per_frame):
            if self.ext_force_duration > 0.:
                self.skel.body('h_spine').add_ext_force(self.ext_force, _isForceLocal=True)
                self.ext_force_duration -= h
                if self.ext_force_duration < 0.:
                    self.ext_force_duration = 0.
            torques = self.skel.get_spd_forces(q_des, self.Kp, self.Kd)
            self.step_torques.append(torques)
            self.skel.set_forces(torques)
            self.world.step()

        self.current_frame = next_frame
        self.count_frame += 1

        # ================================================================

        del self.contact_info[:]
        for contact in self.world.collision_result.contacts:
            if contact.skel_id1 == 0 and contact.skel_id2 == 1:
                self.contact_info.append((contact.bodynode_id1, contact.bodynode_id2))
            elif contact.skel_id1 == 1 and contact.skel_id2 == 0:
                self.contact_info.append((contact.bodynode_id2, contact.bodynode_id1))

        self.contact_info = list(set(self.contact_info))
        if self.contact_info:
            _, skel_only_contact_info = zip(*self.contact_info)
        else:
            skel_only_contact_info = []
        skel_only_contact_info = list(skel_only_contact_info)
        # [lf, rf, lh, rh]
        end_effector_contact_num = sum([c_info[1] in [3, 6, 14, 18] for c_info in self.contact_info])
        no_end_effector_contact_num = len(self.contact_info) - end_effector_contact_num
        self.multi_body_collision = no_end_effector_contact_num > 0

        # box contact info
        # -1 : no contact or multiple contact
        # 0: ground
        # 1: box0
        # 2: box1

        self.p_fc = np.array([-1, -1, -1, -1])
        body_fc_indices = [3, 6, 14, 18]
        for fc_idx, body_fc_index in enumerate(body_fc_indices):
            if skel_only_contact_info.count(body_fc_index) == 1:
                self.p_fc[fc_idx] = self.contact_info[skel_only_contact_info.index(body_fc_index)][0]
            if self.p_fc_mask[fc_idx] != -1:
                self.p_fc[fc_idx] = self.p_fc_mask[fc_idx]

        self.p_fc_hat = np.array([-1, -1, -1, -1])

        if self.both_f_contact_start_frame1 <= self.current_frame <= self.both_f_contact_end_frame1:
            self.p_fc_hat[0] = 0
            self.p_fc_hat[1] = 0

        if self.both_h_contact_start_frame1 <= self.current_frame <= self.both_h_contact_end_frame1:
            self.p_fc_hat[2] = 1
            self.p_fc_hat[3] = 1

        if self.lf_contact_start_frame1 <= self.current_frame <= self.lf_contact_end_frame1:
            self.p_fc_hat[0] = 1

        if self.both_f_contact_start_frame2 <= self.current_frame <= self.both_f_contact_end_frame2:
            self.p_fc_hat[0] = 1
            self.p_fc_hat[1] = 1
            for fc_idx in [0, 1]:
                if self.p_fc[fc_idx] == 1:
                    body = self.skel.body(body_fc_indices[fc_idx])
                    c_avg_pos = get_contact_pos_avg(body.id, 1, self.world.collision_result)
                    local_pos = to_local(body, c_avg_pos)
                    if c_avg_pos[2] < self.box_pos[0][2] - self.box0_size[2] * 0.05:
                        if (body_fc_indices[fc_idx] == 3 and local_pos[2] > 0.05) or \
                            (body_fc_indices[fc_idx] == 6 and local_pos[2] < -0.05):
                            self.p_fc[fc_idx] -= 0.5

        if self.rf_contact_start_frame1 <= self.current_frame <= self.rf_contact_end_frame1:
            self.p_fc_hat[1] = 1

        if self.lf_contact_start_frame2 <= self.current_frame <= self.lf_contact_end_frame2 or \
                self.lf_contact_start_frame3 <= self.current_frame <= self.lf_contact_end_frame3 or \
                self.lf_contact_start_frame4 <= self.current_frame <= self.lf_contact_end_frame4:
            self.p_fc_hat[0] = 0

        if self.rf_contact_start_frame2 <= self.current_frame <= self.rf_contact_end_frame2 or \
                self.rf_contact_start_frame3 <= self.current_frame <= self.rf_contact_end_frame3:
            self.p_fc_hat[1] = 0

        if self.current_frame in [self.both_h_contact_end_frame1]:
            self.p_fc_mask[2] = -1
            self.p_fc_mask[3] = -1
            self.skel.body(body_fc_indices[2]).set_collidable(True)
            self.skel.body(body_fc_indices[3]).set_collidable(True)

        if self.current_frame > 5:
            diff_num = sum(self.p_fc[fc_idx] != self.p_fc_hat[fc_idx] and self.p_fc_hat[fc_idx] != 100 for fc_idx in range(len(self.p_fc)))
            self.is_foot_contact_same_list.append(1 if diff_num > 0 else 0)
        else:
            self.is_foot_contact_same_list.append(0)

        if len(self.is_foot_contact_same_list) < 30:
            self.foot_contact_violation = sum(self.is_foot_contact_same_list) > 7
        else:
            self.foot_contact_violation = sum(self.is_foot_contact_same_list[-30:]) > 7

        return tuple([self.state(), self.reward(), self.is_done(), {}])

    def continue_from_frame(self, frame):
        self.current_frame = frame
        self.ref_skel.set_positions(self.ref_motion.get_q(self.current_frame))
        skel_pelvis_offset = self.skel.joint(0).position_in_world_frame() - self.ref_skel.joint(0).position_in_world_frame()
        skel_pelvis_offset[1] = 0.

    def reset(self):
        self.world.reset()
        self.ref_motion.refine_dqs(self.ref_skel)
        self.continue_from_frame(0)
        self.skel.set_positions(self.ref_motion.get_q(self.current_frame))
        self.skel.set_velocities(np.asarray(self.ref_motion.get_dq(self.current_frame)))

        self.count_frame = 0
        self.foot_contact_violation = False
        self.contact_violation_num = 0

        del self.contact_info[:]
        del self.is_foot_contact_same_list[:]

        box_q = self.world.skeletons[0].q
        box_q[9:12] = self.box_pos[0]
        box_q[15:18] = self.box_pos[1]

        self.world.skeletons[0].set_positions(box_q)

        return self.state()

    def render(self, mode='human', close=False):
        return None

    def close(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def flag_rsi(self, rsi=True):
        self.rsi = rsi

    def hard_reset(self):
        self.__init__()
        self.reset()