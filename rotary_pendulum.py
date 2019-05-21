from gym.envs.mujoco import mujoco_env
from gym import utils
import numpy as np
import os

def normalize_angle_1(angle):
    '''Normalize an angle from [0, 2pi] to [-pi, pi]'''
    pass

def normalize_angle_2(angle):
    '''Normalize an angle from [-pi, pi] to [0, 2pi]'''
    pass



class RotaryPendulumEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, swingup=False):
        utils.EzPickle.__init__(self)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, dir_path+'/rotary_pendulum.xml', 2)
        self.swingup = swingup

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        theta, alpha_un, theta_dot, alpha_dot = self._get_obs()
        alpha = ((alpha_un % (2 * np.pi)) + np.pi) % (2 * np.pi)

        # # reward = 1 - 0.5 * np.abs(alpha) - 0.5 * np.abs(theta)
        # if np.abs(theta) > (20 * (np.pi / 180)) or np.abs(alpha) > (20 * (np.pi / 180)):
        #     reward = 1 + np.cos(alpha)
        # else:
        #     reward = 10 * (1 - np.abs(alpha) - 0.1 * np.abs(theta))

        reward = 1 + np.cos(alpha)

        ob = np.array([
            np.cos(theta),
            np.sin(theta),
            np.cos(alpha),
            np.sin(alpha),
            theta_dot,
            alpha_dot,
        ])
        return ob, reward, self._done(theta, alpha), {}
        # return ob, reward, False, {}

    def _done(self, theta, alpha):
        done = np.abs(theta) > (90 * (np.pi / 180))
        done |= np.abs(alpha) > (20 * (np.pi / 180))
        # print('alpha={:4.4} done={}'.format(alpha, done))
        return done

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
        if not self.swingup:
            qpos[1] += np.pi
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        
        self.set_state(qpos, qvel)
        theta, alpha, theta_dot, alpha_dot = self._get_obs()

        ob = np.array([
            np.cos(theta),
            np.sin(theta),
            np.cos(alpha),
            np.sin(alpha),
            theta_dot,
            alpha_dot,
        ])
        # print('reseting')
        return ob

    def _get_obs(self):
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent

    def close(self):
        pass


class RotaryPendulumSwingupEnv(RotaryPendulumEnv):
    def __init__(self):
        super(RotaryPendulumSwingupEnv, self).__init__(swingup=True)

    def _done(self, theta, alpha):
        done = np.abs(theta) > (115 * (np.pi / 180))
        return done

