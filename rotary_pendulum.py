from gym.envs.mujoco import mujoco_env
from gym import utils
import numpy as np
import os 

class RotaryPendulumEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, swingup=False):
        utils.EzPickle.__init__(self)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        print(dir_path+'/rotary_pendulum.xml')
        mujoco_env.MujocoEnv.__init__(self, dir_path+'/rotary_pendulum.xml', 2)
        self.swingup = swingup

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        theta, alpha, theta_dot, alpha_dot = self._get_obs()
        reward = np.cos(alpha) - 0.01 * np.abs(theta)

        ob = np.array([
            np.cos(theta),
            np.sin(theta),
            np.cos(alpha),
            np.sin(alpha),
            theta_dot,
            alpha_dot,
        ])
        return ob, reward, False, {}

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
        return ob

    def _get_obs(self):
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent


class RotaryPendulumSwingupEnv(RotaryPendulumEnv):
    def __init__(self):
        super(RotaryPendulumSwingupEnv, self).__init__(swingup=True)

