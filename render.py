#!/usr/bin/env python3
from rotary_pendulum import RotaryPendulumEnv
from gym_brt.control import NoControl, \
                             RandomControl, \
                             QubeFlipUpControl, \
                             QubeHoldControl#, \
                             # QubeDampenControl
import numpy as np
import argparse
import gym


parser = argparse.ArgumentParser(description='Renders a Gym environment for quick inspection.')
parser.add_argument('--controller', '-c', type=str, default='flip')
args = parser.parse_args()

controllers_dict = {
    'none': NoControl,
    'rand': RandomControl,
    'flip': QubeFlipUpControl,
    'hold': QubeHoldControl,
    # 'damp': QubeDampenControl,
}
Controller = controllers_dict[args.controller]


env = RotaryPendulumEnv()
env.reset()

step = 0
obs = env.reset()
ctrl_sys = Controller(env, frequency=1000)

while True:
    action = ctrl_sys.action(obs)
    # action = env.action_space.sample()
    obs, reward, _, _ = env.step(action)

    # obs = env.step(1 * (env.action_space.sample()))
    env.render()
    if step % 3000 == 0:
        obs = env.reset()
    step += 1
