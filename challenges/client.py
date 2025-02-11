import os
import argparse
import requests
import numpy as np
import random
from tqdm import tqdm

import lmdb
import msgpack
import msgpack_numpy
msgpack_numpy.patch()

from rlbench.backend.utils import task_file_to_task_class
from pyrep.errors import IKError, ConfigurationPathError
from rlbench.backend.exceptions import InvalidActionError

from genrobo3d.rlbench.environments import RLBenchEnv, Mover


NUM_EPISODES = 25
MAX_STEPS = 25
IMAGE_SIZE = 256


def main(server_addr):
    taskvar = 'push_button+0'
    task_str, variation_id = taskvar.split('+')
    variation_id = int(variation_id)

    episode_id = 0
    step_id = 0
    
    env = RLBenchEnv(
        data_path='',
        apply_rgb=True,
        apply_pc=True,
        apply_mask=False,
        headless=True,
        image_size=[IMAGE_SIZE, IMAGE_SIZE],
        cam_rand_factor=0,
    )
    env.env.launch()
    task_type = task_file_to_task_class(task_str)
    task = env.env.get_task(task_type)
    task.set_variation(variation_id)
    move = Mover(task, max_tries=10)

    success_rate = 0

    for episode_id in tqdm(range(NUM_EPISODES)):
        instructions, obs = task.reset()
        instruction = random.choice(instructions)
        # print(instruction)

        obs_state_dict = env.get_observation(obs)
        move.reset(obs_state_dict['gripper'])

        for step_id in range(MAX_STEPS):
            batch = {
                'taskvar': taskvar,
                'episode_id': episode_id,
                'step_id': step_id,
                'instruction': instruction,
                'obs_state_dict': obs_state_dict,
            }

            data = msgpack_numpy.packb(batch)
            # print(f"Calling the server {server_addr}")
            response = requests.post(f"{server_addr}/predict", data=data)
            action = msgpack_numpy.unpackb(response._content)
            # print('Step id', step_id, action)

            if action is None:
                break

            # update the observation based on the predicted action
            try:
                obs, reward, terminate, _ = move(action, verbose=False)
                obs_state_dict = env.get_observation(obs)  # type: ignore
                if reward == 1:
                    success_rate += 1 / NUM_EPISODES
                    break
                if terminate:
                    print("The episode has terminated!")
            except (IKError, ConfigurationPathError, InvalidActionError) as e:
                print(taskvar, episode_id, step_id, e)
                reward = 0
                break
    
    print('Success Rate: {:.2f}%'.format(success_rate*100))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--ip', type=str, default="localhost")
    parser.add_argument('--port', type=int, default=13000)
    args = parser.parse_args()
    server_addr = f"http://{args.ip}:{args.port}/"
    main(server_addr)