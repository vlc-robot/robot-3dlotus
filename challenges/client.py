import os
import argparse
import requests
import numpy as np
import random
import jsonlines
from tqdm import tqdm

import lmdb
import msgpack
import msgpack_numpy
msgpack_numpy.patch()

from rlbench.backend.utils import task_file_to_task_class
from pyrep.errors import IKError, ConfigurationPathError
from rlbench.backend.exceptions import InvalidActionError

from genrobo3d.rlbench.environments import RLBenchEnv, Mover


def main(taskvar, server_addr, microstep_data_dir='', output_file=None):
    NUM_EPISODES = 25
    MAX_STEPS = 25
    IMAGE_SIZE = 256

    task_str, variation_id = taskvar.split('+')
    variation_id = int(variation_id)

    if output_file is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    episode_id = 0
    step_id = 0
    
    env = RLBenchEnv(
        data_path=microstep_data_dir,
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

    if microstep_data_dir != '':
        episodes_dir = os.path.join(microstep_data_dir, task_str, f"variation{variation_id}", "episodes")
        demos = []
        if os.path.exists(str(episodes_dir)):
            episode_ids = os.listdir(episodes_dir)
            episode_ids.sort(key=lambda ep: int(ep[7:]))
            for idx, ep in enumerate(episode_ids):
                try:
                    demo = env.get_demo(task_str, variation_id, idx, load_images=False)
                    demos.append(demo)
                except Exception as e:
                    print('\tProblem to load demo_id:', idx, ep)
                    print(e)
        NUM_EPISODES = len(demos)
    else:
        demos = None

    success_rate = 0

    for episode_id in tqdm(range(NUM_EPISODES)):
        if demos is None:
            instructions, obs = task.reset()
        else:
            print("Resetting to demo", episode_id)
            instructions, obs = task.reset_to_demo(demos[episode_id])  # type: ignore

        instruction = random.choice(instructions)
        print(instruction)

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
            # np.save('batch_test.npy', batch)

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
                error_type = None
                obs_state_dict = env.get_observation(obs)  # type: ignore
                if reward == 1:
                    success_rate += 1 / NUM_EPISODES
                    break
                if terminate:
                    print("The episode has terminated!")
            except (IKError, ConfigurationPathError, InvalidActionError) as e:
                print(taskvar, episode_id, step_id, e)
                error_type = str(e)
                reward = 0
                break
    
        if output_file is not None:
            with jsonlines.open(output_file, 'a', flush=True) as outf:
                outf.write({
                    'episode_id': episode_id,
                    'instr': instruction, 
                    'success': reward,
                    'error': error_type,
                    'nsteps': step_id+1,
                })

    print('Success Rate: {:.2f}%'.format(success_rate*100))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--taskvar', default='push_button+0')
    parser.add_argument('--ip', type=str, default="localhost")
    parser.add_argument('--port', type=int, default=13000)
    parser.add_argument('--output_file', type=str, default=None)
    parser.add_argument('--microstep_data_dir', default='')
    args = parser.parse_args()
    server_addr = f"http://{args.ip}:{args.port}/"
    main(args.taskvar, server_addr, args.microstep_data_dir, args.output_file)
