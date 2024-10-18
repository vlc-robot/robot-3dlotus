from typing import Tuple, Dict, List

import os
import json
import jsonlines
import torch.multiprocessing as mp
import tap
from termcolor import colored
import numpy as np
import yaml
from easydict import EasyDict

from genrobo3d.rlbench.environments import RLBenchEnv, Mover
from rlbench.backend.utils import task_file_to_task_class
from pyrep.errors import IKError, ConfigurationPathError
from rlbench.backend.exceptions import InvalidActionError

from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor
from genrobo3d.rlbench.recorder import (
    TaskRecorder, StaticCameraMotion, CircleCameraMotion, AttachedCameraMotion
)

from genrobo3d.train.utils.misc import set_random_seed
from genrobo3d.evaluation.common import write_to_file

from genrobo3d.evaluation.robot_pipeline_gt import GroundtruthRobotPipeline
from genrobo3d.evaluation.robot_pipeline import RobotPipeline


class ServerArguments(tap.Tap):
    full_gt: bool = False
    pipeline_config_file: str

    device: str = 'cuda'  # cpu, cuda

    # motion planner
    mp_expr_dir: str = None
    mp_ckpt_step: int = None

    image_size: List[int] = [256, 256]
    max_tries: int = 10
    max_steps: int = 25

    microstep_data_dir: str = ''
    seed: int = 100  # seed for RLBench
    num_workers: int = 4
    queue_size: int = 20

    taskvar_file: str = 'assets/taskvars_train.json'
    num_demos: int = 20

    save_obs_outs: bool = False

    best_disc_pos: str = 'max' # max, ens1

    record_video: bool = False
    video_dir: str = None
    not_include_robot_cameras: bool = False
    video_rotate_cam: bool = False
    video_resolution: int = 480

    pc_label_type: str = None
    gt_og_label_file: str = 'assets/taskvars_train_target_label.json'

    run_action_step: int = 1

    llm_cache_file: str = None
    no_gt_llm: bool = False
    llm_master_port: int = None


def consumer_fn(args, pipeline_config, batch_queue, result_queues):
    print('consumer start')
    set_random_seed(args.seed)

    # build model
    if args.full_gt:
        actioner = GroundtruthRobotPipeline(pipeline_config)
    else:
        actioner = RobotPipeline(pipeline_config)

    while True:
        print('while loop consumer starts')
        data = batch_queue.get()
        print(f'get data from batch_queue')
        if data is None:
            print('Received None value -> Producers finished.')
            break

        # run one batch
        k_prod, batch = data
        out = actioner.predict(**batch)
        print(f"put data to result_queue: {out['action']}")
        result_queues[k_prod].put(out)
        print('put data to result_queue done')


def producer_fn(
    proc_id, k_res, args, pipeline_config, taskvar, pred_file, 
    batch_queue, result_queue, producer_queue
):

    task_str, variation = taskvar.split('+')
    variation = int(variation)

    set_random_seed(args.seed)

    if args.microstep_data_dir != '':
        episodes_dir = os.path.join(args.microstep_data_dir, task_str, f"variation{variation}", "episodes")
        if not os.path.exists(str(episodes_dir)):
            print(f'{taskvar} does not need to be evaluated.')
            producer_queue.put((proc_id, k_res))
            return

    env = RLBenchEnv(
        data_path=args.microstep_data_dir,
        apply_rgb=True,
        apply_pc=True,
        apply_mask=pipeline_config.object_grounding.use_groundtruth,
        headless=True,
        image_size=args.image_size,
        cam_rand_factor=0,
    )

    env.env.launch()
    task_type = task_file_to_task_class(task_str)
    task = env.env.get_task(task_type)
    task.set_variation(variation)  # type: ignore

    if args.record_video:
        # Add a global camera to the scene
        cam_placeholder = Dummy('cam_cinematic_placeholder')
        cam_resolution = [args.video_resolution, args.video_resolution]
        cam = VisionSensor.create(cam_resolution)
        cam.set_pose(cam_placeholder.get_pose())
        cam.set_parent(cam_placeholder)

        if args.video_rotate_cam:
            global_cam_motion = CircleCameraMotion(cam, Dummy('cam_cinematic_base'), 0.005)
        else:
            global_cam_motion = StaticCameraMotion(cam)

        cams_motion = {"global": global_cam_motion}

        if not args.not_include_robot_cameras:
            # Env cameras
            cam_left = VisionSensor.create(cam_resolution)
            cam_right = VisionSensor.create(cam_resolution)
            cam_wrist = VisionSensor.create(cam_resolution)

            left_cam_motion = AttachedCameraMotion(cam_left, task._scene._cam_over_shoulder_left)
            right_cam_motion = AttachedCameraMotion(cam_right, task._scene._cam_over_shoulder_right)
            wrist_cam_motion = AttachedCameraMotion(cam_wrist, task._scene._cam_wrist)

            cams_motion["left"] = left_cam_motion
            cams_motion["right"] = right_cam_motion
            cams_motion["wrist"] = wrist_cam_motion
        tr = TaskRecorder(cams_motion, fps=30)
        task._scene.register_step_callback(tr.take_snap)

        video_log_dir = os.path.join(args.video_dir, f'{task_str}+{variation}')
        os.makedirs(str(video_log_dir), exist_ok=True)

    move = Mover(task, max_tries=args.max_tries)

    if args.microstep_data_dir != '':
        episodes_dir = os.path.join(args.microstep_data_dir, task_str, f"variation{variation}", "episodes")
        demos = []
        if os.path.exists(str(episodes_dir)):
            episode_ids = os.listdir(episodes_dir)
            episode_ids.sort(key=lambda ep: int(ep[7:]))
            for idx, ep in enumerate(episode_ids):
                try:
                    demo = env.get_demo(task_str, variation, idx, load_images=False)
                    demos.append(demo)
                except Exception as e:
                    print('\tProblem to load demo_id:', idx, ep)
                    print(e)
        if len(demos) == 0:
            print(f'{taskvar} does not need to be evaluated.')
            return
    else:
        demos = None

    num_demos = len(demos) if demos is not None else args.num_demos

    success_rate = 0.0
    for demo_id in range(num_demos):
        reward = None

        if demos is None:
            instructions, obs = task.reset()
        else:
            instructions, obs = task.reset_to_demo(demos[demo_id])

        obs_state_dict = env.get_observation(obs)  # type: ignore
        move.reset(obs_state_dict['gripper'])
        # print('reset')

        cache = None
        for step_id in range(args.max_steps):
            # fetch the current observation, and predict one action
            batch = {
                'task_str': task_str,
                'variation': variation,
                'step_id': step_id,
                'obs_state_dict': obs_state_dict,
                'episode_id': demo_id,
                'instructions': instructions,
                'cache': cache,
            }
            batch_queue.put((k_res, batch))

            output = result_queue.get()
            action = output["action"]
            cache =  output["cache"]

            if action is None:
                break

            # update the observation based on the predicted action
            try:
                # print('get new step', step_id)
                print('Before move', step_id)
                obs, reward, terminate, _ = move(action, verbose=True)
                print('After move', step_id)
                # print('finish mover')
                obs_state_dict = env.get_observation(obs)  # type: ignore
                # print('finish get new step')

                if reward == 1:
                    success_rate += 1 / num_demos
                    break
                if terminate:
                    print("The episode has terminated!")
            except (IKError, ConfigurationPathError, InvalidActionError) as e:
                print(taskvar, demo_id, step_id, e)
                reward = 0
                break

        if args.record_video: # and reward < 1:
            tr.save(os.path.join(video_log_dir, f"{demo_id}_SR{reward}"))

        print(
            taskvar, "Demo", demo_id, 'Step', step_id+1,
            "Reward", reward, "Accumulated SR: %.2f" % (success_rate * 100),
            'Estimated SR: %.2f' % (success_rate * num_demos / (demo_id+1) * 100)
        )
    
    write_to_file(
        pred_file,
        {
            'checkpoint': pipeline_config.motion_planner.ckpt_step,
            'task': task_str, 'variation': variation,
            'num_demos': num_demos, 'sr': success_rate
        }
    )

    env.env.shutdown()
    print(colored(f'Taskvar: {taskvar} SR: {success_rate:.2f}', 'black', 'on_yellow'))
    producer_queue.put((proc_id, k_res))
    

def main():
    # To use gpu in subprocess: https://pytorch.org/docs/stable/notes/multiprocessing.html
    mp.set_start_method('spawn')

    args = ServerArguments().parse_args(known_only=True)
    args.remained_args = args.extra_args

    with open(args.pipeline_config_file, 'r') as f:
        pipeline_config = yaml.safe_load(f)
    pipeline_config = EasyDict(pipeline_config)

    if args.no_gt_llm:
        pipeline_config.llm_planner.use_groundtruth = False
    if args.llm_cache_file is not None:
        pipeline_config.llm_planner.cache_file = args.llm_cache_file
    if args.llm_master_port is not None:
        pipeline_config.llm_planner.master_port = args.llm_master_port

    if args.gt_og_label_file is not None:
        pipeline_config.object_grounding.gt_label_file = args.gt_og_label_file
    if args.pc_label_type is not None:
        pipeline_config.motion_planner.pc_label_type = args.pc_label_type
    pipeline_config.motion_planner.run_action_step = args.run_action_step

    pred_dirname = 'preds'
    if pipeline_config.llm_planner.use_groundtruth:
        pred_dirname += '-llm_gt'
    if pipeline_config.object_grounding.use_groundtruth:
        pred_dirname += f'-og_gt_{pipeline_config.motion_planner.pc_label_type}'
    if pipeline_config.motion_planner.run_action_step > 1:
        pred_dirname += f'-runstep{pipeline_config.motion_planner.run_action_step}'       
    pred_dir = os.path.join(args.mp_expr_dir, pred_dirname, f'seed{args.seed}')
    os.makedirs(pred_dir, exist_ok=True)
    pred_file = os.path.join(pred_dir, 'results.jsonl')

    if args.mp_expr_dir is None:
        args.mp_expr_dir = pipeline_config.motion_planner.expr_dir
    if args.mp_ckpt_step is None:
        args.mp_ckpt_step = pipeline_config.motion_planner.ckpt_step
    mp_checkpoint_file = os.path.join(
        args.mp_expr_dir, 'ckpts', f'model_step_{args.mp_ckpt_step}.pt'
    )
    if not os.path.exists(mp_checkpoint_file):
        print(mp_checkpoint_file, 'not exists')
        return
    
    pipeline_config.motion_planner.expr_dir = args.mp_expr_dir
    pipeline_config.motion_planner.ckpt_step = args.mp_ckpt_step
    pipeline_config.motion_planner.checkpoint = mp_checkpoint_file
    pipeline_config.motion_planner.config_file = os.path.join(
        args.mp_expr_dir, 'logs', 'training_config.yaml'
    )
    pipeline_config.motion_planner.save_obs_outs = args.save_obs_outs
    pipeline_config.motion_planner.pred_dir = pred_dir
    
    existed_taskvars = set()
    if os.path.exists(pred_file):
        with jsonlines.open(pred_file, 'r') as f:
            for item in f:
                item_step = item['checkpoint']
                if item_step == args.mp_ckpt_step:
                    existed_taskvars.add(f"{item['task']}+{item['variation']}")

    taskvars = json.load(open(args.taskvar_file))
    taskvars = [taskvar for taskvar in taskvars if taskvar not in existed_taskvars]
    print('checkpoint', args.mp_ckpt_step, '#taskvars', len(taskvars))

    batch_queue = mp.Queue(args.queue_size)
    result_queues = [mp.Queue(args.queue_size) for _ in range(args.num_workers)]
    producer_queue = mp.Queue(args.queue_size)

    consumer = mp.Process(target=consumer_fn, args=(args, pipeline_config, batch_queue, result_queues))
    consumer.start()

    producers = {}
    i, k_res = 0, 0
    while i < len(taskvars):
        taskvar = taskvars[i]
        if len(producers) < args.num_workers:
            print('start', i, taskvar)
            producer = mp.Process(
                target=producer_fn,
                args=(i, k_res, args, pipeline_config, taskvar, pred_file, batch_queue, result_queues[k_res], producer_queue),
                name=taskvar
            )
            producer.start()
            producers[i] = producer
            i += 1
            k_res += 1
        else:
            proc_id, k_res = producer_queue.get()
            producers[proc_id].join()
            del producers[proc_id]

    for p in producers.values():
        p.join()

    batch_queue.put(None)
    consumer.join()


if __name__ == '__main__':
    main()
