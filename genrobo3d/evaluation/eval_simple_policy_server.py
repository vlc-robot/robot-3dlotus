from typing import List

import os
import json
import jsonlines
import torch.multiprocessing as mp
import tap
from termcolor import colored

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
from genrobo3d.evaluation.eval_simple_policy import Actioner

CAMERA_NAMES = ("left_shoulder", "right_shoulder", "wrist", "front")

class ServerArguments(tap.Tap):
    expr_dir: str
    ckpt_step: int
    device: str = 'cuda'  # cpu, cuda

    image_size: List[int] = [256, 256]
    max_tries: int = 10
    max_steps: int = 25
    cam_ids: List[int] = [0, 1, 2, 3]

    microstep_data_dir: str = ''
    seed: int = 100  # seed for RLBench
    num_workers: int = 4
    queue_size: int = 20
    taskvar_file: str = 'assets/taskvars_train.json'
    num_demos: int = 20
    num_ensembles: int = 1

    save_obs_outs_dir: str = None

    best_disc_pos: str = 'max' # max, ens1

    record_video: bool = False
    video_dir: str = None
    not_include_robot_cameras: bool = False
    video_rotate_cam: bool = False
    video_resolution: int = 480

    real_robot: bool = False


def consumer_fn(args, batch_queue, result_queues):
    print('consumer start')
    # build model
    set_random_seed(args.seed)
    actioner = Actioner(args)

    while True:
        data = batch_queue.get()
        if data is None:
            print('Received None value -> Producers finished.')
            break
        
        # run one batch
        k_prod, batch = data
        out = actioner.predict(**batch)
        result_queues[k_prod].put(out)
    
def producer_fn(proc_id, k_res, args, taskvar, pred_file, batch_queue, result_queue, producer_queue):
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
        apply_mask=True,
        headless=True,
        image_size=args.image_size,
        cam_rand_factor=0,
        apply_cameras=[CAMERA_NAMES[cam_id] for cam_id in args.cam_ids]
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

        for step_id in range(args.max_steps):
            # fetch the current observation, and predict one action
            batch = {
                'task_str': task_str,
                'variation': variation,
                'step_id': step_id,
                'obs_state_dict': obs_state_dict,
                'episode_id': demo_id,
                'instructions': instructions,
            }
            batch_queue.put((k_res, batch))

            output = result_queue.get()
            action = output["action"]

            if action is None:
                break

            # update the observation based on the predicted action
            try:
                obs, reward, terminate, _ = move(action, verbose=False)
                obs_state_dict = env.get_observation(obs)  # type: ignore

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
            'checkpoint': args.checkpoint,
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
    args.exp_config = os.path.join(args.expr_dir, 'logs', 'training_config.yaml')
    args.checkpoint = os.path.join(args.expr_dir, 'ckpts', f'model_step_{args.ckpt_step}.pt')
    if not os.path.exists(args.checkpoint):
        print(args.checkpoint, 'not exists')
        return

    pred_dir = os.path.join(args.expr_dir, 'preds', f'seed{args.seed}')
    os.makedirs(pred_dir, exist_ok=True)
    pred_file = os.path.join(pred_dir, 'results.jsonl')
    existed_taskvars = set()
    if os.path.exists(pred_file):
        with jsonlines.open(pred_file, 'r') as f:
            for item in f:
                item_step = int(os.path.basename(item['checkpoint']).split('.')[0].split('_')[-1])
                if item_step == args.ckpt_step:
                    existed_taskvars.add(f"{item['task']}+{item['variation']}")

    taskvars = json.load(open(args.taskvar_file))
    taskvars = [taskvar for taskvar in taskvars if taskvar not in existed_taskvars]
    print('checkpoint', args.ckpt_step, '#taskvars', len(taskvars))

    batch_queue = mp.Queue(args.queue_size)
    result_queues = [mp.Queue(args.queue_size) for _ in range(args.num_workers)]
    producer_queue = mp.Queue(args.queue_size)

    consumer = mp.Process(target=consumer_fn, args=(args, batch_queue, result_queues))
    consumer.start()

    producers = {}        
    i, k_res = 0, 0
    while i < len(taskvars):
        taskvar = taskvars[i]
        if len(producers) < args.num_workers:
            print('start', i, taskvar)
            producer = mp.Process(
                target=producer_fn, 
                args=(i, k_res, args, taskvar, pred_file, batch_queue, result_queues[k_res], producer_queue),
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
            # producers[0].join()
            # producers = producers[1:]

    for p in producers.values():
        p.join()

    batch_queue.put(None)
    consumer.join()
    

if __name__ == '__main__':
    main()
