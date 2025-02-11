import os
import numpy as np
import yaml
from easydict import EasyDict

from genrobo3d.evaluation.robot_pipeline import RobotPipeline


class RandomActioner(object):
    def predict(self, taskvar, episode_id, step_id, instruction, obs_state_dict):
        '''Args:
            taskvar: str, 'task+variation'
            episode_id: int
            step_id: int, [0, 25]
            instruction: str
            obs_state_dict: observations from genrobo3d.rlbench.environments.RLBenchEnv 
        '''
        pos = np.random.rand(3).astype(np.float32)

        quat = np.random.rand(4).astype(np.float32)
        quat = quat / np.maximum(np.sqrt(np.sum(quat**2)), 1e-10)

        openness = np.array(np.random.rand(1) > 0.5, dtype=np.float32)
        print(pos, quat, openness)

        action = np.concatenate([pos, quat, openness], -1)

        return action


class ThreeDLotusPlusActioner(object):
    def __init__(self):
        with open('genrobo3d/configs/rlbench/robot_pipeline.yaml', 'r') as f:
            pipeline_config = yaml.safe_load(f)
        pipeline_config = EasyDict(pipeline_config)

        pipeline_config.llm_planner.use_groundtruth = False
        pipeline_config.object_grounding.use_groundtruth = False
        pipeline_config.motion_planner.run_action_step = 5
        pipeline_config.motion_planner.save_obs_outs = False

        mp_expr_dir = pipeline_config.motion_planner.expr_dir
        mp_ckpt_step = 140000
        pipeline_config.motion_planner.checkpoint = os.path.join(
            mp_expr_dir, 'ckpts', f'model_step_{mp_ckpt_step}.pt'
        )
        pipeline_config.motion_planner.config_file = os.path.join(
            mp_expr_dir, 'logs', 'training_config.yaml'
        )

        self.pipeline = RobotPipeline(pipeline_config)

        self.cache = None

    def predict(self, taskvar, episode_id, step_id, instruction, obs_state_dict):
        task_str, variation_id = taskvar.split('+')
        variation_id = int(variation_id)

        if step_id == 0:
            self.cache = None
        
        output = self.pipeline.predict(
            task_str=task_str, 
            variation=variation_id,
            episode_id=episode_id,
            step_id=step_id,
            instructions=[instruction],
            obs_state_dict=obs_state_dict,
            cache=self.cache
        )

        action = output['action']
        self.cache = output['cache']

        return action
