# GemBench Challenge in CVPR 2025 GRAIL Workshop


## Training Dataset
The official training data for the GemBench challenge can be found [here](https://github.com/vlc-robot/robot-3dlotus?tab=readme-ov-file#dataset), consisting of 31 task variations in GemBench L1 with 100 episodes per task variation.

Participants are allowed to use additional data or pretrained models in their solutions. However, **any data related to the task variations in GemBench L2-L4 is strictly prohibited**, as it would compomise the evaluation of generalization performance on unseen tasks.


## Observation and Action Space

The observation input contains the robot's proprioceptive state and RGB-D images from 4 cameras with camera parameters. The resolution of the images is 256x256.

The required output action is a 7-dimensional vector representing the end-effector pose, which includes 3D position, 4D quaternion rotation, and 1D openness state.

Participants may use the predefined waypoints from previous work or define new waypoints.
The default motion planner in RLBench is used to execute the predicted action, though it may not always reach the target position precisely. 


## Evaluation

The **average success rate** is the primary evaluation metric.
Participants can use GemBench L1–L4 for self-evaluation. For the challenge, we introduce a **private testing split** consisting of new, unseen tasks designed following the principles of L2–L4. Participants will be ranked based on their average success rate on this private testing split.


## Submission Guideline

- Participated teams should submit a [singularity image](https://docs.sylabs.io/guides/3.0/user-guide/quick_start.html) containing the inference codes and models, and a brief document for each submission. 

The construction of the singularity image is described in [Starter Code](#starter-code). The submitted model should be able to run on a single NVIDIA A100 GPU, without Internet access.

The document should briefly describe any additional data or pretrained models used in the submitted solution. Report the performance on GemBench L1-L4, including success rate at each level, runtime and hardware requirement. Submissions that outperform the 3D-LOTUS++ baseline will be evaluated on the private testing split.

- Each team is limited to 5 total submissions during the challenge period. We will provide teams with opportunities to verify that their submitted code can run successfully in our machines. The highest-performing submission will be considered for final evaluation. 

- At the end of the challenge, participating teams should submit a detailed report (4–8 pages) describing their methods and results. Selected teams will be invited to present their work (oral or poster) at the workshop.


## Starter Code

The inference code must follow [the specified API format](https://github.com/vlc-robot/robot-3dlotus/blob/main/challenges/actioner.py) to ensure compatibility with our evaluation setup. We will run it using the following command:
```bash
# Run your submitted model as a server
singularity exec --bind $HOME:$HOME,$SCRATCH:$SCRATCH --nv ${submitted_sif_image} xvfb-run -a python challenges/server.py --port 13000

# Run RLBench with new tasks and query the server
singularity exec --bind $HOME:$HOME,$SCRATCH:$SCRATCH --nv ${gembench_sif_image} xvfb-run -a python challenges/client.py --port 13000
```

An example Singularity image for running 3D-LOTUS++ will be released soon, along with additional instructions for building the Singularity image.

Happy hacking!