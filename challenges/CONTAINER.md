# Instruction for building containers

We use [Apptainer](https://apptainer.org/docs/user/latest/) (formerly Singularity) to run the submitted solutions, as Apptainer better aligns with the security policies in HPC systems.

However, since Docker is more commonly used in research communities, we will first demonstrate how to build a Docker image of the 3D-LOTUS models, and then show how to convert it into an Apptainer image.

Please ensure you are using a system where you have sudo privileges.

1. Install Docker and Apptainer

- [Official instruction for installing Docker](https://docs.docker.com/engine/install/)

- [Installing NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

- [Official instruction for installing Apptainer](https://apptainer.org/docs/admin/main/installation.html).
For example, you can install Apptainer on Ubuntu systems by running the following commands:
```bash
sudo add-apt-repository -y ppa:apptainer/ppa
sudo apt update
sudo apt install -y apptainer
```

2. Build Docker Image

We provide the Dockerfile in `robot-3dlotus/challenges/robot3dlotus_24.04` to build the Docker image.
- Go to the directory `robot-3dlotus/challenges/robot3dlotus_24.04`;
- Replace your huggingface token id in the `Dockerfile` which is used for downloading LLAMA3
- Run the following command:
```bash
docker build -t robot3dlotus:v1 .
```

After building the image, you should see `robot3dlotus:v1` listed when you run `docker images`.

3. Convert Docker Image into Apptainer Image
Docker images can be easily converted into Apptainer images as shown [here](https://apptainer.org/docs/user/latest/build_a_container.html#downloading-an-existing-container-from-docker-hub).

You can run the command as follows:
```bash
apptainer build robot3dlotus_v1.sif docker-daemon:robot3dlotus:v1
```

**Note on Docker and Apptainer Compatibility:**
For best practices when ensuring compatibility between Docker and Apptainer, please refer to the official guide [here](https://apptainer.org/docs/user/main/docker_and_oci.html#best-practices-for-docker-apptainer-compatibility).

In our experience, issues can arise when the host system and the container image use different base distributions, when the host runs a newer OS than the image (e.g., [issue #945](https://github.com/apptainer/apptainer/issues/945), [GLIBC problem](https://apptainer.org/docs/user/main/gpu.html)). 

To address this, we upgraded the image base from Ubuntu 20.04 to 24.04. However, since RLBench is designed to run on Ubuntu 20.04, we chose not to install RLBench in the submitted image to avoid compatibility issues.

4. Run the Apptainer Image

```bash
apptainer exec --env HF_HOME=/root/.cache/huggingface --nv robot3dlotus_v1.sif bash -c "cd /opt/codes/robot-3dlotus && conda run -n gembench --no-capture-output python challenges/server.py --port 13000 --model 3dlotusplus"
```

5. Test the Apptainer Image

You can test the Apptainer image under the environment where you have installed RLBench:
```bash
python challenges/client.py
```

We provide the constructed Apptainer Image in [Dropbox](https://www.dropbox.com/scl/fi/kfw97z8gmead6pokfvyp9/robot3dlotus_v1.sif?rlkey=juyn4a7bjcnyqnsq4xy6vrs01&st=1fzvupg9&dl=0). Have fun!