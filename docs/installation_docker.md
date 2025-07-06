# Install AlphaFold 3 with Docker (Ubuntu Desktop CUDA Driver 535 - Cuda Version 12.2)

## 1\. System Requirements Check

### Check Ubuntu Version

```css
lsb_release -a
```

*   Should show **Ubuntu 22.04** or higher.

### Check GPU and Driver

```plain
nvidia-smi
```

*   Should show:
    *   GPU (e.g., `RTX 4090`)
    *   Driver Version (e.g., `535.183.01`)
    *   CUDA Version (e.g., `12.2`)

If `nvidia-smi` is not found, install NVIDIA drivers:

```plain
ubuntu-drivers devices
sudo ubuntu-drivers autoinstall
sudo reboot
```

* * *

## 2\. Docker Installation

### Remove Old Versions

```plain
sudo apt-get remove docker docker-engine docker.io containerd runc
```

### Install Docker Prerequisites

```plain
sudo apt-get update
sudo apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release
```

### Add Docker Repository

**Add Docker’s GPG key:**

```plain
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
```

**Add repository:**

```plain
echo \
  "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```

**Install Docker**

```plain
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

**Add user to docker group (so you can run docker without sudo):**

```plain
sudo usermod -aG docker $USER
```

_(Log out and back in, or reboot, for the group changes to take effect.)_

* * *

## 3\. NVIDIA Container Toolkit Installation

### Add NVIDIA repository

```plain
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container.gpg] https://#' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

### Install

```plain
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

* * *

## 4\. AlphaFold 3 Setup

### Create Directory Structure

```javascript
mkdir -p ~/Repositories/alphafold
cd ~/Repositories/alphafold
mkdir af_input af_output af_weights public_databases
```

### Clone AlphaFold 3

```plain
git clone https://github.com/google-deepmind/alphafold3.git
```

### Decompress weights

**Move the weights to the af\_weights directory**

```plain
# Install zstd if not already installed
sudo apt-get install zstd

# Decompress the weights
cd af_weights
zstd -d weights.bin.zst
```

### Get the databases

```plain
cd ~/alphafold/alphafold3

# Make sure is executable
chmod +x fetch_databases.sh

# Define the directory
./fetch_databases.sh ~/thesis/public_databases


```

**To see the download progress modify the file** `./fetch_databases.sh` **:**

```plain
# wget --quiet --output-document=- \
wget --progress=bar:force --output-document=- \

# wget --quiet --output-document=- "${SOURCE}/${NAME}.zst" | \
wget --progress=bar:force --output-document=- "${SOURCE}/${NAME}.zst" | \
```

### Include an input example

```plain
cd ~/alphafold/af_input
nano fold_input.json
```

**Paste the following**

```plain
   {
     "name": "2PV7",
     "sequences": [
       {
         "protein": {
           "id": ["A", "B"],
           "sequence": "GMRESYANENQFGFKTINSDIHKIVIVGGYGKLGGLFARYLRASGYPISILDREDWAVAESILANADVVIVSVPINLTLETIERLKPYLTENMLLADLTSVKREPLAKMLEVHTGAVLGLHPMFGADIASMAKQVVVRCDGRFPERYEWLLEQIQIWGAKIYQTNATEHDHNMTYIQALRHFSTFANGLHLSKQPINLANLLALSSPIYRLELAMIGRLFAQDAELYADIIMDKSENLAVIETLKQTYDEALTFFENNDRQGFIDAFHKVRDWFGDYSEQFLKESRQLLQQANDLKQG"
         }
       }
     ],/
     "modelSeeds": [1],
     "dialect": "alphafold3",
     "version": 1
   }
```

### Modify Dockerfile

```bash
cd alphafold3
nano docker/Dockerfile
```

**Change the line 12 to:**

CUDA Drivers > 535 on Ubuntu Desktop are unstable; on a server, update to a newer version (545 or 550) that supports CUDA 12.6 to avoid this modification.

```plain
# FROM nvidia/cuda:12.6.0-base-ubuntu22.04
FROM nvidia/cuda:12.2.0-base-ubuntu22.04
```

**Change the line 52 to:**

```plain
# RUN pip3 install -r dev-requirements.txt
RUN pip3 install --retries 3 --default-timeout=200 -r dev-requirements.txt || \
    pip3 install --retries 3 --default-timeout=200 -r dev-requirements.txt
```

### Build Docker Image

**Clean existing installations (optional housekeeping):**

```powershell
# 1. Container Management
docker ps                      # List running containers
docker ps -a                   # List ALL containers (running and stopped)
docker stop $(docker ps -a -q) # Stop all running containers
docker rm $(docker ps -a -q)   # Remove all containers

# 2. Image Cleanup
docker images                  # List all images
docker rmi alphafold3         # Remove AlphaFold 3 image
docker rmi $(docker images -q) # (Optional) Remove ALL images

# 3. System Cleanup
docker system prune -f         # Remove unused data (stopped containers, unused networks)
docker system prune -a -f      # (Optional) More aggressive cleanup (includes unused images)

# 4. Volume Cleanup (Optional - be careful!)
docker volume ls              # List volumes
docker volume prune -f        # Remove unused volumes
```

**Verify cleanup:**

```plain
docker ps -a       # Should show no containers
docker images      # Should show no alphafold3 image
```

**Build new image**:

```plain
docker build --network=host -f docker/Dockerfile -t alphafold3 .
```

* * *

## 5\. Testing Installation

### Test Basic GPU Access

```plain
docker run --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

*       *   Error: “could not select device driver with capabilities: \[\[gpu\]\]”
    *   **Solution:** Install `nvidia-container-toolkit` and configure Docker.

### Test AlphaFold 3

```plain
docker run --gpus all alphafold3 python -c "import jax; print(jax.devices())"
```

*   Result 🟢:
    ```scheme
    [CudaDevice(id=0)]
    ```

### Test AlphaFold 3 Help

```plain
docker run --gpus all alphafold3 python run_alphafold.py --help
```

* * *

## 6\. Running AlphaFold 3

After setup, run predictions with:

```javascript
docker run -it \
  --volume ~/Repositories/alphafold/af_input:/root/af_input \
  --volume ~/Repositories/alphafold/af_output:/root/af_output \
  --volume ~/Repositories/alphafold/af_weights:/root/models \
  --volume ~/Repositories/alphafold/public_databases:/root/public_databases \
  --gpus all \
  alphafold3 \
  python run_alphafold.py \
    --json_path=/root/af_input/fold_input.json \
    --model_dir=/root/models \
    --output_dir=/root/af_output
```

**Remember:**

1. Place model weights in `af_weights` directory.
2. Place databases in `public_databases` directory.
3. Create input JSON files in `af_input` directory.
4. Results will appear in `af_output` directory.