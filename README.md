# NVIDIA NIM SDXL 

### What is Stable Diffusion XL NIM?

The Stable Diffusion XL NIM is a container that allows you to run Stability.AI’s Stable Diffusion XL model - one of the most popular visual generative AI models in the world - in the most optimal manner. 

## Getting Started with Stable Diffusion XL NIM with ComfyUI

Before installing, ensure your system meets the following requirements:  
Operating System: Windows 11 (22H1 or later)  
GPU: AD100 or above  
GPU Driver: Version 565.xx or later  
Virtualization Settings: Enabled in SBIOS - [Instructions to enable virtualization if it is not enabled](https://support.microsoft.com/en-gb/windows/enable-virtualization-on-windows-c5578302-6e43-4b4b-a449-8ced115f58e1)


The node can automatically detect if you have already set up NVIDIA NIM, if not, it will navigate to the NIMsetup.exe download page where you can download and install NIMs.  
Alternatively you can download and run the the installer from [here](https://storage.googleapis.com/comfy-assets/NIMSetup.exe).

After NIM is installed, please perform the following steps to start NIMs in Comfy UI:

1. Install ComfyUI following [this](https://github.com/comfyanonymous/ComfyUI?tab=readme-ov-file#installing) and prepare running enviorment for ComfyUI
2. Open ComfyUI folder, clone this repo and put it under `...\ComfyUI\custom_nodes\`
3. Go to `...\ComfyUI\custom_nodes\comfyui_nim\`and install dependency with `pip install -r requirements.txt`
4. (*when using staging NIMs*) Customize [API key](https://gitlab-master.nvidia.com/ruixiangw/comfyui_nim/-/blob/main/nim.py?ref_type=heads#L43) to your personal API key to access staging NIMs 
5. Run ComfyUI APP with `python main.py` under `...\ComfyUI\` 
6. Open ComfyUI in browser, and import workflow `...\ComfyUI\custom_nodes\comfyui_nim\example_workflows\sdxl_nim_workflow.json`
7. Run the workflow. *The first time you run this it will download and configure the container, this may take a while.*
![sdxl nims workflow](assets/sdxl_nim_workflow.png) 
8. When shutdown ComfyUI APP, the running NIMs will also be stopped  

### Install node in ComfyUI
The recommended way to install these nodes is to use the [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager) to easily install them to your ComfyUI instance.  
You can also manually install them by git cloning the repo to your ComfyUI/custom_nodes folder.


### Outdated Getting Started
1. Open the Terminal app  
![Search for Terminal in your Start Menu](assets/terminal-startmenu.png)  
2. Open the NVIDIA-Workbench profile by clicking on the arrow along the top  
![Load the NVIDIA-Workbench profile in Terminal](assets/terminal-workbench.png)
3. Setup NIM directories by running the following commands
    ```
    export LOCAL_NIM_CACHE=~/.cache/nim
    mkdir -p "$LOCAL_NIM_CACHE"
    chmod -R a+w "$LOCAL_NIM_CACHE"
    ```

4. You will need to start the container each time you want to use the node.  
To do this, navigate to the NVIDIA-Workbench profile in Terminal (see getting started section) and run the following command:
    ```
    podman run -it --rm \
    --device=nvidia.com/gpu=all \
    --shm-size=16GB \
    -e NGC_API_KEY=$NGC_API_KEY \
    -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
    -u $(id -u) \
    -p 8000:8000 \
    {todo: URL}
    ```
 The first time you run this it will download and configure the container, this may take a while.
 
5. After the container is running you can then simply add the NIM SDXL node to generate images, download the [example workflow here](example_workflows/sdxl_nim_workflow.json) or you can drag the image below into ComfyUI to view the embedded workflow:
![Example workflow](assets/workflow.png)  
