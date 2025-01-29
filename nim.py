import os
import socket
import subprocess
from enum import Enum
from pathlib import Path


class ModelType(Enum):
    SDXL = "SDXL"
    FLUX = "FLUX"
    StudioVoice = "StudioVoice"

class NIMManager:
    '''
    This class is responsible for managing the NIM containers.

    Supports: 
    - SDXL
    - FLUX
    - StudioVoice (https://build.nvidia.com/nvidia/studiovoice/docker)
    '''
    # Registry paths for different model types
    MODEL_REGISTRY: dict[ModelType, str] = {
        ModelType.SDXL: "nvcr.io/0593836488614755/stable-diffusion-xl:1.0.0.ea_4090",
        ModelType.FLUX: "nvcr.io/nvstaging/nim/flux:1.0.0",
        ModelType.StudioVoice: "nvcr.io/nim/nvidia/maxine-studio-voice:latest"
    }

    def __init__(self):
        self._nim_server_proc_dict: dict[str, subprocess.Popen] = {}
        
    def setup_directories(self, model_name: str) -> None:
        """Create necessary directories for NIM cache"""
        home = os.path.expanduser("~")
        cache_path = Path(f"{home}/nimcache/{model_name}/latest/.cache")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Set permissions using WSL
        chmod_command = f"wsl -d NVIDIA-Workbench -- chmod 777 -R {cache_path}"
        result = subprocess.run(chmod_command, shell=True, capture_output=True)

        if result.returncode != 0:
            error_msg = (
                f"Failed to set permissions for {cache_path}:\n"
                f"stdout: {result.stdout.decode('ascii')}\n"
                f"stderr: {result.stderr.decode('ascii')}"
            )
            raise Exception(error_msg)
        

        print(f"Directory setup completed for {model_name}")
        return
    
    def pull_nim_image(self, nim_id: str, registry_path: str) -> None:
        """Pull NIM image from the registry"""
        command = f"wsl -d NVIDIA-Workbench -- podman pull {registry_path}"
        result = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=True
        )
        
        if result.returncode == 0:
            print(f"{nim_id} image downloaded successfully")
        else:
            raise Exception(f"Failed to download image {nim_id}")
          
    
    def start_nim_container(self, 
                          nim_id: str,
                          model_name: str,
                          port: int,
                          api_key: str,
                          registry_path: str) -> None:
        """Start a NIM container with the specified configuration"""
        home = os.path.expanduser("~")
        cache_path = f"{home}/nimonwsl2/{model_name}/latest/.cache"
        
        command = (
            f"wsl -d NVIDIA-Workbench -- podman run -it --rm "
            f"--device=nvidia.com/gpu=all "
            f"--name={model_name} "
            f"--shm-size=8GB "
            f"-e NGC_API_KEY={api_key} "
            f"-v {cache_path}:/opt/nim/.cache "
            f"-e NIM_RELAX_MEM_CONSTRAINTS=1 "
            f"-p {port}:8000 "
            f"{registry_path}"
        )
        
        run_process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=True
        )
        
        if run_process:
            self._nim_server_proc_dict[nim_id] = run_process
            print(f"Started NIM container for {nim_id} on port {port}")

    def is_port_in_use(self, port: int) -> bool:
        """Check if a port is already in use"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0

    def deploy_nim(self, 
                   model_name: ModelType,
                   port: int,
                   api_key: str) -> None:
        """Deploy a NIM model with all necessary setup"""
        nim_id = f"{model_name.value}-{port}"
        
        # Check if port is already in use
        if self.is_port_in_use(port):
            print(f"Port {port} is already in use. Assuming {nim_id} is already running.")
            return
            
        # Get registry path from MODEL_REGISTRY
        registry_path = self.MODEL_REGISTRY[model_name]
        
        # Setup directories
        self.setup_directories(model_name.value)
            
        # Pull image
        self.pull_nim_image(nim_id, registry_path)
            
        # Start container
        return self.start_nim_container(
            nim_id=nim_id,
            model_name=model_name.value,
            port=port,
            api_key=api_key,
            registry_path=registry_path
        )
