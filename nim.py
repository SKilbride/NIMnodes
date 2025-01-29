import os
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
        ModelType.SDXL: "nvcr.io/nvstaging/nim/sdxl:1.0.0",
        ModelType.FLUX: "nvcr.io/nvstaging/nim/flux:1.0.0",
        ModelType.StudioVoice: "nvcr.io/nim/nvidia/maxine-studio-voice:latest"
    }

    def __init__(self):
        self._nim_server_proc_dict: dict[str, subprocess.Popen] = {}
        
    def setup_directories(self, model_name: str, version: str) -> bool:
        """Create necessary directories for NIM cache"""
        try:
            home = os.path.expanduser("~")
            cache_path = Path(f"{home}/nimcache/{model_name}/{version}/.cache")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Set permissions
            chmod_command = f"chmod 777 -R {cache_path}"
            result = subprocess.run(chmod_command, shell=True, check=True)
            
            print(f"Directory setup completed for {model_name}")
            return True
            
        except Exception as e:
            print(f"Failed to setup directories: {str(e)}")
            return False
    
    def pull_nim_image(self, nim_id: str, registry_path: str) -> bool:
        """Pull NIM image from the registry"""
        try:
            command = f"wsl -d NVIDIA-Workbench -- podman pull {registry_path}"
            result = subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                shell=True
            )
            
            if result.returncode == 0:
                print(f"{nim_id} image downloaded successfully")
                return True
            else:
                print(f"Failed to download image {nim_id}")
                return False
                
        except Exception as e:
            print(f"Error pulling image: {str(e)}")
            return False
    
    def start_nim_container(self, 
                          nim_id: str,
                          model_name: str,
                          version: str,
                          port: int,
                          api_key: str,
                          registry_path: str) -> bool:
        """Start a NIM container with the specified configuration"""
        try:
            home = os.path.expanduser("~")
            cache_path = f"{home}/nimonwsl2/{model_name}/{version}/.cache"
            
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
                return True
            else:
                print(f"Failed to start container for {nim_id}")
                return False
                
        except Exception as e:
            print(f"Error starting container: {str(e)}")
            return False
    
    def deploy_nim(self, 
                   model_name: ModelType,
                   version: str,
                   port: int,
                   api_key: str) -> bool:
        """Deploy a NIM model with all necessary setup"""
        nim_id = f"{model_name.value}-{version}"
        
        # Get registry path from MODEL_REGISTRY
        registry_path = self.MODEL_REGISTRY[model_name]
        
        # Setup directories
        if not self.setup_directories(model_name.value, version):
            return False
            
        # Pull image
        if not self.pull_nim_image(nim_id, registry_path):
            return False
            
        # Start container
        return self.start_nim_container(
            nim_id=nim_id,
            model_name=model_name.value,
            version=version,
            port=port,
            api_key=api_key,
            registry_path=registry_path
        )

# Example usage
if __name__ == "__main__":
    manager = NIMManager()
    
    # Example deployment
    success = manager.deploy_nim(
        model_name=ModelType.SDXL,
        version="1.5.1-rtx.rc3",
        port=18001,
        api_key="your-api-key-here"
    )
    
    if success:
        print("NIM deployment successful")
    else:
        print("NIM deployment failed")
