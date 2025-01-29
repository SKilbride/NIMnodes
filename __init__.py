import base64
from io import BytesIO

import numpy as np
import requests
import torch
from PIL import Image

from .ngc import get_ngc_key
from .nim import ModelType

invoke_url = "http://localhost:8003/v1/infer"

class NIMSDXLNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": (["768", "832", "896", "960", "1024", "1088", "1152", "1216", "1280", "1344"], {  # noqa: E501
                    "default": "1024",
                    "tooltip": "Width of the image to generate, in pixels."
                }),
                "height": (["768", "832", "896", "960", "1024", "1088", "1152", "1216", "1280", "1344"], {
                    "default": "1024",
                    "tooltip": "Height of the image to generate, in pixels."
                }),
                "positive": ("STRING", {
                    "multiline": True,
                    "default": "beautiful scenery nature glass bottle landscape, purple galaxy bottle",
                    "tooltip": "The attributes you want to include in the image."
                }),
                "negative": ("STRING", {
                    "multiline": True,
                    "default": "text, watermark",
                    "tooltip": "The attributes you want to exclude from the image."
                }),
                "cfg_scale": ("FLOAT", {
                    "default": 5.0,
                    "min": 1.0,
                    "max": 20.0,
                    "step": 0.5,
                    "display": "slider",
                    "tooltip": "How strictly the diffusion process adheres to the prompt text (higher values keep your image closer to your prompt)"
                }),
                "sampler": (["DDIM", "K_EULER_ANCESTRAL", "K_DPM_2_ANCESTRAL"], {
                    "default": "K_DPM_2_ANCESTRAL",
                    "tooltip": "The sampler to use for generation. Varying diffusion samplers will vary outputs significantly."
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 4294967295,
                    "display": "number",
                    "tooltip": "The seed which governs generation. Use 0 for a random seed"
                }),
                "steps": ("INT", {
                    "default": 25,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "display": "slider",
                    "tooltip": "Number of diffusion steps to run"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "image generation"

    def generate(self, width, height, positive, negative, cfg_scale, sampler, seed, steps):
        payload = {
            "width": width,
            "height": height,
            "text_prompts": [
                {
                    "text": positive,
                    "weight": 1
                },
                {
                    "text": negative,
                    "weight": -1
                }
            ],
            "cfg_scale": cfg_scale,
            "sampler": sampler,
            "seed": seed,
            "steps": steps
        }

        try:
            response = requests.post(invoke_url, json=payload)
        except requests.exceptions.ConnectionError:
            raise ConnectionError("Unable to connect to NIM API. Please ensure that you have started the container via podman.")
            
        data = response.json()
        response.raise_for_status()
        img_base64 = data["artifacts"][0]["base64"]
        img_bytes = base64.b64decode(img_base64)

        print("Result: " + data["artifacts"][0]["finishReason"])

        image = Image.open(BytesIO(img_bytes))
        image = image.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]  

        return (image,)

class FetchNGCApiKey:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}} 
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("api_key",)
    FUNCTION = "fetch_key"
    CATEGORY = "NGC"
    
    def fetch_key(self):
        api_key = get_ngc_key()
        return (api_key,)

class LoadNimNode:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_type": ([e.value for e in ModelType], {
                    "default": ModelType.SDXL.value,
                    "tooltip": "The type of NIM model to load"
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "NGC API Key for authentication"
                }),
                "port": ("INT", {
                    "default": 8003,
                    "min": 1,
                    "max": 65535,
                    "tooltip": "Port number where NIM is running"
                })
            }
        }
    
    RETURN_TYPES = ("STRING",)  # Returns success/failure message
    FUNCTION = "load_nim"
    
    def load_nim(self, api_key, port):
        global invoke_url
        invoke_url = f"http://localhost:{port}/v1/infer"
        
        # Here you could add logic to validate the API key and connection
        # For now, we'll just return a success message
        return (f"NIM configured on port {port}",)

# Update the mappings
NODE_CLASS_MAPPINGS = {
    "NIMSDXLNode": NIMSDXLNode,
    "FetchNGCApiKey": FetchNGCApiKey,
    "LoadNimNode": LoadNimNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NIMSDXLNode": "NIM SDXL",
    "FetchNGCApiKey": "NGC API Key",
    "LoadNimNode": "Load NIM"
}
