import requests
import base64
import json

 

invoke_url = "http://localhost:8003/v1/infer"

 

# payload = {
#     "text_prompts": [
#         {
#             "text": "realistic futuristic city-downtown with short buildings, sunset",
#             "weight": 1
#         },
#         {
#             "text":  "" ,
#             "weight": -1
#         }
#     ],
#     "cfg_scale": 5,
#     "sampler": "K_DPM_2_ANCESTRAL",
#     "seed": 0,
#     "steps": 25
# }

# # Size is fixed for now
# payload_two = {  "cfg_scale": 5,  "clip_guidance_preset": "NONE",  
#  "disable_safety_checker": 'false',  # Remain false for now. NIM_ALLOW_UNCHECKED_GENERATION=true
#  "height": 1024,  
#  "sampler": "K_DPM_2_ANCESTRAL",  
#  "samples": 1,  "seed": 0,  "steps": 25,  
#  "style_preset": "none",  
#  "text_prompts": [    {      
#      "text": "A photo of a Shiba Inu dog with a backpack riding a bike",      "weight": 1    
#      }  ],  
#      "use_refiner": 'false',  
#      "width": 1024
# } 

# response = requests.post(invoke_url, json=payload)

# response.raise_for_status()

# data = response.json()

# img_base64 = data['artifacts'][0]["base64"]

# img_bytes = base64.b64decode(img_base64)

class SDXLNIMNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "realistic futuristic city-downtown with short buildings, sunset"
                }),
                "cfg_scale": ("FLOAT", {
                    "default": 5.0,
                    "min": 1.0,
                    "max": 20.0,
                    "step": 0.5,
                    "display": "slider"
                }),
                "sampler": (["K_DPM_2_ANCESTRAL"],),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,  # Max 64-bit unsigned int
                    "display": "number"
                }),
                "steps": ("INT", {
                    "default": 25,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "display": "slider"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "image generation"

    def generate(self, text, cfg_scale, sampler, seed, steps):
        payload = {
            "text_prompts": [
                {
                    "text": text,
                    "weight": 1
                }
            ],
            "cfg_scale": cfg_scale,
            "sampler": sampler,
            "seed": seed,
            "steps": steps
        }

        # Your existing API call code here
        response = requests.post(invoke_url, json=payload)
        response.raise_for_status()
        data = response.json()
        img_base64 = data['artifacts'][0]["base64"]
        img_bytes = base64.b64decode(img_base64)
        
        # Convert to the expected format for ComfyUI nodes
        # Note: You'll need to convert img_bytes to the proper tensor format
        # This is a placeholder - you'll need to implement the actual image conversion
        return (img_bytes,)

# Update the mappings
NODE_CLASS_MAPPINGS = {
    "SDXLNIMNode": SDXLNIMNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDXLNIMNode": "SDXL NIM Node"
}