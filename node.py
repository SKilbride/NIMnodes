import base64

import requests

invoke_url = "http://localhost:8003/v1/infer"

 

payload = {

    "text_prompts": [

        {

            "text": "realistic futuristic city-downtown with short buildings, sunset",

            "weight": 1

        },

        {

            "text":  "" ,

            "weight": -1

        }

    ],

    "cfg_scale": 5,

    "sampler": "K_DPM_2_ANCESTRAL",

    "seed": 0,

    "steps": 25

}

 

response = requests.post(invoke_url, json=payload)

response.raise_for_status()

data = response.json()

img_base64 = data['artifacts'][0]["base64"]

img_bytes = base64.b64decode(img_base64)

 

with open("test_image.jpg", "wb") as f:
    f.write(img_bytes)

 