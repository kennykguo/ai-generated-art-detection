import requests
from PIL import Image
import os 
from io import BytesIO
import pandas as pd
import random
import json
import time

def image_generation(prompt):
    #Model used: https://docs.modelslab.com/image-generation/realtime-stable-diffusion/text2img
    url = 'https://modelslab.com/api/v6/realtime/text2img'
    f = open("API_key.txt", "r")
    key = f.read()
    #Randomly choose a style
    styles = ["colored-pencil-art", "digital-art", "fantasy-art", "line-art", "ornate-and-intricate"]
    style = random.choice(styles)
    body = json.dumps({
        "key": key,
        "prompt": prompt,
        "negative_prompt": "low quality",
        "width": 256,
        "height": 256,
        "samples": 1,
        "enhance_prompt": True,
        "safety_checker": False,
        "enhance_style": style
    })
    headers = {
    'Content-Type': 'application/json'
    }
    response = requests.post(url, data=body, headers=headers)
    
    return response
    
if __name__ == "__main__":
    #Read blip_captions.csv
    df = pd.read_csv("ai-generated-art-detection\\blip_captions.csv")
    captions_list = df["Caption"].tolist()
    #print(len(captions_list)) #842
    
    for i in range(len(captions_list)):
        time.sleep(13) 
        
        if i % 10 == 0:
            print(f"Generating img no.{i}")

        try: 
            response = image_generation(captions_list[i])
            
            img_url = response.json()["output"][0]
            
            img_name = f"{captions_list[i]}.png"
            output_dir = "Generated_Images"
            img_path = os.path.join(output_dir, img_name)
            
            img_response = requests.get(img_url)
            img = Image.open(BytesIO(img_response.content))
            img.save(img_path, "PNG")
        except Exception as e:
            print(f"Error {e}")
        