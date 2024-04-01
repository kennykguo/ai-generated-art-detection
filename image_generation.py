import requests
from PIL import Image
import os 
from io import BytesIO

def image_generation():
    
    #--request POST 'https://modelslab.com/api/v6/realtime/text2img' \
    keywords = ['A dog drinking from a mug of coffee digital art style']
    url = 'https://modelslab.com/api/v6/realtime/text2img'
    f = open("API_key.txt", "r")
    key = f.read()
    prompt = f"{keywords[0]}"
    body = {
        'key': key,
        "prompt": prompt,
        "negative_prompt": "low quality",
        "width": 256,
        "height": 256,
        "samples": 1
    }
    
    response = requests.post(url, data=body)
    return response
    
if __name__ == "__main__":
    response = image_generation()
    img_url = response.json()["output"][0]
    
    img_name = "test.png"
    output_dir = "Generated_Images"
    img_path = os.path.join(output_dir, img_name)
    
    img_response = requests.get(img_url)
    img = Image.open(BytesIO(img_response.content))
    img.save(img_path, "PNG")
    img.show()