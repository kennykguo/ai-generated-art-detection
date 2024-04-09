from PIL import Image
from torchvision import models, transforms
import torch
import streamlit as st
import torch.nn as nn

def test_image(img, model):
    output = model(img)
    return output

if __name__ == "__main__":
    st.title("AI Generated Art Detection")

    img = st.file_uploader(
        "Upload an image to predict whether it is AI generated or human drawn",
        type = "jpg")
    
    #Test an image
    #img_path = 'painting-Vincent-Van-Gogh-Sunflowers-canvas-Oil.jpg'
    #img = Image.open(img)
    transform = transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()])

    left, center, right = st.columns(3)
    with center:
        #Display image after it is uploaded
        if img is not None:
            img = Image.open(img).convert('RGB')
            st.image(img, caption='Test Image', width=256)
            img = transform(img).unsqueeze(0)

    #Create columns 
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button('Resnet152 - Trained FC Layer'):
            resnet152_model = models.resnet152(pretrained=False)
            for param in resnet152_model.parameters():
                param.requires_grad = False

            resnet152_model.fc = nn.Linear(2048, 2) #set number of output classes to 2
            resnet152_model.load_state_dict(torch.load('resnet152.pth'))
            resnet152_model.eval()
            
            output = test_image(img, resnet152_model)
            _, pred = torch.max(output, dim=1)
            
            if pred == 0:
                st.markdown("Model prediction: **Generated**")
            if pred == 1: 
                st.markdown("Model prediction: **Real**")
    
    with col2: 
        if st.button('Resnet152 - Trained Layer 4'):
            resnet152_layer4 = models.resnet152(pretrained=False)
            for param in resnet152_layer4.parameters():
                param.requires_grad = False

            resnet152_layer4.fc = nn.Linear(2048, 2) #set number of output classes to 2
            resnet152_layer4.load_state_dict(torch.load('resnet152_train_layer4.pth'))
            resnet152_layer4.eval()
            
            output = test_image(img, resnet152_layer4)
            _, pred = torch.max(output, dim=1)
            
            if pred == 0:
                st.markdown("Model prediction: **Generated**")
            if pred == 1: 
                st.markdown("Model prediction: **Real**")
    with col3: 
        if st.button('VGG'):
            vgg = models.vgg16(pretrained=False)
            for param in vgg.parameters():
                param.requires_grad = False

            vgg.classifier[-1] = nn.Linear(4096, 2)
            vgg.load_state_dict(torch.load('vgg.pth'))
            vgg.eval()
            
            output = test_image(img, vgg)
            _, pred = torch.max(output, dim=1)
            
            if pred == 0:
                st.markdown("Model prediction: **Generated**")
            if pred == 1: 
                st.markdown("Model prediction: **Real**")


    

    
