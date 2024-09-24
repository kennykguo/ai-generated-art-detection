# AI Generated Art Detection

Github for TMI's AI Generated Art Detection Project.

Our pipeline consists of the following:
* Read images from Reddit /art dataset
* Generate captions of images using BLIP caption model
* Create AI-generated images with Stable Diffusion using the generated captions
* Finetune ResNet and VGG models to detect human-drawn vs. AI-generated art
