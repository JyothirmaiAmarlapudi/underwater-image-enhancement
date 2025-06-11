# Underwater Image Enhancement using Flask
Flask-based web app for underwater image enhancement using OpenCV
This web app allows users to enhanced underwater images using:
- Color Balance Correction
- CLAHE (Contrast Enhancement)
- Gamma Correction
- Denoising
- Image Fusion

## How to Use
1. Upload an image
2. Choose the enhancement method
3. View and download the enhanced image

## Technologies Used
- Python + Flask
- OpenCV
- HTML + CSS (inside `templates/index.html`)
##input images for image fusion

![input img fusion1](https://github.com/user-attachments/assets/d7437e3c-95e0-45f4-8787-4f0d7e0a6961)
![input img fusion2](https://github.com/user-attachments/assets/a2ce38c4-6ff2-4180-a248-e87c26dac536)


##output image for image fusion

![output img fusion](https://github.com/user-attachments/assets/a994fdc6-9247-4d7c-aa87-19cee8b64965)

## How to Run
```bash
pip install -r requirements.txt
python app.py
