from flask import Flask, render_template, request
import cv2
import numpy as np
import base64

app = Flask(__name__)


# Convert image to Base64
def image_to_base64(image):
    _, buffer = cv2.imencode(".jpg", image)
    return f"data:image/jpeg;base64,{base64.b64encode(buffer.tobytes()).decode('utf-8')}"


# Color Balance Correction
def color_balance(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_avg, a_avg, b_avg = np.mean(l), np.mean(a), np.mean(b)

    l = np.clip(l + (128 - l_avg), 0, 255).astype(np.uint8)
    a = np.clip(a + (128 - a_avg), 0, 255).astype(np.uint8)
    b = np.clip(b + (128 - b_avg), 0, 255).astype(np.uint8)

    balanced_image = cv2.merge([l, a, b])
    return cv2.cvtColor(balanced_image, cv2.COLOR_LAB2BGR)


# CLAHE (Contrast Limited Adaptive Histogram Equalization)
def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=25.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced_image = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced_image, cv2.COLOR_LAB2BGR)


# Gamma Correction
def gamma_correction(image, gamma=1.5):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0)**inv_gamma * 255
                      for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)


# Denoising
def denoise_image(image):
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)


# Image Fusion
def image_fusion(image1, image2):
    image1_resized = cv2.resize(image1, (image2.shape[1], image2.shape[0]))
    return cv2.addWeighted(image1_resized, 0.5, image2, 0.5, 0)


@app.route("/", methods=["GET", "POST"])
def index():
    input_image_base64 = None
    input_image2_base64 = None
    result_image_base64 = None

    if request.method == "POST":
        uploaded_file = request.files.get("image")
        uploaded_file2 = request.files.get(
            "image2")  # Get the second image file
        enhancement_option = request.form.get("option")

        if uploaded_file:
            uploaded_image = cv2.imdecode(
                np.frombuffer(uploaded_file.read(), np.uint8),
                cv2.IMREAD_COLOR)

            # Apply selected enhancement
            if enhancement_option == "Color Balance":
                result_image = color_balance(uploaded_image)
            elif enhancement_option == "CLAHE":
                result_image = apply_clahe(uploaded_image)
            elif enhancement_option == "Gamma Correction":
                result_image = gamma_correction(uploaded_image)
            elif enhancement_option == "Denoising":
                result_image = denoise_image(uploaded_image)
            elif enhancement_option == "Fusion" and uploaded_file2:
                uploaded_image2 = cv2.imdecode(
                    np.frombuffer(uploaded_file2.read(), np.uint8),
                    cv2.IMREAD_COLOR)
                input_image2_base64 = image_to_base64(uploaded_image2)
                result_image = image_fusion(uploaded_image, uploaded_image2)
            else:
                result_image = uploaded_image  # Default to original if no enhancement is selected

            # Convert images to Base64 for HTML rendering
            input_image_base64 = image_to_base64(uploaded_image)
            result_image_base64 = image_to_base64(result_image)

    return render_template(
        "index.html",
        input_image=input_image_base64,
        input_image2=
        input_image2_base64,  # Include second image if fusion is applied
        result_image=result_image_base64)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
