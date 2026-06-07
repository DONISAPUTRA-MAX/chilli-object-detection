import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ai_edge_litert.interpreter import Interpreter
import cv2

# =====================================
# CONFIG PAGE
# =====================================
st.set_page_config(
    page_title="chilly quality detection",
    layout="wide"
)

st.title("CAYENNE PEPPER QUALITY DETECTION🌶️")
st.write("Chili quality detection based on physical condition")
st.info(
""" USAGE TIPS FOR MAXIMUM RESULT💡  
USE WHITE BACKGROUND, DO NOT STACK CHILI, AND DO NOT EXCEED 3 """
)
st.divider()

# =====================================
# CLASS LABEL
# SESUAI URUTAN TRAINING YOLO
# =====================================
class_names = [
    'defective',
    'half-ripe',
    'ripe'
]

# =====================================
# LOAD MODEL
# =====================================
@st.cache_resource
def load_model():
    interpreter = Interpreter(
        model_path="best_float32.tflite"
    )
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# =====================================
# PILIH INPUT
# =====================================
st.subheader("Select Image Source")

option = st.radio(
    "Choose Input",
    ["Upload Image", "Use Camera"]
)

image = None

# =====================================
# UPLOAD IMAGE
# =====================================
if option == "Upload Image":

    uploaded_file = st.file_uploader(
        "Upload chilli image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

# =====================================
# CAMERA INPUT
# =====================================
elif option == "Use Camera":

    camera_image = st.camera_input(
        "Take chili foto"
    )

    if camera_image is not None:
        image = Image.open(camera_image).convert("RGB")

# =====================================
# DETEKSI
# =====================================
if image is not None:

    st.subheader("Real Image")
    st.image(image, use_container_width=True)

    original_width = image.width
    original_height = image.height

    # Resize ke input model
    resized_img = image.resize((640, 640))

    img_array = np.array(
        resized_img,
        dtype=np.float32
    )

    img_array = img_array / 255.0
    input_data = np.expand_dims(
        img_array,
        axis=0
    )

    # =====================================
    # INFERENCE
    # =====================================
    interpreter.set_tensor(
        input_details[0]['index'],
        input_data
    )

    interpreter.invoke()

    output_data = interpreter.get_tensor(
        output_details[0]['index']
    )

    # =====================================
    # PROCESS OUTPUT YOLO
    # =====================================
    output = np.squeeze(output_data).T

    conf_thres = 0.30
    iou_thres = 0.25

    boxes = []
    confidences = []
    class_ids = []

    for row in output:

        class_scores = row[4:]
        max_score = np.max(class_scores)

        if max_score > conf_thres:

            class_id = np.argmax(
                class_scores
            )

            cx, cy, w, h = row[:4]

            # Konversi koordinat
            x = int(
                (cx - w / 2)
                * original_width
            )

            y = int(
                (cy - h / 2)
                * original_height
            )

            width = int(
                w * original_width
            )

            height = int(
                h * original_height
            )

            boxes.append(
                [x, y, width, height]
            )

            confidences.append(
                float(max_score)
            )

            class_ids.append(
                class_id
            )

    # =====================================
    # NMS
    # =====================================
    indices = cv2.dnn.NMSBoxes(
        boxes,
        confidences,
        conf_thres,
        iou_thres
    )

    draw = ImageDraw.Draw(image)

    # Font dinamis menyesuaikan ukuran gambar
    font_size = max(32, image.width // 20)

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()


    if len(indices) > 0:

        for i in indices.flatten():

            x, y, w, h = boxes[i]

            label = class_names[class_ids[i]]
            conf = confidences[i]

            # Bounding box objek
            draw.rectangle(
                [(x, y), (x + w, y + h)],
                outline="red",
                width=4
            )

            # Text label
            text = f"{label} {conf:.2f}"

            # Hitung ukuran text
            bbox_text = draw.textbbox((0, 0), text, font=font)
            text_width = bbox_text[2] - bbox_text[0]
            text_height = bbox_text[3] - bbox_text[1]

            padding = 6
            # Kalau label tidak muat di atas box, taruh di DALAM (bawah tepi atas)
            if y - text_height - padding * 2 < 0:
                label_y = y + padding  # taruh di dalam box
            else:
                label_y = y - text_height - padding * 2  # taruh di atas box
            
            # Background text
            draw.rectangle(
                [
                    (x, label_y),
                    (x + text_width + padding * 2, label_y + text_height + padding)
                ],
                fill="red"
            )

            # Teks
            draw.text(
                (x + padding, label_y + padding // 2),
                text,
                fill="white",
                font=font
            )
            
    st.divider()
    st.subheader("Prediction Image")
    st.image(
        image,
        use_container_width=True
    )

else:
    st.info(
        "Upload from gallery or take foto with camera"
    )