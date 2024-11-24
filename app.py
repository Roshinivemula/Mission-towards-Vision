import streamlit as st
import google.generativeai as genai
from PIL import Image, ImageDraw
from pytesseract import pytesseract  # Import pytesseract for OCR
import pyttsx3
import torch
import os
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# Load environment variables
from dotenv import load_dotenv

# Explicitly specify the .env path
load_dotenv(dotenv_path="D:\innomatics\Finalproject\.env")  # Replace with the actual path to your .env file

# Configure the API key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("API key not found! Check the .env path and file content.")

# Print the API key for debugging purposes
#print("GEMINI_API_KEY:", api_key)

genai.configure(api_key=api_key)

# Specify the path to Tesseract-OCR
pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

# Streamlit App
st.set_page_config(page_title="Mission to Vision", layout="centered", page_icon="🤖")

# response function
def get_response(input_prompt, image_data):
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content([input_prompt, image_data[0]])
    return response.text

# function to convert image to bytes
def image_to_bytes(uploaded_file):
    try:
        bytes_data = uploaded_file.getvalue()

        image_parts = [
            {
                "mime_type": uploaded_file.type,
                "data": bytes_data
            }
        ]

        return image_parts
    
    except Exception as e:
        raise FileNotFoundError(f"Failed to process image. Please try again. Error: {e}")

# function to extract text from image
def extract_text_from_image(uploaded_file):
    try:
        img = Image.open(uploaded_file)

        # pytesseract to extract text
        extracted_text = pytesseract.image_to_string(img)

        if not extracted_text.strip():
            return "No text found in the image."
        
        return extracted_text

    except Exception as e:
        raise ValueError(f"Failed to extract text. Error: {e}")

# function for text to speech
def text_to_speech_pyttsx3(text):
    try:
        # Initialize TTS engine
        engine = pyttsx3.init()
        # Get available voices
        voices = engine.getProperty('voices')

        # Set the voice to Female (usually voice index 1 is female on most systems)
        engine.setProperty('voice', voices[1].id)  # Selects the second voice (usually female)

        # Set properties for rate and volume (optional)
        engine.setProperty('rate', 150)  # Speed of speech (higher = faster)
        engine.setProperty('volume', 1)  # Volume (0.0 to 1.0)
        
        # Speak the text
        engine.say(text)
        engine.runAndWait()
        
        engine.stop()  # Stop the engine after use
        
    except Exception as e:
        raise RuntimeError(f"Failed to convert text to speech. Error: {e}")


# Load object detection model (Faster R-CNN)
@st.cache_resource
def load_object_detection_model():
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

object_detection_model = load_object_detection_model()

def detect_objects(image, threshold=0.5, iou_threshold=0.5):
    try:
        # Transform image to tensor
        transform = transforms.Compose([transforms.ToTensor()])
        img_tensor = transform(image)
        
        # Get predictions
        predictions = object_detection_model([img_tensor])[0]
        
        # Perform Non-Maximum Suppression
        keep = torch.ops.torchvision.nms(predictions['boxes'], predictions['scores'], iou_threshold)
        
        # Filter results based on NMS and score threshold
        filtered_predictions = {
            'boxes': predictions['boxes'][keep],
            'labels': predictions['labels'][keep],
            'scores': predictions['scores'][keep]
        }
        
        return filtered_predictions
    except Exception as e:
        raise RuntimeError(f"Failed to detect objects. Error: {e}")

# COCO class labels (91 categories)
COCO_CLASSES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "N/A", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella", "N/A",
    "N/A", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "N/A", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "N/A", "dining table", "N/A", "N/A", "toilet",
    "N/A", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "N/A", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush","tree"
]

# Highlight detected objects in the image
def draw_boxes(image, predictions, threshold=0.5):
    draw = ImageDraw.Draw(image)
    labels = predictions['labels']
    boxes = predictions['boxes']
    scores = predictions['scores']

    for label, box, score in zip(labels, boxes, scores):
        if score > threshold:
            x1, y1, x2, y2 = box
            class_name = COCO_CLASSES[label.item()]  # Map label ID to class name
            draw.rectangle([x1, y1, x2, y2], outline="Red", width=3)
            draw.text((x1, y1), f"{class_name} ({score:.2f})", fill="black")
    return image

# Prompt Engineering
input_prompt = """
You are an AI assistant designed to assist visually impaired individuals
by analyzing images and providing descriptive outputs.
Your task is to:
- Analyze the uploaded image and describe its content in clear and simple language.
- Provide detailed information about objects, people, settings, or activities in the scene.
"""

st.title("AI Powered Solution for Assisting Visually Impaired Individuals")

st.markdown("""
**Features:**
- **Text-to-Speech Conversion**: Convert text to audio using description generated.
- **Real-Time Scene Analysis**: Describe scenes from the images uploaeded from User.
- **Object and Obstacle Detection**: Detect objects/obstacles from the image for the safe navigation.
""")

# File uploader
st.sidebar.header("**UPLOAD IMAGE**")
uploaded_file = st.sidebar.file_uploader("Choose your image:", type=['jpg', 'jpeg', 'png','WebP'])

if uploaded_file:
    st.sidebar.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

# Buttons
stop_audio_button = st.button("Stop Audio🚫")

bt1, bt2, bt3 = st.columns(3)
scene_analysis_button = bt1.button("Scene Description🏖️")
object_detection_button = bt2.button("Detect Objects🪄")
text_tts_button = bt3.button("Extract Text📖")

# Object Detection
if object_detection_button and uploaded_file:
    with st.spinner("Detecting objects....."):
        st.subheader("Detected Objects:")
        image = Image.open(uploaded_file)
        predictions = detect_objects(image)
        image_with_boxes = draw_boxes(image.copy(), predictions)
        st.image(image_with_boxes, caption="Objects Highlighted", use_container_width=True)

# Scene Analysis
if scene_analysis_button and uploaded_file:
    with st.spinner("I am Analyzing....."):
        st.subheader("Scene Description:")
        image_data = image_to_bytes(uploaded_file)
        response = get_response(input_prompt, image_data)
        st.write(response)
        
        # Convert response to audio
        text_to_speech_pyttsx3(response)

# Extract Text and TTS
if text_tts_button and uploaded_file:
    with st.spinner("Extracting text....."):
        text = extract_text_from_image(uploaded_file)

        st.subheader("Extracted Text:")
        st.write(text)
        
        # Convert extracted text to audio
        text_to_speech_pyttsx3(text)

# Stop audio if the button is pressed
if stop_audio_button:
    pyttsx3.init().stop()
    st.success("Audio playback stopped.")
