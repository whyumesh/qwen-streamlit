import streamlit as st
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from PIL import Image
import tempfile
import av

# Load the model and processor
@st.cache_resource
def load_model():
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        torch_dtype=torch.float32  # Use float32 for CPU compatibility
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    return model, processor

model, processor = load_model()

# Streamlit UI
st.title("Image/Video Description Generator with Qwen2-VL")
st.write("Upload an image or video, and the model will describe it.")

# Upload file (image or video)
file_type = st.selectbox("Choose the file type", ("Image", "Video"))

if file_type == "Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load the image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Convert image to RGB if it has an alpha channel
        if image.mode == 'RGBA':
            image = image.convert('RGB')

        # Save the image temporarily
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            image.save(tmp_file.name, format='JPEG')  # Specify format to ensure it's saved as JPEG
            image_path = tmp_file.name
        
        # Prepare input for the model
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]
        
        # Process inputs...

        
        # Process inputs
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # Move inputs to CPU
        inputs = inputs.to("cpu")  # Ensure CPU usage

        # Inference: Generate description (running on CPU)
        with st.spinner('Generating description...'):
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        
        # Display the output
        st.subheader("Generated Description")
        st.write(output_text[0])

elif file_type == "Video":
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi"])

    if uploaded_file is not None:
        # Load and display the video
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name

        # Use PyAV to read and display video in Streamlit
        video_container = st.empty()
        with av.open(video_path) as video_stream:
            for frame in video_stream.decode(video=0):
                img = frame.to_image()
                video_container.image(img)

        # Prepare input for the model
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path},
                    {"type": "text", "text": "Describe this video."},
                ],
            }
        ]
        
        # Process inputs
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # Move inputs to CPU
        inputs = inputs.to("cpu")  # Ensure CPU usage
        
        # Inference: Generate description (running on CPU)
        with st.spinner('Generating description...'):
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        
        # Display the output
        st.subheader("Generated Description")
        st.write(output_text[0])
