# app.py
import streamlit as st
from transformers import pipeline
from PIL import Image

# Заголовок приложения
st.title("🖼️ Image Captioning")

# Инструкция
st.markdown("Загрузите изображение, получите его описание!")

# Загрузка изображения
uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])

# Инициализация моделей
@st.cache_resource
def load_caption_model():
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
caption_model = load_caption_model()

if uploaded_file:
    # Показ изображения
    image = Image.open(uploaded_file)
    st.image(image, caption="Загруженное изображение", use_container_width=True)

    # Генерация описания
    with st.spinner("Генерация описания..."):
        caption = caption_model(image)[0]['generated_text']
    st.success("Описание изображения:")
    st.write(caption)


else:
    st.info("Пожалуйста, загрузите изображение, чтобы начать.")
    


