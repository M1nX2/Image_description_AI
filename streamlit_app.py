# app.py
import streamlit as st
from transformers import pipeline
from PIL import Image

# Заголовок приложения
st.title("🖼️ Image Captioning + Chat Bot")

# Инструкция
st.markdown("Загрузите изображение, получите его описание и задавайте вопросы о нём!")

# Загрузка изображения
uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])

# Инициализация моделей
@st.cache_resource
def load_caption_model():
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

@st.cache_resource
def load_chat_model():
    return pipeline("text-generation", model="microsoft/DialoGPT-medium")

caption_model = load_caption_model()
chat_model = load_chat_model()


