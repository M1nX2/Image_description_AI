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

if uploaded_file:
    # Показ изображения
    image = Image.open(uploaded_file)
    st.image(image, caption="Загруженное изображение", use_container_width=True)

    # Генерация описания
    with st.spinner("Генерация описания..."):
        caption = caption_model(image)[0]['generated_text']
    st.success("Описание изображения:")
    st.write(caption)

    # Простая чат-секция
    st.markdown("---")
    st.subheader("💬 Чат с ботом")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("Ваш вопрос:")

    if user_input:
        prompt = f"Изображение описано так: {caption}. Пользователь спрашивает: {user_input}"
        response = chat_model(prompt, max_length=150, do_sample=True, temperature=0.7)
        bot_reply = response[0]["generated_text"]
        st.session_state.chat_history.append(("Вы", user_input))
        st.session_state.chat_history.append(("Бот", bot_reply))

    for sender, msg in st.session_state.chat_history:
        st.markdown(f"{sender}: {msg}")
else:
    st.info("Пожалуйста, загрузите изображение, чтобы начать.")
    


