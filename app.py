from pathlib import Path
import PIL
import streamlit as st
import settings
import helper


st.set_page_config(
    page_title="Python ile Ürün Tanıma",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)

with st.sidebar:
    st.image('logo.png',width=100)
st.title("Python ile Ürün Tanıma Uygulaması")
st.sidebar.header("Model Ayarları")
model_type = st.sidebar.radio(
    "Model Seçimi", settings.DETECTION_MODEL_LIST)

model_path = ""
if model_type:
    model_path = Path(settings.MODEL_DIR, str(model_type))
else:
    st.error("Lütfen Model Seçimi Yapınız...")

try:
    model = helper.load_model(model_path)
except Exception as e:
    st.error(f"Model Yüklenirken Hata Oluştu... {model_path}")


confidence = float(st.sidebar.slider(
    "Model Doğrulama Aralığı", 25, 100, 40)) / 100

st.sidebar.header("Resim-Video-Kamera")
source_radio = st.sidebar.radio(
    "Ürün Tanıma Kaynağını Seçiniz", settings.SOURCES_LIST)

source_img = None

if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Resim Seçimi", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Örnek Resim",
                         use_container_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Yüklenen Resim",
                         use_container_width=True)
        except Exception as ex:
            st.error("Resim Yüklenirken Hata Oluştu...")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image_path, caption='Tanınan Resim',
                     use_container_width=True)
        else:
            if st.sidebar.button('Resimden Ürün Tanıma'):
                res = model.predict(uploaded_image,
                                    conf=confidence
                                    )
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Tanınan Resim',
                         use_container_width=True)
                try:
                    with st.expander("Ürün Tanıma Sonucu"):
                        for box in boxes:
                            st.write(box.data)
                except Exception as ex:
                    st.write("Yüklenen Resim Yok..!")

elif source_radio == settings.VIDEO:
    helper.play_stored_video(confidence, model)

elif source_radio == settings.WEBCAM:
    helper.play_webcam(confidence, model)

else:
    st.error("Lütfen Geçerli Bir Kaynak Seçiniz..!")
