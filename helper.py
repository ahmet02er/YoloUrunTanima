from ultralytics import YOLO
import streamlit as st
import cv2
import settings


def load_model(model_path):   
    model = YOLO(model_path)
    return model



def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=None, tracker=None):
    image = cv2.resize(image, (720, int(720*(9/16))))
    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, tracker=tracker)
    else:
        res = model.predict(image, conf=conf)

    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Ürün Tanıma',
                   channels="BGR",
                   use_container_width=True
                   )


def play_webcam(conf, model):
    source_webcam = settings.WEBCAM_PATH
    if st.sidebar.button('Kameradan Ürün Tanıma'):
        try:
            vid_cap = cv2.VideoCapture(source_webcam)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Kamera Açılırken Hata Oluştu... " + str(e))


def play_stored_video(conf, model):
    source_vid = st.sidebar.selectbox(
        "Video Seçimi...", settings.VIDEOS_DICT.keys())

    with open(settings.VIDEOS_DICT.get(source_vid), 'rb') as video_file:
        video_bytes = video_file.read()
    if video_bytes:
        st.video(video_bytes)

    if st.sidebar.button('Video İle Ürün Tanıma'):
        try:
            vid_cap = cv2.VideoCapture(
                str(settings.VIDEOS_DICT.get(source_vid)))
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Video Açılırken Hata Oluştu... " + str(e))
