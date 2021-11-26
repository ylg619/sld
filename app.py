# CV2 module

from os import MFD_ALLOW_SEALING
import cv2
from sld.handdetector import HandDetector
import av
import streamlit as st
import tempfile
import tensorflow as tf
from tensorflow.keras.models import load_model


# Import component
from streamlit_webrtc import (
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,


)

#Set up STUN servers
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Your class where you put the intelligence
class SignPredictor(VideoProcessorBase):

    def __init__(self) -> None:
        # Hand detector

        self.hand_detector = HandDetector(detectionCon=0.8, maxHands=1)
        

    def load_model(self):
        model = load_model('models/model_h5.h5')
        return model


    def find_hands(self, image):

        #add the rectangle in your image around the hands 
        hands, image_hand = self.hand_detector.findHands(image)

        if hands:
            bbox1 = hands[0]["bbox"] # Bounding box info x,y,w,h
            x, y, w, h = bbox1
            hand_img = image_hand[y-100:y+h-100, x-100:x + w + 100] # image of the hand
            model = self.load_model()
            #img_resize = hand_img.resize((256,256))
            print(hand_img)
            #print(model.predict(hand_img))
            #cv2.rectangle

        return hands, image_hand

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        hands, annotated_image = self.find_hands(image)
        return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")


webrtc_ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=SignPredictor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )


# Upload a file


file = st.file_uploader("Upload file")

if file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(file.read())

    vf = cv2.VideoCapture(tfile.name)

    stframe = st.empty()

    while vf.isOpened():
        ret, frame = vf.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        stframe.image(gray)

