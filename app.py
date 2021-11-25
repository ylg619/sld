# CV2 module
import cv2
from cvzone.HandTrackingModule import HandDetector
import av
import streamlit as st
import tempfile

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
        self.hand_detector = HandDetector(detectionCon=0.8, maxHands=2)

    def find_hands(self, image):

        #add the rectangle in your image around the hands 
        hands, image_hand = self.hand_detector.findHands(image)
        #load your deep learning model
        #make a prediction
        # add the prediction on the image
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

