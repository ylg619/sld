# CV2 module

from os import MFD_ALLOW_SEALING
import cv2
from sld.handdetector import HandDetector
import av
import streamlit as st
import tempfile
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from keras.backend import manual_variable_initialization

# Import component
from streamlit_webrtc import (
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,


)

#dictionary of traduction letters
dict_letter = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I', 9:'K', 10:'L', 11:'M', 
               12:'N', 13:'O', 14:'P', 15:'Q', 16:'R', 17:'S', 18:'T', 19:'U', 20:'V', 21:'W', 22:'X', 23:'Y' }

#Set up STUN servers
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

@st.cache
def load_mo():
    model = load_model('models/model_resnet50_V2_82_3.h5')
    return model 
# Your class where you put the intelligence
class SignPredictor(VideoProcessorBase):

    def __init__(self) -> None:
        # Hand detector

        self.hand_detector = HandDetector(detectionCon=0.8, maxHands=1)
        self.model = load_mo()
        
    def find_hands(self, image):

        #add the rectangle in your image around the hands 
        hands, image_hand = self.hand_detector.findHands(image)

        if hands:
            bbox1 = hands[0]["bbox"] # Bounding box info x,y,w,h
            x, y, w, h = bbox1
            hand_img = image_hand[y-100:y+h+100, x-100:x + w + 100] # image of the hand
            
            img_hand_resize = cv2.resize(hand_img,(256,256))     # resize image to match model's expected sizing
            img_hand_resize = img_hand_resize.reshape(1,256,256,3)
            img_hand_resize = tf.math.divide(img_hand_resize,255)
            pred = self.model.predict(img_hand_resize)[0]
            for index, value in enumerate(pred):
                if value >= 0.80:
                    # print(model.predict(img_hand_resize, batch_size=1))
                    # print(img_hand_resize)
                    print(dict_letter[index])
                    # print(hands[0]["type"])
                    # hands[0]["type"] = dict_letter[index]
                    # print(hands[0]["type"])
                    break
        

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
    # manual_variable_initialization(True)
    stframe = st.empty()
    model = load_model('models/model_resnet50_V2_82_3.h5')
    hand_detector2 = HandDetector(detectionCon=0.7, maxHands=1)
    
    
    
    while vf.isOpened():
        ret, frame = vf.read()
        # if frame is read correctly ret is True
        
        hands, image_hand = hand_detector2.findHands(frame)

        if hands:
            bbox1 = hands[0]["bbox"] # Bounding box info x,y,w,h
            x, y, w, h = bbox1
            hand_img = image_hand[y-100:y+h+100, x-100:x + w + 100] # image of the hand
            img_hand_resize = cv2.resize(hand_img,(256,256))     # resize image to match model's expected sizing
            img_hand_resize = img_hand_resize.reshape(1,256,256,3)
            img_hand_resize = tf.math.divide(img_hand_resize,255)
            for index, value in enumerate(model.predict(img_hand_resize)[0]):
                if value >= 0.80:
                    # print(model.predict(img_hand_resize, batch_size=1))
                    # print(img_hand_resize)
                    print(dict_letter[index])
                    # print(hands[0]["type"])
                    # hands[0]["type"] = dict_letter[index]
                    # print(hands[0]["type"])
                    break
        
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(rgb)

