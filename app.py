import streamlit as st
import pycaret
from pycaret.classification import *
from io import StringIO
import tempfile
import coor_collector
import pandas as pd
import numpy as np
import cv2
import pickle
import os

# Header of page
st.header('Diagnosis application for Parkinson’s disease by hand tremor analysis')

# #Sub-header of page
# st.subheader('The Residual Values')

# # Body paragraph
# st.write(
#     """
#     The Residual Values (RV) model is a model to predict car values for given times. There are several car conditions having
#     effects to future prices. In the model, transformed categorical features via One-Hot Encoding and scaled numeric
#     features via normalization and polynomial transformer. The core model is used CatBoost regression.
#     """
# )

# st.subheader('To use model prediction, please following below steps:')

# st.write(
#     """
#     1. From the left side of this page, there is an area to input several car conditions. \n
#     2. To input car conditions that needed to be predicted. \n
#     3. See the results below.
#     """
# )

frame_label = ['Wrist_F1', 'Wrist_F2', 'Wrist_F3', 'Wrist_F4', 'Wrist_F5', 'Wrist_F6', 'Wrist_F7', 'Wrist_F8', 'Wrist_F9', 'Wrist_F10', 'Wrist_F11', 'Wrist_F12', 'Wrist_F13', 'Wrist_F14', 'Wrist_F15', 'Wrist_F16', 'Wrist_F17', 'Wrist_F18', 'Wrist_F19', 'Wrist_F20', 'Wrist_F21', 'Wrist_F22', 'Wrist_F23', 'Wrist_F24', 'Wrist_F25','Thumb_F1', 'Thumb_F2', 'Thumb_F3', 'Thumb_F4', 'Thumb_F5', 'Thumb_F6', 'Thumb_F7', 'Thumb_F8', 'Thumb_F9', 'Thumb_F10', 'Thumb_F11', 'Thumb_F12', 'Thumb_F13', 'Thumb_F14', 'Thumb_F15', 'Thumb_F16', 'Thumb_F17', 'Thumb_F18', 'Thumb_F19', 'Thumb_F20', 'Thumb_F21', 'Thumb_F22', 'Thumb_F23', 'Thumb_F24', 'Thumb_F25','Index_F1', 'Index_F2', 'Index_F3', 'Index_F4', 'Index_F5', 'Index_F6', 'Index_F7', 'Index_F8', 'Index_F9', 'Index_F10', 'Index_F11', 'Index_F12', 'Index_F13', 'Index_F14', 'Index_F15', 'Index_F16', 'Index_F17', 'Index_F18', 'Index_F19', 'Index_F20', 'Index_F21', 'Index_F22', 'Index_F23', 'Index_F24', 'Index_F25','Middle_F1', 'Middle_F2', 'Middle_F3', 'Middle_F4', 'Middle_F5', 'Middle_F6', 'Middle_F7', 'Middle_F8', 'Middle_F9', 'Middle_F10', 'Middle_F11', 'Middle_F12', 'Middle_F13', 'Middle_F14', 'Middle_F15', 'Middle_F16', 'Middle_F17', 'Middle_F18', 'Middle_F19', 'Middle_F20', 'Middle_F21', 'Middle_F22', 'Middle_F23', 'Middle_F24', 'Middle_F25','Ring_F1', 'Ring_F2', 'Ring_F3', 'Ring_F4', 'Ring_F5', 'Ring_F6', 'Ring_F7', 'Ring_F8', 'Ring_F9', 'Ring_F10', 'Ring_F11', 'Ring_F12', 'Ring_F13', 'Ring_F14', 'Ring_F15', 'Ring_F16', 'Ring_F17', 'Ring_F18', 'Ring_F19', 'Ring_F20', 'Ring_F21', 'Ring_F22', 'Ring_F23', 'Ring_F24', 'Ring_F25','Pinky_F1', 'Pinky_F2', 'Pinky_F3', 'Pinky_F4', 'Pinky_F5', 'Pinky_F6', 'Pinky_F7', 'Pinky_F8', 'Pinky_F9', 'Pinky_F10', 'Pinky_F11', 'Pinky_F12', 'Pinky_F13', 'Pinky_F14', 'Pinky_F15', 'Pinky_F16', 'Pinky_F17', 'Pinky_F18', 'Pinky_F19', 'Pinky_F20', 'Pinky_F21', 'Pinky_F22', 'Pinky_F23', 'Pinky_F24', 'Pinky_F25']
datasets_label = []
for i in frame_label :
    datasets_label.append('Dist_{0}'.format(i))
for i in frame_label :
    datasets_label.append('Dirt_{0}'.format(i))
for i in frame_label :
    datasets_label.append('Dist_mul_Dirt_{0}'.format(i))
for i in frame_label :
    datasets_label.append('Dist_div_Dirt_{0}'.format(i))

datasets = pd.DataFrame(columns=datasets_label)

def get_datasets(coor):
    case = []
    distance_change, direction_change = coor_collector.get_data.dist_direction_find(coor)
    for j in distance_change :  
        case.extend(j)
    for j in direction_change :  
        case.extend(j)
    for j in range(len(distance_change)) :  
        case.extend([a * b for a, b in zip(distance_change[j], direction_change[j])])
    for j in range(len(distance_change)) :
        case.extend([a / b if b!=0 else 0 for a, b in zip(distance_change[j], direction_change[j])])
    datasets.loc[len(datasets)] = case

video_file = st.file_uploader("Upload file")
if video_file is not None :
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    vf = cv2.VideoCapture(tfile.name)

    datasets = pd.DataFrame(columns=datasets_label)

    coor_L, coor_R, output_frames = coor_collector.get_data(vf).get_coor()

    # Show the video
    frame_idx = 0

    # Define the codec and create the VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter("video.mp4", fourcc, 30.0, (1920, 1080))

    # Write the frames to the video file
    for frame in output_frames:
        video.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    video.release()

    # Use Streamlit to display the video
    video_file = open("video.mp4","rb")
    video_bytes = video_file.read()
    st.video(video_bytes)
   
    get_datasets(coor_L)
    get_datasets(coor_R)
    
    # st.write(datasets)

    # st.write(pycaret.__version__)
    
    model = load_model('parkinson_lightgbm_model')
    prediction = predict_model(model, data = datasets.copy())

    # Make predictions
    # predictions = model.predict(data)
    
    #prediction_dict = {0: "Parkinson",1: "Essential Tremor",2: "Normal / No hand tremor"}
    res_L = prediction["Label"][0]
    res_R = prediction["Label"][1]

    if res_L == 0 or res_R == 0 :
        final_result = "Parkinson"

    elif res_L == 1 or res_R == 1 :
        final_result = "Essential Tremor"

    else :
        final_result = "Normal / No hand tremor"

    st.subheader("Diagnosis Result : {0}".format(final_result))