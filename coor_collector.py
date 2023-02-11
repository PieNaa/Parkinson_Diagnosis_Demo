import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st
import math

class get_data :
    def __init__(self, cap):
        self.cap = cap
        self.output_frames = []
        self.keep_coor_L = [] #To keep 26 values(from 78 frames) coor from left hand
        self.keep_coor_R = [] #To keep 26 values(from 78 frames) coor from right hand

    def get_coor(self):
        # print(self.folder_name, self.file_name)
        cap = self.cap
        mpHands = mp.solutions.hands
        hands = mpHands.Hands()
        mpDraw = mp.solutions.drawing_utils
        frame_no = 0

        keep_3frames_L_hand = [] #To keep tempt coor from each frame in 3 frames from left hand
        keep_3frames_R_hand = [] #To keep tempt coor from each frame in 3 frames from right hand

        while cap.isOpened():
            success, image = cap.read()
            width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
            if np.sum(image) == None :
                break
            # print("for frame : " + str(frame_no) + "   timestamp is: ", str(cap.get(cv2.CAP_PROP_POS_MSEC)))

            if frame_no>0 and frame_no%3==0 and len(self.keep_coor_L)<26:
                tmp_keep_coor_L = []
                tmp_keep_coor_R = []
                for i in [0,4,8,12,16,20] :
                    L_sumx=0
                    L_sumy=0
                    R_sumx=0
                    R_sumy=0
                    len_keep = len(keep_3frames_L_hand)
                    for j in range(len_keep) :
                        L_sumx += keep_3frames_L_hand[j][i][1]
                        L_sumy += keep_3frames_L_hand[j][i][2]
                        R_sumx += keep_3frames_R_hand[j][i][1]
                        R_sumy += keep_3frames_R_hand[j][i][2]
                    if len_keep>0 :
                        tmp_keep_coor_L.append([i,round((L_sumx/len_keep)/width, 3),round((L_sumy/len_keep)/height, 3)])
                        tmp_keep_coor_R.append([i,round((R_sumx/len_keep)/width, 3),round((R_sumy/len_keep)/height, 3)])
                    else :
                        tmp_keep_coor_L.append([i,-1,-1])
                        tmp_keep_coor_R.append([i,-1,-1])
                
                self.keep_coor_L.append(tmp_keep_coor_L)
                self.keep_coor_R.append(tmp_keep_coor_R)
                # print(self.keep_coor_L)
                # print(self.keep_coor_R)

                keep_3frames_L_hand = []
                keep_3frames_R_hand = []

            frame_no += 1

            imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(imageRGB)
            seconds = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)/int(cap.get(cv2.CAP_PROP_FPS)))

            L_lmList = []
            R_lmList = []
            handLms_tolist_1 = []
            handLms_tolist_2 = []
            side = 1
            
            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks: # working with each hand
                    for id, lm in enumerate(handLms.landmark):
                        h, w, c = image.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        if side == 1 :
                            handLms_tolist_1.append([cx,cy])
                        elif side == 2 :
                            handLms_tolist_2.append([cx,cy])

                    mpDraw.draw_landmarks(image, handLms, mpHands.HAND_CONNECTIONS, mpDraw.DrawingSpec(color=(0,0,255), thickness=5, circle_radius=5), mpDraw.DrawingSpec(color=(0,255,0), thickness=4, circle_radius=2))
                    if side == 1 :
                        side = 2
                if len(handLms_tolist_1)>0 and len(handLms_tolist_2)>0 :
                    if handLms_tolist_1[0][0] > handLms_tolist_2[0][0] :
                        for i in range(21) :
                            L_lmList.append([i,handLms_tolist_1[i][0],handLms_tolist_1[i][1]])
                            R_lmList.append([i,handLms_tolist_2[i][0],handLms_tolist_2[i][1]])
                    else :
                        for i in range(21) :
                            L_lmList.append([i,handLms_tolist_2[i][0],handLms_tolist_2[i][1]])
                            R_lmList.append([i,handLms_tolist_1[i][0],handLms_tolist_1[i][1]])

                    keep_3frames_L_hand.append(L_lmList)
                    keep_3frames_R_hand.append(R_lmList)

                # print(keep_3frames_L_hand)
                # print(keep_3frames_R_hand)

            self.output_frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            # cv2.imshow("Output", image)
            cv2.waitKey(1)

        # data_set.append(case)
        cap.release()
        return [self.keep_coor_L, self.keep_coor_R, self.output_frames]

    def dist_direction_find(hand_coor):
        distance = []
        direction = []
        df = pd.DataFrame(hand_coor)
        for i in range(6) :
            df_pos = pd.DataFrame(df[i].tolist())
            tmp_dist = []
            tmp_dirt = []
            prev_x = df_pos.iat[0,1]
            prev_y = df_pos.iat[0,2]
            for j in range(26) :
                if j == 0 :
                    continue
                elif prev_x == -1 or prev_y == -1 :
                    tmp_dist.append(0)
                    tmp_dirt.append(0)
                    continue
                else :
                    pres_x = df_pos.iat[j,1]
                    pres_y = df_pos.iat[j,2]
                    tmp_dist.append(round(math.sqrt((pres_x-prev_x)**2+(pres_y-prev_y)**2),6))
                    if pres_x != prev_x :
                        tmp_dirt.append(round(math.atan((pres_y-prev_y)/(pres_x-prev_x)),6))
                    elif pres_y > prev_y:
                        tmp_dirt.append(1.570800)
                    else :
                        tmp_dirt.append(4.712390)
                    prev_x = df_pos.iat[j,1]
                    prev_y = df_pos.iat[j,2]
            distance.append(tmp_dist)
            direction.append(tmp_dirt)
        return distance, direction
        


