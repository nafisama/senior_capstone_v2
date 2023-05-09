# Importing the necessary Libraries
import cv2
from flask_cors import cross_origin
from flask import Flask, render_template, request
from main import text_to_speech
from flask import Flask, render_template, Response, request
import datetime, time
import os, sys
import numpy as np
from threading import Thread
import mediapipe as mp
import tensorflow as tf
import time 

# Set up the hand detection pipeline
mp_hands = mp.solutions.hands.Hands()

# Open the camera
#cap = cv2.VideoCapture(0)
camera = cv2.VideoCapture(0)
# Initialize the hand variables
hand = None
labels_dict = {'A':0,'B':1,'C':2,'L':3,'NO':4,'W':5,'YES':6}

model = tf.keras.models.load_model('./keras3_model.h5')

global capture, switch, out, counter,text, gender 
capture=0
switch=1
counter=0
text="ALEXA. WHO AM I"
gender='Female'

camera = cv2.VideoCapture(0)
#@staticmethod
def draw_landmarks(image, results):
        mp_holistic = mp.solutions.holistic  # Holistic model
        mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

        # Draw left hand connections
        mp_drawing.draw_landmarks(
            image,
            landmark_list=results.left_hand_landmarks,
            connections=mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(
                color=(232, 254, 255), thickness=1, circle_radius=1
            ),
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=(255, 249, 161), thickness=2, circle_radius=2
            ),
        )
    # Draw right hand connections
        mp_drawing.draw_landmarks(
            image,
            landmark_list=results.right_hand_landmarks,
            connections=mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(
                color=(232, 254, 255), thickness=1, circle_radius=2
            ),
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=(255, 249, 161), thickness=2, circle_radius=2
            ),
        )



#make shots directory to save pics
try:
    os.mkdir('./shots')
except OSError as error:
    pass





def gen_frames():  # generate frame by frame from camera
    global out, capture,rec_frame,counter
    while True:
        success, frame = camera.read() 
        if success:   
            if(capture):
                capture=0
                counter+=1
                now = datetime.datetime.now()
                #p = os.path.sep.join(['shots', "shot_{}.png".format(str(now).replace(":",''))])
                p2= os.path.sep.join(['shots', "shot_{}.png".format(str(counter).replace(":",''))])
                #cv2.imwrite(p, frame)
                #cv2.imwrite(p2,frame)
                frame = cv2.flip(frame, 1)
                
                # Process the frame
                results = mp_hands.process(frame)
        
                # Check if a hand was detected
                if results.multi_hand_landmarks:
                    # Get the first hand landmark
                    hand_landmarks = results.multi_hand_landmarks[0]
                    # Calculate the bounding box
                    x_min, y_min, x_max, y_max = float('inf'), float('inf'), float('-inf'), float('-inf')
                    for landmark in hand_landmarks.landmark:
                        x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                        x_min, y_min = min(x_min, x), min(y_min, y)
                        x_max, y_max = max(x_max, x), max(y_max, y)
                        cv2.circle(frame, (x, y), 4, (255, 5, 4), -1)
                        
                    # Draw the bounding box
                    cv2.rectangle(frame, (x_min-50, y_min-50), (x_max+50, y_max+50), (30, 9, 3), 2)
                    #draw_landmarks(frame, results)

                    # print(frame.shape)
                    # Crop the hand region
                    frame_capt = frame[max(y_min-25,1):y_max+25, max(x_min-25,1):x_max+25]
                    frame_capt = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_capt =cv2.resize(frame_capt,(224,224))
                    frame_capt=frame_capt[np.newaxis,:,:,:]
                    predictions=model.predict(frame_capt)

                    #pred_value=predictions[predictions.argmax()]
                    #print(frame_capt.shape)
                    #print(pred_value)
                    print(predictions.argmax())
                    key, value = list(labels_dict.items())[predictions.argmax()]

                    hand = frame[max(y_min-25,1):y_max+25, max(x_min-25,1):x_max+25]
                    #cv2.putText(frame, key, (x_min-45, y_max+45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    #time.sleep(.5)
                    if(key=='A'):
                        text="ALEXA. TELL ME A JOKE"
                    elif(key=='B'):
                        text="ALEXA. SING A SONG"
                    elif(key=='W'):
                        text=="ALEXA. WHAT'S THE WEATHER TODAY"
                    elif(key=='L'):
                        text="ALEX. TURN ON THE LIGHTS"
                    else:
                        text="ALEXA. I'm tired "
                    print(text)
                    text_to_speech(text, gender)

            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass
                
        else:
            pass
# import request
app = Flask(__name__)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/requests',methods=['POST','GET'])
def tasks():
    global switch,camera
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture
            capture=1
        elif  request.form.get('stop') == 'Stop/Start':
            
            if(switch==1):
                switch=0
                camera.release()
                cv2.destroyAllWindows()
            else:
                camera = cv2.VideoCapture(0)
                switch=1
        
        return render_template('frontend.html')
    else:
        return render_template('frontend.html')


@app.route('/', methods=['POST', 'GET'])
@cross_origin()
def homepage():
    if request.method == 'POST':
        text = request.form['speech']
        gender = request.form['voices']
        text_to_speech(text, gender)
        return render_template('frontend.html')
    else:
        return render_template('frontend.html')


if __name__ == "__main__":
    app.run(port=8000, debug=True)
camera.release()
cv2.destroyAllWindows()