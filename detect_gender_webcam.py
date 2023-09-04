from tensorflow import keras
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import cv2
import cvlib as cv

# Loads model created previously
# gender_detection.model
model = load_model(r'/Users/vasudevnair113/Downloads/faceRecog/gender_detection.model')

# Opens up the webcam
# Index = camera you want to use
webcam = cv2.VideoCapture(0)

classes = ['male', 'female']

# Loops through the frames coming from the webcam
while webcam.isOpened():
    # Looks at the current frame
    status, frame = webcam.read()
    if webcam.read() is None:
        continue
    
    # This function from cvlib module detects the face found in the frame
    face, confidence = cv.detect_face(frame)

    for idx, f in enumerate(face):
        # Points around the rectangle
        # Start values = bottom left
        # End values = top right
        startX = f[0]
        startY = f[1]
        endX = f[2]
        endY = f[3]

        # Last 2 params: color and thickness of frame
        cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)

        # Crops face out from frame and converts to numpy array
        face_crop = np.copy(frame[startY:endY,startX:endX])

        if (face_crop.shape[0]) < 10 or (face_crop.shape[1] < 10):
            continue

        # Data preprocessing of face before feeding it to model
        face_crop = cv2.resize(face_crop, (96,96)) # Dimensions the model is trained on
        face_crop = face_crop.astype('float') / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis = 0)

        # Passing face through our model
        # [0] is because predict returns a 2D array
        conf = model.predict(face_crop)[0]

        # Argmax returns index of the largest element in an array
        idx = np.argmax(conf)

        label = classes[idx]
        # Formatting into string (gender and confidence level)
        label = '{}: {:.2f}%'.format(label, conf[idx]*100)
        #label = 'Best Instructor of all Time: 10000000000%'

        Y = startY - 10 if startY - 10 > 10 else startY + 10

        # Writing onto the frame
        # 0.7 is font scale factor multiplied by font-specific base size
        cv2.putText(frame, label, (startX,Y), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,255,0), 2)

    # Displays output on screen
    cv2.imshow('Gender detection', frame)

    # If user presses 's' then the loop breaks
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

# Once loop is broken, this closes webcam
webcam.release()
cv2.destroyAllWindows()