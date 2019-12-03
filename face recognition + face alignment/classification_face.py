from mtcnn.mtcnn import MTCNN
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

folder_path = './Images'
file_list = os.listdir(folder_path)

for file in file_list:

    img = cv2.imread(os.path.join(folder_path,file))
    # img = cv2.cvtColor(cv2.imread('./temp2.jpg'), cv2.COLOR_BGR2RGB)
    detector = MTCNN()
    result = detector.detect_faces(img)

    bounding_box = result[0]['box']
    keypoints = result[0]['keypoints']

    #cv2.rectangle(img, (bounding_box[0], bounding_box[1]), (bounding_box[0]+bounding_box[2], bounding_box[1]+bounding_box[3]), (0, 155, 255), 2)

    #cv2.circle(img,(keypoints['left_eye']), 2, (0, 155, 255), 2)
    #cv2.circle(img,(keypoints['right_eye']), 2, (0, 155, 255), 2)
    #cv2.circle(img,(keypoints['nose']), 2, (0, 155, 255), 2)
    #cv2.circle(img,(keypoints['mouth_left']), 2, (0, 155, 255), 2)
    #cv2.circle(img,(keypoints['mouth_right']), 2, (0, 155, 255), 2)

    #cv2.imwrite("change_people.jpeg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # 눈과 눈 사이의 선 그리기
    #cv2.line(img, keypoints['left_eye'], keypoints['right_eye'], (255, 0, 0), 2)

    # find and angle of line by using slop of the line.
    dY = keypoints['right_eye'][1] - keypoints['left_eye'][1]
    dX = keypoints['right_eye'][0] - keypoints['left_eye'][0]
    angle = np.degrees(np.arctan2(dY, dX))

    #cv2.circle(img,((dX, dY)),2,(0, 155, 255), 2)

    # to get the face at the center of the image,
    # set desired left eye location. Right eye location
    # will be found out by using left eye location.
    # this location is in percentage.
    desiredlefteye = (0.35, 0.35)
    # 회전후 자른 이미지(얼굴) 크기를 설정
    desiredfacewidth = 48
    desiredfaceheight = 48

    desiredrighteyex = 1.0 - desiredlefteye[0]

    dist = np.sqrt((dX ** 2) + (dY ** 2))
    desireddist = (desiredrighteyex - desiredlefteye[0])
    desireddist *= desiredfacewidth
    scale = desireddist / dist

    eyescenter = ((keypoints['left_eye'][0] + keypoints['right_eye'][0]) // 2, (keypoints['left_eye'][1] + keypoints['right_eye'][1]) // 2)

    M = cv2.getRotationMatrix2D(eyescenter, angle, scale)

    tX = desiredfacewidth * 0.5
    tY = desiredfaceheight * desiredlefteye[1]
    M[0, 2] += (tX - eyescenter[0])
    M[1, 2] += (tY - eyescenter[1])

    (w, h) = (desiredfacewidth, desiredfaceheight)

    output = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    gray_resize = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_CUBIC)
    gray_resize = np.resize(gray_resize, (1,48,48,1))
    gray_resize = gray_resize / 255

    model = load_model('test2.h5')
    label = model.predict(gray_resize)
    print(label)
    label = np.argmax(label)
    print(label)
    labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    cv2.rectangle(img, (bounding_box[0], bounding_box[1]), (bounding_box[0]+bounding_box[2], bounding_box[1]+bounding_box[3]), (0, 155, 255), 2)
    cv2.putText(img, labels[label],  (bounding_box[0], bounding_box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 155, 255), 2, cv2.LINE_AA)
    cv2.imwrite(os.path.join('./result_Images', labels[label] + '_' +file), img)
