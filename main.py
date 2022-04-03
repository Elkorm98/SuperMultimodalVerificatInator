import tkinter as tk
import time
import argparse
# import imutils
import time
import cv2
import os
from modules import *
from PIL import Image, ImageTk
from pipeline2 import feature_extraction2, training_testing_report2, transformations_concatenation1
clicked1 = False
clicked2 = False
clicked3 = False

is_model = False


def click_1():
    global clicked1
    clicked1 = True


def click_2():
    global clicked2
    clicked2 = True


def click_3():
    global clicked3
    clicked3 = True


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

root = tk.Tk()
take_photo = tk.Button(root, text="Take a photo", command=click_1)
create_model = tk.Button(root, text="Create model", command=click_2)
predict_user = tk.Button(root, text="Predict user", command=click_3)
user_name = tk.Entry(root)
predicted_user = tk.Label(root, text="")
take_photo.grid(row=1, column=1)
create_model.grid(row=2, column=1)
predict_user.grid(row=3, column=1)
user_name.grid(row=1, column=2)
predicted_user.grid(row=3, column=2)
l = tk.Label(root)
l.grid(row=4, column=2)
# To capture video from webcam.
cap = cv2.VideoCapture(0)
# To use a video file as input
# cap = cv2.VideoCapture('filename.mp4')

while True:
    _,img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blue, green, red = cv2.split(img)
    img2 = cv2.merge((red, green, blue))

    img_tk = ImageTk.PhotoImage(Image.fromarray(img2))
    l['image'] = img_tk
    if clicked1:
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) != 0:
            for (x, y, w, h) in faces:
                img_to_photo = img[y:y + w, x:x + h]
            un = user_name.get()
            take_a_photo(img_to_photo, un)
        else:
            predicted_user.config(text="Nie wykryto twarzy!")
        clicked1 = False
    if clicked2:
        (train_sig_X, train_sig_Y, test_sig_X, test_sig_Y,
         train_img_X, train_img_Y, test_img_X, test_img_Y,
         train_wrist_X, train_wrist_Y, test_wrist_X, test_wrist_Y) = feature_extraction2()
        (trainX, trainY, testX, testY) = transformations_concatenation1(train_sig_X, train_sig_Y, test_sig_X,
                                                                        test_sig_Y,
                                                                        train_img_X, train_img_Y, test_img_X,
                                                                        test_img_Y,
                                                                        train_wrist_X, train_wrist_Y, test_wrist_X,
                                                                        test_wrist_Y)
        class_model = training_testing_report2(trainX, trainY, testX, testY)
        is_model = True
        clicked2 = False
    if clicked3:
        if is_model:
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) != 0:
                for (x, y, w, h) in faces:
                    img_to_photo = img[y:y + w, x:x + h]
                    u = predikt(img_to_photo)
                    predicted_user.config(text=u)
            else:
                 predicted_user.config(text="Nie wykryto twarzy!")
        else:
            predicted_user.config(text="Nie utworzono modelu!")
        clicked3 = False

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        root.destroy()
        break
    root.update()

# Release the VideoCapture object
root.mainloop()
cap.release()