import tkinter as tk
import time
import argparse
# import imutils
import time
import cv2
import os
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from modules import *
from PIL import Image, ImageTk
from pipeline2 import feature_extraction2, training_testing_report2, transformations_concatenation1, input_test
def find_folder(file_list, foldername):
    for file in file_list:
        if (file['title'] == foldername):
            folder_id = file['id']
            return folder_id


gauth = GoogleAuth()
gauth.LocalWebserverAuth()  # client_secrets.json need to be in the same directory as the script
drive = GoogleDrive(gauth)

# View all folders and file in your Google Drive
fileList = drive.ListFile({'q': "'root' in parents and trashed=false"}).GetList()
folder_id = find_folder(fileList, 'Data')
fileList = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()
folder_id = find_folder(fileList, 'Res')
fileList = fileList = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()



clicked1 = False
clicked2 = False
clicked3 = False
model_mode = 'both'
is_model = False


def click_1():
    global clicked1
    clicked1 = True


def click_2():
    global clicked2, model_mode
    model_mode = 'both'
    clicked2 = True


def click_3():
    global clicked3
    clicked3 = True


def click_4():
    global clicked2, model_mode
    model_mode = 'signature'
    clicked2 = True
def click_5():
    global clicked2, model_mode
    model_mode = 'wrist_gyroscope'
    clicked2 = True
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

root = tk.Tk()
# take_photo = tk.Button(root, text="Take a photo", command=click_1)
create_model = tk.Button(root, text="Create model", command=click_2)
create_model_sig = tk.Button(root, text="Create model signature", command=click_4)
create_model_wrist = tk.Button(root, text="Create model wrist_gyroscope", command=click_5)
predict_user = tk.Button(root, text="Predict user", command=click_3)

v = tk.StringVar(root, value='Kim jestes?')
# user_name = tk.Entry(root)
whoami = tk.Entry(root,textvariable=v)
predicted_user = tk.Label(root, text="")
# take_photo.grid(row=1, column=1)
create_model.grid(row=2, column=1)
create_model_sig.grid(row=3, column=1)
create_model_wrist.grid(row=4, column=1)
predict_user.grid(row=5, column=1)
# user_name.grid(row=1, column=2)
predicted_user.grid(row=5, column=2)

l = tk.Label(root)
l.grid(row=6, column=2)
whoami.grid(row=6,column=1)
# To capture video from webcam.
cap = cv2.VideoCapture(0)
# To use a video file as input
# cap = cv2.VideoCapture('filename.mp4')

while True:
    # _,img = cap.read()
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #
    # blue, green, red = cv2.split(img)
    # img2 = cv2.merge((red, green, blue))

    # img_tk = ImageTk.PhotoImage(Image.fromarray(img2))
    # l['image'] = img_tk
    # if clicked1:
    #     faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    #     if len(faces) != 0:
    #         for (x, y, w, h) in faces:
    #             img_to_photo = img[y:y + w, x:x + h]
    #         un = user_name.get()
    #         take_a_photo(img_to_photo, un)
    #     else:
    #         predicted_user.config(text="Nie wykryto twarzy!")
    #     clicked1 = False
    if clicked2:
        (train_sig_X, train_sig_Y, test_sig_X, test_sig_Y,
         train_wrist_X, train_wrist_Y, test_wrist_X, test_wrist_Y) = feature_extraction2()

        (trainX, trainY, testX, testY) = transformations_concatenation1(model_mode,train_sig_X, train_sig_Y, test_sig_X,
                                                                        test_sig_Y,train_wrist_X, train_wrist_Y, test_wrist_X,
                                                                        test_wrist_Y)
        class_model = training_testing_report2(trainX, trainY, testX, testY)
        # input_test(cv2.imread("users/Szymon/faces/img_130122200553.jpg"), "users/Szymon/signatures/predict1.txt", "users/Szymon/wrist_gyroscope/20220322_152308_Gyroscope.csv", "Szymon")
        is_model = True

        clicked2 = False
    if clicked3:
        if is_model:
            # faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            # if len(faces) != 0:
            #     for (x, y, w, h) in faces:
            #         img_to_photo = img[y:y + w, x:x + h]
            # u = predikt()
            try:
                for file in fileList:
                    if file['title'].find('Gyroscope') != -1:
                        fileG = drive.CreateFile({'id': file['id']})
                        fileG.GetContentFile('predict/wrist_gyroscope.csv')
                    if file['title'].find('Magnetometer') != -1 or file['title'].find('Gyroscope') != -1 or file['title'].find('Accelerometer') != -1:
                        fileT = drive.CreateFile({'id': file['id']})
                        fileT.Trash()
            except:
                pass

            u = input_test('predict/signature.txt', 'predict/wrist.csv', str(whoami.get()),model_mode)
            predicted_user.config(text=str(u))
                    # predicted_user.config(text=str(u))
            # else:
            #      predicted_user.config(text="Nie wykryto twarzy!")
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