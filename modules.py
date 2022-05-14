import os
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from datetime import datetime
from sklearn.svm import SVC
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from skimage.exposure import rescale_intensity
import numpy as np
from signature_modules import *
import cv2

labels = []
name_number = 0
alfa=0.7
beta=0.3

def take_a_photo(img, username):
    global name_number
    if not os.path.isdir(f'./users/{username}/faces'):
        os.mkdir(f'./users/{username}/faces')
    name_date = datetime.now().strftime("%d%m%y%H%M%S")
    cv2.imwrite(f"./users/{username}/faces/img_{name_date}.jpg", img)
    name_number += 1

def merge_sets(img_matrix,signature_matrix):
    merged_array=[]
    for count ,img in enumerate(img_matrix):
        merged_array.append([img,signature_matrix[count]])
    return merged_array
def split_sets_back(merged_matrix):
    img_matrix=[]
    sig_matrix=[]
    for  it in merged_matrix:
        img_matrix.append(it[0])
        sig_matrix.append(it[1])
    return img_matrix,sig_matrix

def pca_algorithm():
    global model, labels, pca, norm,scaler
    labels = []
    image_matrix = []
    image_labels = []
    signature_features_matrix=[]
    for user in os.listdir('./users'):
        labels.append(user)
        for filename in os.listdir(f'./users/{user}/faces'):

            img = cv2.imread(f'./users/{user}/faces/{filename}')
            # print(img.shape)
            img = cv2.resize(img, (50, 50))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img.flatten()
            image_matrix.append(img)
            image_labels.append(labels.index(user))
        for filename in os.listdir(f'./users/{user}/signatures'):

            signature=read_signature(f'./users/{user}/signatures/{filename}')
            # print(img.shape)
            stats = get_stats(signature)
            signature_features_matrix.append(stats)



    merged_array=merge_sets(image_matrix, signature_features_matrix);

    split = train_test_split(merged_array, image_labels,test_size=0.1)
    # print(split)
    (traincX, testcX, trainY, testY) = split
    train_img_X,train_sig_X = split_sets_back(traincX)
    #print(norm)
    #trainX = image_matrix
    #trainY = image_labels

    pca = PCA(svd_solver="randomized",n_components=len(train_sig_X[0]), whiten=True)
    trainX = pca.fit_transform(np.array(train_img_X))
    #train_sig_X, norm = normalize(train_sig_X, axis=0, return_norm=True)
    scaler = StandardScaler()
    train_sig_X=scaler.fit_transform(train_sig_X)
    trainX = trainX*alfa+np.array(train_sig_X)*beta

    model = SVC(kernel="poly", C=10, gamma="scale", random_state=3,probability=True)
    model.fit(trainX, trainY)

    test_img_X, test_sig_X = split_sets_back(testcX)
    testX = pca.transform(np.array(test_img_X))
    testX = testX * alfa + scaler.transform(np.array(test_sig_X)) * beta
    predictions = model.predict(testX)
    print(classification_report(testY, predictions, target_names=labels))



def predikt():
    # img_to_photo = cv2.resize(img_to_photo, (50, 50))
    # img_to_photo = cv2.cvtColor(img_to_photo, cv2.COLOR_BGR2GRAY)
    # img_to_photo = img_to_photo.flatten()
    signature = read_signature('predict/signature.txt')
    stats = np.array(get_stats(signature))
    stats=scaler.transform(stats.reshape(1, -1))

    # pca_trans_inp=pca.transform(np.array([img_to_photo]))
    pca_trans_inp=stats*beta
    pred = model.predict(pca_trans_inp)
    pred_proba = model.predict_proba(pca_trans_inp)
    return labels[pred[0]]