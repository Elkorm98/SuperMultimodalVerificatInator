import os
import cv2
import numpy as np
import librosa
import pandas as pd
# import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
import sqlite3

def read_wrist_csv2(filename):
    dataframe = pd.read_csv(filename, skiprows=3)
    features = []

    dataframe = dataframe.sort_values(by='HostTimestamp (ms)')

    dataframediff = dataframe[['X (dps)', 'Y (dps)', 'Z (dps)', 'HostTimestamp (ms)']]

    dataframediff = dataframediff.diff(periods=1)
    dataframediff = dataframediff[dataframediff['HostTimestamp (ms)'] > 15]
    dataframediff = dataframediff.head(80)
    # print(np.mean(dataframediff['HostTimestamp (ms)']))
    # print(dataframediff['HostTimestamp (ms)'])
    # print(dataframediff.shape)
    fs = 1 / np.mean(dataframediff['HostTimestamp (ms)']) * 1000

    rms = librosa.feature.rms(y=np.array(dataframediff['X (dps)']))
    features += [np.mean(rms)]
    features += [np.std(rms)]
    rms = librosa.feature.rms(y=np.array(dataframediff['Y (dps)']))
    features += [np.mean(rms)]
    features += [np.std(rms)]
    rms = librosa.feature.rms(y=np.array(dataframediff['Z (dps)']))
    features += [np.mean(rms)]
    features += [np.std(rms)]
    try:
        zrc = librosa.feature.zero_crossing_rate(y=np.array(dataframediff['X (dps)']))
        features += [np.mean(zrc)]
        features += [np.std(zrc)]
        zrc = librosa.feature.zero_crossing_rate(y=np.array(dataframediff['Y (dps)']))
        features += [np.mean(zrc)]
        features += [np.std(zrc)]
        zrc = librosa.feature.zero_crossing_rate(y=np.array(dataframediff['Z (dps)']))
        features += [np.mean(zrc)]
        features += [np.std(zrc)]
    except:
        print(filename)
    return features


def read_wrist_csv(filename):
  dataframe = pd.read_csv(filename, skiprows=3)
  features=[]

  dataframe = dataframe.sort_values(by='HostTimestamp (ms)')
  dataframediff = dataframe[['X (dps)','Y (dps)','Z (dps)','HostTimestamp (ms)']].diff(periods=1)
  dataframediff=dataframediff[dataframediff['HostTimestamp (ms)']>15]
  # print(np.mean(dataframediff['HostTimestamp (ms)']))
  # print(dataframediff['HostTimestamp (ms)'])
  # print(dataframediff.shape)
  fs=1/np.mean(dataframediff['HostTimestamp (ms)'])*1000
  mfccs = librosa.feature.mfcc(y=np.array(dataframediff['X (dps)']), sr=fs, n_mfcc=5,n_fft=20).tolist()
  for i,el in enumerate(mfccs):
    mfccs[i]=el[0]
  features+=mfccs
  mfccs = librosa.feature.mfcc(y=np.array(dataframediff['Y (dps)']), sr=fs, n_mfcc=5,n_fft=20).tolist()
  for i,el in enumerate(mfccs):
    mfccs[i]=el[0]
  features+=mfccs
  mfccs = librosa.feature.mfcc(y=np.array(dataframediff['Z (dps)']), sr=fs, n_mfcc=5,n_fft=20).tolist()
  for i,el in enumerate(mfccs):
    mfccs[i]=el[0]
  features+=mfccs
  cent = librosa.feature.spectral_centroid(y=np.array(dataframediff['X (dps)']), sr=fs,n_fft=20)
  features += [np.mean(cent)]
  features += [np.std(cent)]
  cent = librosa.feature.spectral_centroid(y=np.array(dataframediff['Y (dps)']), sr=fs,n_fft=20)
  features += [np.mean(cent)]
  features += [np.std(cent)]
  cent = librosa.feature.spectral_centroid(y=np.array(dataframediff['Z (dps)']), sr=fs,n_fft=20)
  features += [np.mean(cent)]
  features=np.array(features)
  features += [np.std(cent)]
  return features


def split_sets_back(merged_matrix):
    sig_matrix=[]
    wrist_matrix=[]
    for  it in merged_matrix:
        sig_matrix.append(it[0])
        wrist_matrix.append(it[1])
    return sig_matrix, wrist_matrix

def split_mono_set(array, labels,seed):
    traincX = []
    trainY = []
    testcX = []
    testY = []
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    for train_index, test_index in sss.split(array, labels):
        for ind in train_index:
            traincX.append(array[ind])
            trainY.append(labels[ind])
        for ind in test_index:
            testcX.append(array[ind])
            testY.append(labels[ind])
    return traincX,trainY,testcX,testY

def merge_sets(img_matrix,signature_matrix,wrist_matrix,image_labels,signature_labels,wrist_labels):
  merged_array=[]
  new_labels=[]
  for count ,sig in enumerate(signature_matrix):
      # if(image_labels[count]==signature_labels[count2]):
      merged_array.append([img_matrix[count],signature_matrix[count],wrist_matrix[count]])
      new_labels.append(image_labels[count])
  return merged_array,new_labels

def merge_sets_simple(signature_matrix,wrist_matrix,signature_labels,wrist_labels):
  merged_array=[]
  new_labels=[]
  for count ,sig in enumerate(signature_matrix):
    for count2, wrist in enumerate(wrist_matrix):
      if(wrist_labels[count2]==signature_labels[count]):
        merged_array.append([signature_matrix[count],wrist_matrix[count2]])
        wrist_labels=np.delete(wrist_labels, count2, 0)
        wrist_matrix=np.delete(wrist_matrix, count2, 0)
        new_labels.append(signature_labels[count])
        break
  return merged_array,new_labels

# euclidean distnace between 2 points (input 2 points)
def distance(curpoint,prevpoint):
  return np.sqrt((curpoint[0]-prevpoint[0])**2+(curpoint[1]-prevpoint[1])**2)

# return acc stats (min,max, mean, zerocrossings) stat (input non-stationary points with vel and acc)
def acceleration_stats(points):
  arr=np.array(points)
  arr=np.transpose(arr)
  zero_crossings = librosa.zero_crossings(arr[6], pad=False)
  return(min(arr[6]),max(arr[6]),np.mean(arr[6]),sum(zero_crossings))

# return vector with acc/dur stat (input non-stationary points with vel and acc)
def acceleration_by_duration(points):
  sum=0
  for i in range(1,len(points)):
    if(points[i][6]>0):
      sum+= points[i][4]-points[i-1][4]
  dur=points[len(points)-1][4]-points[0][4]
  return sum/dur

# return vector with all strokes durations (input non-stationary points)
def calc_vel_acc(points):
  points[0].append(0)  #0 velocity at 1st point
  points[0].append(0)  #0 acceleration at 1st point

  for i in range(1,len(points)):
    dist=distance(points[i],points[i-1])
    velocity= dist/(points[i][4]-points[i-1][4])
    points[i].append(velocity)
    acceleration=(points[i][5]-points[i-1][5])/(points[i][4]-points[i-1][4])
    points[i].append(acceleration)

# return vector with all strokes durations (input all points)
def stroke_dur(points):
  strokes_durs=[]
  start=points[0][4]
  for i in range(1,len(points)):
    if(points[i][3]>0 and points[i-1][3]==0):
      start=points[i][4]
    if(points[i][3]==0 and points[i-1][3]>0):
      strokes_durs.append(points[i][4]-start)
  return strokes_durs

# return vector with all strokes lengths (input all points)
def stroke_len(points):
  strokes_lens=[]
  sum=0
  if points[0][3]>0:
    start=1
  else:
    start=0
  for i in range(1,len(points)):
    if(points[i][3]>0 and points[i-1][3]==0):
      start=1
    if(points[i][3]==0 and points[i-1][3]>0):
      start=0
      strokes_lens.append(sum)
      sum=0
    if(start==1):
      sum+=distance(points[i],points[i-1])
  return strokes_lens

def average_stroke_press(points):
  strokes_press=[]
  it = 0
  sum = 0
  if points[0][3]>0:
    start=1
  else:
    start=0
  for i in range(1,len(points)):
    if(points[i][3]>0 and points[i-1][3]==0):
      it=0
      sum=0
      start=1
    if(points[i][3]==0 and points[i-1][3]>0):
      start=0
      strokes_press.append(sum/it)
    if(start==1):
      sum+=points[i][3]
      it+=1
  return strokes_press

#visualization of signature
def vizualization(points):
  im=np.zeros((200,200))
  for p in points:
    im[200-int(p[1]),int(p[0])]=255
  cv2.imshow(im)

# remove points with no x and y changes or no time changes
def remove_stationary_points(points):
  new_points=[]
  for i in range(1,len(points)):
    if((points[i][0]!=points[i-1][0] or points[i][1]!=points[i-1][1])and (points[i][4]!=points[i-1][4])):
      new_points.append(points[i])
  return new_points
def remove_noisy_points(points):
  new_points=[]
  for i in range(1,len(points)-1):
    if(~(points[i][3]==0 and points[i-1][3]>0 and points[i+1][3]>0)):
      new_points.append(points[i])
  return new_points

class Signature_features:
  def __init__(self, acc_by_dur, m_str_dur,std_str_dur,m_str_len,std_str_len, acc_stats,m_press,std_press):
    self.acc_by_dur = acc_by_dur
    self.m_str_dur = m_str_dur
    self.std_str_dur=std_str_dur
    self.m_str_len = m_str_len
    self.std_str_len=std_str_len
    self.acc_stats = acc_stats
    self.m_press= m_press
    self.std_press=std_press

# 0 acc_by_dur
# 1 m_str_dur
# 2 std_str_dur
# 3 m_str_len
# 4 std_str_len
# 5 m_str_len
# 6 min_acc
# 7 max_acc
# 8 mean_acc
# 9 zrc_acc
# 10 m_press
# 11 std_press
def get_stats(lines):
  points=[]
  for l in lines:
    a=l.replace("\n","")
    a=a.split(", ")
    a=[float(numeric_string) for numeric_string in a]
    points.append(a)
  points_reduced=remove_stationary_points(points)
  points_filtered=remove_noisy_points(points)
  calc_vel_acc(points_reduced)
  return [acceleration_by_duration(points_reduced), np.mean(stroke_dur(points_filtered)),np.std(stroke_dur(points_filtered)), np.mean(stroke_len(points_filtered)), np.std(stroke_len(points_filtered)), acceleration_stats(points_reduced)[0],acceleration_stats(points_reduced)[1],acceleration_stats(points_reduced)[2],acceleration_stats(points_reduced)[3],np.mean(average_stroke_press(points_filtered)), np.std(average_stroke_press(points_filtered))]

def read_signature(path):
  with open(path) as f:
    lines = f.readlines()
  return lines

def decision_module(class_model,X, Y):
  X=np.array(X)
  Y=np.array(Y)
  dim=X.ndim
  if (dim==1):
    X = np.array([X])
    Y = np.array([Y])
  res = np.array(class_model.predict_proba(X))
  res_matrix = np.array([np.argmax(res,axis=1),np.max(res,axis=1)])
  max_cond = np.array(Y) == res_matrix[0]
  thresh_cond = decision_threshold < res_matrix[1]
  test_res = thresh_cond * max_cond
  # print(Y)
  # print(res_matrix)
  if(dim==1):
    return test_res[0]
  else:
    return test_res


seed=25
decision_threshold=0.6
# comp_discard=3
comp_img=20
comp_sig=10
comp_wrist=6
labels = []

def feature_extraction2():
    root = "./users/"
    global labels, image_matrix, image_labels, wrist_matrix, signature_features_matrix, signature_labels, wrist_labels
    labels = []
    image_matrix = []
    image_labels = []
    wrist_matrix = []
    signature_features_matrix = []
    signature_labels = []
    wrist_labels = []
    for user in os.listdir(f'{root}'):
        labels.append(user)
        # for filename in os.listdir(f'{root}/{user}/faces'):
        #     img = cv2.imread(f'{root}/{user}/faces/{filename}')
        #     # print(img.shape)
        #     img = cv2.resize(img, (120, 120))
        #     # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #     # img = img.flatten()
        #     image_matrix.append(np.array(img))
        #     image_labels.append(labels.index(user))
        for filename in os.listdir(f'{root}/{user}/signatures'):
            signature = read_signature(f'{root}/{user}/signatures/{filename}')
            # print(img.shape)
            stats = get_stats(signature)
            if stats[1] > 0 and stats[0] < 1:
                signature_features_matrix.append(stats)
                signature_labels.append(labels.index(user))
        for filename in os.listdir(f'{root}/{user}/wrist_gyroscope'):
            # print(f'{root}/{user}/wrist_gyroscope/{filename}')
            wrist_matrix.append(read_wrist_csv2(f'{root}/{user}/wrist_gyroscope/{filename}'))
            # print(img.shape)
            wrist_labels.append(labels.index(user))

    train_sig_X, train_sig_Y, test_sig_X, test_sig_Y = split_mono_set(signature_features_matrix, signature_labels, seed)
    # train_img_X, train_img_Y, test_img_X, test_img_Y = split_mono_set(image_matrix, image_labels,seed)
    train_wrist_X, train_wrist_Y, test_wrist_X, test_wrist_Y = split_mono_set(wrist_matrix, wrist_labels, seed)
    return train_sig_X, train_sig_Y, test_sig_X, test_sig_Y, train_wrist_X, train_wrist_Y, test_wrist_X, test_wrist_Y

def training_testing_report2(trainX, trainY, testX, testY):
  global class_model, labels
  class_model = GaussianNB()
  class_model.fit(trainX, trainY)
  predictions = class_model.predict(testX)
  res=classification_report(testY, predictions, target_names=labels)
  print(res)
  return class_model

def transformations_concatenation1(model_mode,train_sig_X, train_sig_Y, test_sig_X, test_sig_Y,
                                     train_wrist_X, train_wrist_Y, test_wrist_X, test_wrist_Y):
    global labels, scaler_sig, pca_sig, scaler_wrist, pca_wrist, image_matrix, image_labels, wrist_matrix, signature_features_matrix, signature_labels, wrist_labels

    # train_sig_X, train_sig_Y, test_sig_X, test_sig_Y = split_mono_set(signature_features_matrix, signature_labels,seed)
    # # train_img_X, train_img_Y, test_img_X, test_img_Y = split_mono_set(image_matrix, image_labels,seed)
    # train_wrist_X, train_wrist_Y, test_wrist_X, test_wrist_Y = split_mono_set(wrist_matrix, wrist_labels,seed)

    # pca = PCA(svd_solver="randomized",n_components=len(train_sig_X[0])-comp_discard, whiten=True)
    # train_img_X = pca.fit_transform(np.array(train_img_X))

    scaler_sig = StandardScaler()
    train_sig_X = scaler_sig.fit_transform(train_sig_X)
    pca_sig = PCA(svd_solver="randomized", n_components=comp_sig, whiten=True)
    train_sig_X = pca_sig.fit_transform(train_sig_X)
    # train_sig_X=np.fliplr(train_sig_X)

    scaler_wrist = StandardScaler()
    train_wrist_X = scaler_wrist.fit_transform(train_wrist_X)
    pca_wrist = PCA(svd_solver="randomized", n_components=comp_wrist, whiten=True)
    train_wrist_X = pca_wrist.fit_transform(train_wrist_X)
    # train_wrist_X=np.fliplr(train_wrist_X)

    # test_img_X=pca.transform(np.array(test_img_X))

    test_sig_X = scaler_sig.transform(np.array(test_sig_X))
    test_sig_X = pca_sig.transform(test_sig_X)
    # test_sig_X = np.fliplr(test_sig_X)

    test_wrist_X = scaler_wrist.transform(np.array(test_wrist_X))
    test_wrist_X = pca_wrist.transform(test_wrist_X)
    # test_sig_X = np.fliplr(test_sig_X)

    merged_trainX, merged_trainY = merge_sets_simple(train_sig_X, train_wrist_X,
                                                     train_sig_Y, train_wrist_Y)
    train_sig_X, train_wrist_X = split_sets_back(merged_trainX)

    merged_testX, merged_testY = merge_sets_simple(test_sig_X, test_wrist_X,
                                                   test_sig_Y, test_wrist_Y)
    test_sig_X, test_wrist_X = split_sets_back(merged_testX)

    if model_mode=='both':
        trainX = np.concatenate((np.array(train_sig_X), np.array(train_wrist_X)), axis=1)
        trainY = merged_trainY
        testX = np.concatenate((np.array(test_sig_X), np.array(test_wrist_X)), axis=1)
        testY = merged_testY
    elif model_mode == 'signature':
        trainX = np.array(train_sig_X)
        trainY = merged_trainY
        testX = np.array(test_sig_X)
        testY = merged_testY
    elif model_mode == 'wrist_gyroscope':
        trainX = np.array(train_wrist_X)
        trainY = merged_trainY
        testX = np.array(test_wrist_X)
        testY = merged_testY
    return trainX, trainY, testX, testY

def input_test(sig_filename, wrist_filename, username, model_mode):
    global model, pca, scaler_sig, pca_sig, scaler_wrist, pca_wrist, class_model
    # img=np.array(img)
    # img = cv2.resize(img, (120, 120))
    # img = np.array(img)
    signature = read_signature(sig_filename)
    sig_stats = get_stats(signature)
    wrist_stats=read_wrist_csv2(wrist_filename)

    # img_X= model.predict(np.array([img]))
    # img_X= pca.transform(np.array(img_X))

    sig_X = scaler_sig.transform(np.array([sig_stats]))
    sig_X = pca_sig.transform(sig_X)

    wrist_X = scaler_wrist.transform(np.array([wrist_stats]))
    wrist_X = pca_wrist.transform(wrist_X)
    if model_mode == 'both':
        X = np.concatenate((np.array(sig_X), np.array(wrist_X)), axis=1)
    elif model_mode == 'signature':
        X = np.array(sig_X)
    elif model_mode == 'wrist_gyroscope':
        X = np.array(wrist_X)

    answer=decision_module(class_model, X, labels.index(username))
    print(answer)
    return answer



