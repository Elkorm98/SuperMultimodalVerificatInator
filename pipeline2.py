import os
import cv2
import numpy as np
import librosa
import pandas as pd
import tensorflow as tf
seed=25
decision_threshold=0.6
# comp_discard=3
comp_img=20
comp_sig=8
comp_wrist=4


def feature_extraction2():
    root = "./users/"
    labels = []
    image_matrix = []
    image_labels = []
    wrist_matrix = []
    signature_features_matrix = []
    signature_labels = []
    wrist_labels = []
    for user in os.listdir(f'{root}'):
        labels.append(user)
        for filename in os.listdir(f'{root}/{user}/faces'):
            img = cv2.imread(f'{root}/{user}/faces/{filename}')
            # print(img.shape)
            img = cv2.resize(img, (120, 120))
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # img = img.flatten()
            image_matrix.append(np.array(img))
            image_labels.append(labels.index(user))
        for filename in os.listdir(f'{root}/{user}/signatures'):
            signature = read_signature(f'{root}/{user}/signatures/{filename}')
            # print(img.shape)
            stats = get_stats(signature)
            if stats[1] > 0 and stats[0] < 1:
                signature_features_matrix.append(stats)
                signature_labels.append(labels.index(user))
        for filename in os.listdir(f'{root}/{user}/wrist_gyroscope'):
            # print(f'{root}/{user}/wrist_gyroscope/{filename}')
            wrist_matrix.append(read_wrist_csv(f'{root}/{user}/wrist_gyroscope/{filename}'))
            # print(img.shape)
            wrist_labels.append(labels.index(user))

    IMG_SIZE = (120, 120)
    IMG_SHAPE = IMG_SIZE + (3,)
    base_model = tf.keras.applications.vgg16.VGG16(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')
    preprocess_input = tf.keras.applications.vgg16.preprocess_input
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    inputs = tf.keras.Input(shape=(120, 120, 3))
    x = preprocess_input(inputs)
    x = base_model(x, training=False)
    outputs = global_average_layer(x)
    model = tf.keras.Model(inputs, outputs)

    image_matrix = model.predict(np.array(image_matrix))

    train_sig_X, train_sig_Y, test_sig_X, test_sig_Y = split_mono_set(signature_features_matrix, signature_labels, seed)
    train_img_X, train_img_Y, test_img_X, test_img_Y = split_mono_set(image_matrix, image_labels, seed)
    train_wrist_X, train_wrist_Y, test_wrist_X, test_wrist_Y = split_mono_set(wrist_matrix, wrist_labels, seed)
    return train_sig_X, train_sig_Y, test_sig_X, test_sig_Y, train_img_X, train_img_Y, test_img_X, test_img_Y, train_wrist_X, train_wrist_Y, test_wrist_X, test_wrist_Y

def training_testing_report2(trainX, trainY, testX, testY):
  class_model = GaussianNB()
  class_model.fit(trainX, trainY)
  predictions = class_model.predict(testX)
  res=classification_report(testY, predictions, target_names=labels)
  print(res)
  return class_model

def transformations_concatenation1(train_sig_X, train_sig_Y, test_sig_X, test_sig_Y,
                                     train_img_X, train_img_Y, test_img_X, test_img_Y,
                                     train_wrist_X, train_wrist_Y, test_wrist_X, test_wrist_Y):
  pca = PCA(svd_solver="randomized",n_components=comp_img, whiten=True)
  train_img_X = pca.fit_transform(np.array(train_img_X))


  scaler_sig = StandardScaler()
  train_sig_X=scaler_sig.fit_transform(train_sig_X)
  pca_sig = PCA(svd_solver="randomized", n_components=comp_sig, whiten=True)
  train_sig_X=pca_sig.fit_transform(train_sig_X)
  # train_sig_X=np.fliplr(train_sig_X)

  scaler_wrist= StandardScaler()
  train_wrist_X=scaler_wrist.fit_transform(train_wrist_X)
  pca_wrist = PCA(svd_solver="randomized", n_components=comp_wrist, whiten=True)
  train_wrist_X=pca_wrist.fit_transform(train_wrist_X)
  # train_wrist_X=np.fliplr(train_wrist_X)

  test_img_X=pca.transform(np.array(test_img_X))

  test_sig_X = scaler_sig.transform(np.array(test_sig_X))
  test_sig_X = pca_sig.transform(test_sig_X)
  # test_sig_X = np.fliplr(test_sig_X)

  test_wrist_X = scaler_wrist.transform(np.array(test_wrist_X))
  test_wrist_X = pca_wrist.transform(test_wrist_X)
  # test_sig_X = np.fliplr(test_sig_X)

  merged_trainX, merged_trainY = merge_sets(train_img_X, train_sig_X, train_wrist_X, train_img_Y,
                                                      train_sig_Y,train_wrist_Y)
  train_img_X, train_sig_X, train_wrist_X = split_sets_back(merged_trainX)

  merged_testX, merged_testY = merge_sets(test_img_X, test_sig_X, test_wrist_X, test_img_Y,
                                                      test_sig_Y,test_wrist_Y)
  test_img_X, test_sig_X, test_wrist_X = split_sets_back(merged_testX)

  trainX=np.concatenate((np.array(train_img_X),np.array(train_sig_X),np.array(train_wrist_X)),axis=1)
  trainY=merged_trainY
  testX=np.concatenate((np.array(test_img_X),np.array(test_sig_X),np.array(test_wrist_X)),axis=1)
  testY=merged_testY
  return trainX, trainY, testX, testY

def read_wrist_csv2(filename):
  dataframe = pd.read_csv(filename, skiprows=3)
  features=[]

  dataframe = dataframe.sort_values(by='HostTimestamp (ms)')
  dataframediff = dataframe[['X (dps)','Y (dps)','Z (dps)','HostTimestamp (ms)']].diff(periods=1)
  dataframediff=dataframediff[dataframediff['HostTimestamp (ms)']>15]
  # print(np.mean(dataframediff['HostTimestamp (ms)']))
  # print(dataframediff['HostTimestamp (ms)'])
  # print(dataframediff.shape)
  fs=1/np.mean(dataframediff['HostTimestamp (ms)'])*1000
  rms=librosa.feature.rms(y=np.array(dataframediff['X (dps)']))
  features += [np.mean(rms)]
  features += [np.std(rms)]
  rms=librosa.feature.rms(y=np.array(dataframediff['Y (dps)']))
  features += [np.mean(rms)]
  features += [np.std(rms)]
  rms=librosa.feature.rms(y=np.array(dataframediff['Z (dps)']))
  features += [np.mean(rms)]
  features += [np.std(rms)]

  zrc=librosa.feature.zero_crossing_rate(y=np.array(dataframediff['X (dps)']))
  features += [np.mean(zrc)]
  features += [np.std(zrc)]
  zrc=librosa.feature.zero_crossing_rate(y=np.array(dataframediff['Y (dps)']))
  features += [np.mean(zrc)]
  features += [np.std(zrc)]
  zrc=librosa.feature.zero_crossing_rate(y=np.array(dataframediff['Z (dps)']))
  features += [np.mean(zrc)]
  features += [np.std(zrc)]

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
    img_matrix=[]
    sig_matrix=[]
    wrist_matrix=[]
    for  it in merged_matrix:
        img_matrix.append(it[0])
        sig_matrix.append(it[1])
        wrist_matrix.append(it[2])
    return img_matrix,sig_matrix, wrist_matrix


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
