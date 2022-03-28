import numpy as np
import cv2
import librosa, librosa.display
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
  start=0
  for i in range(1,len(points)):
    if(points[i][3]>0 and points[i-1][3]==0):
      start=points[i][4]
    if(points[i][3]==0 and points[i-1][3]>0):
      strokes_durs.append(points[i][4]-start)
  return strokes_durs

# return vector with all strokes lengths (input all points)
def stroke_len(points):
  strokes_lens=[]
  start=0
  sum=0
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

def get_stats(lines):
  points=[]
  for l in lines:
    a=l.replace("\n","")
    a=a.split(", ")
    a=[float(numeric_string) for numeric_string in a]
    points.append(a)
  points_reduced=remove_stationary_points(points)
  calc_vel_acc(points_reduced)
  return [acceleration_by_duration(points_reduced), np.mean(stroke_dur(points)),np.std(stroke_dur(points)), np.mean(stroke_len(points)), np.std(stroke_len(points)), acceleration_stats(points_reduced)[0],acceleration_stats(points_reduced)[1],acceleration_stats(points_reduced)[2],acceleration_stats(points_reduced)[3],np.mean(average_stroke_press(points)), np.std(average_stroke_press(points))]

def read_signature(path):
  with open(path) as f:
    lines = f.readlines()
  return lines

# signature= read_signature('test_files/Lukasz (1).txt')
# stats=get_stats(signature)
# print("siema")