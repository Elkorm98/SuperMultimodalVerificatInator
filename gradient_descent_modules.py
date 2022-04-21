from random import random
from timeit import default_timer as timer

def gradient_descent(gradient, start ,minx , maxx, learn_rate=0.1, stop_time = 30, n_iter=200):
    vector = start
    start_time = timer()
    for _ in range(n_iter):
        diff = -learn_rate * gradient(vector)
        vector += diff
        if vector < minx:
          vector=minx
        elif vector > maxx:
          vector=maxx
        if ((timer()-start_time)*1000>stop_time):
          break
    return vector