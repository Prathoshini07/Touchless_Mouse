#2 functions
#1)Getting the angle of the finger
#2)Getting the distance
import numpy as np

def getting_angle(a,b,c):
    #diff between angle created by ab and x axis and bc and x axis
    #subtracting the bigger angel from the smaller one
    radians=np.arctan2(c[1]-b[1],c[0]-b[0])-np.arctan2(a[1]-b[1],a[0]-b[0])
    #converting from radians to degrees
    angle=np.abs(np.degrees(radians))
    return angle

def getting_distance(list_of_landmarks):
    if len(list_of_landmarks)<2:
        return
    
    (x1,y1),(x2,y2)=list_of_landmarks[0],list_of_landmarks[1]
    #finding euclidean distance using np.hypotenuse function
    L=np.hypot(x2-x1,y2-y1)
    #Doing interpolation to convert the resulting distance value to a larger range
    return np.interp(L,[0,1],[0,1000])