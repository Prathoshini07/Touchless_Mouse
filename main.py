import cv2
import mediapipe as mp
import random
import util
import pyautogui
from pynput.mouse import Button,Controller
mouse=Controller()


screen_width,screen_height=pyautogui.size()

mphands_detection=mp.solutions.hands
hands=mphands_detection.Hands(
    static_image_mode=False,#As we need to capture video
    model_complexity=1,#to get a better model
    min_detection_confidence=0.7,#The minimum confidence score for a hand detection to be valid is 0.7
    min_tracking_confidence=0.7,#When the confidence of detecting hands is more then 70% we must continue to track
    max_num_hands=1#Maximum hands to be detected is 1
)

def find_finger_tip(processed):
    if processed.multi_hand_landmarks:
        hand_landmarks=processed.multi_hand_landmarks[0]
        return hand_landmarks.landmark[mphands_detection.HandLandmark.INDEX_FINGER_TIP]
    
    return None

def move_mouse(index_finger_tip):
    if index_finger_tip is not None:
        #getting the x coordinate of index finger relative to screen width
        x=int(index_finger_tip.x * screen_width)
        y=int(index_finger_tip.y * screen_height)
        #To move the mouse to new x,y coordinate
        #pyautogui.moveTo(x,y)
        mouse.position=(x,y)
        
        
def is_left_click(list_of_landmarks,thumb_index_distance):
    #check if index finger is bent,thumb is straight and middle finger is straight
    return(util.getting_angle(list_of_landmarks[5],list_of_landmarks[6],list_of_landmarks[8])<50 and
           util.getting_angle(list_of_landmarks[9],list_of_landmarks[10],list_of_landmarks[12]) > 90 and
           thumb_index_distance > 50)
    
def is_right_click(list_of_landmarks,thumb_index_distance):
    #check if middle finger is bent,thumb is straight and index finger is straight
    return(util.getting_angle(list_of_landmarks[5],list_of_landmarks[6],list_of_landmarks[8])>90 and
           util.getting_angle(list_of_landmarks[9],list_of_landmarks[10],list_of_landmarks[12]) <50 and
           thumb_index_distance > 50)
    
def is_double_click(list_of_landmarks,thumb_index_distance):
    #check if index finger is bent,thumb is straight and middle finger is bent
    return(util.getting_angle(list_of_landmarks[5],list_of_landmarks[6],list_of_landmarks[8])<50 and
           util.getting_angle(list_of_landmarks[9],list_of_landmarks[10],list_of_landmarks[12]) < 90 and
           thumb_index_distance > 50)

def is_screenshot(list_of_landmarks,thumb_index_distance):
    #check if index finger is bent,thumb is bent and middle finger is bent
    return(util.getting_angle(list_of_landmarks[5],list_of_landmarks[6],list_of_landmarks[8])<50 and
           util.getting_angle(list_of_landmarks[9],list_of_landmarks[10],list_of_landmarks[12]) < 90 and
           thumb_index_distance < 50)

def gesture_detect(frame,list_of_landmarks,processed):
    if len(list_of_landmarks)>=21:
        
        index_finger_tip=find_finger_tip(processed)
        thumb_index_distance=util.getting_distance([list_of_landmarks[4],list_of_landmarks[5]])
        #for mouse movement using index tip
        if thumb_index_distance <50 and util.getting_angle(list_of_landmarks[5],list_of_landmarks[6],list_of_landmarks[8])>90:
            move_mouse(index_finger_tip)
            
        #left click
        elif is_left_click(list_of_landmarks,thumb_index_distance):
            mouse.press(Button.left)
            mouse.release(Button.left)
            cv2.putText(frame,"Left Click",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            
        #right click
        elif is_right_click(list_of_landmarks,thumb_index_distance):
            mouse.press(Button.right)
            mouse.release(Button.right)
            cv2.putText(frame,"Right Click",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            
        #double click
        elif is_double_click(list_of_landmarks,thumb_index_distance):
            pyautogui.doubleClick()
            cv2.putText(frame,"Double Click",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2)

        #screenshot
        elif is_screenshot(list_of_landmarks,thumb_index_distance):
            im1=pyautogui.screenshot()
            label=random.randint(1,1000)
            im1.save(f'my_screenshot_{label}.png')
            cv2.putText(frame,"Screenshot",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)


def main():
    #get Video frame by frame
    #0 is passed to use the primary camera
    video_capture=cv2.VideoCapture(0)
    draw=mp.solutions.drawing_utils
    
    
    try:
        while video_capture.isOpened():
            #return_val is a bool value and frame contains the frame at a particular milli second
            return_val,frame=video_capture.read()
            
            if not return_val:
                break
            #mirroring the frame for convenience
            frame=cv2.flip(frame,1)
            #mediapipe requires the frame to be passed in RGB format and cv2 uses BGR format as default.So we have to change frome BGR to RGB
            newframeRGB=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            
            processed=hands.process(newframeRGB)
            
            list_of_landmarks=[]
            #if multiple hands are seen,detect only one
            if processed.multi_hand_landmarks:
                hand_landmarks=processed.multi_hand_landmarks[0]
                #drawing hand connections in the original BGR frame
                draw.draw_landmarks(frame,hand_landmarks,mphands_detection.HAND_CONNECTIONS)
                
                #Looping through all the landmarks detected and appending x and y coordinates to the list of landmarks
                for lm in hand_landmarks.landmark:
                    list_of_landmarks.append((lm.x,lm.y))
                    
                #print(list_of_landmarks)
                
            gesture_detect(frame,list_of_landmarks,processed)
            
            cv2.imshow('Frame',frame)
            #wait for 1 millisec after each frame is read and if the keyboard input is q we can break it(Mainly to close the window)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
    finally:
        #finally closing and destroying all the frames
        video_capture.release()
        cv2.destroyAllWindows()
        
        
if __name__=="__main__":
    main()