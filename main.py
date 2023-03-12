import cv2
import numpy as np
from gui_buttons import Buttons

#intialize the button
button= Buttons()
button.add_button("person",20,20)
button.add_button("car",20,100)
button.add_button("cell phone",20,180)


#opencv DNN
net =cv2.dnn.readNet(r"C:\\Users\\Tobi\\Desktop\\objects detection app\\project\\dnn_model\\yolov4-tiny.weights", r"C:\\Users\\Tobi\\Desktop\\objects detection app\\project\\dnn_model\\yolov4-tiny.cfg")
model=cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320,320),scale=1/255)


classes=[]
with open("C:\\Users\\Tobi\\Desktop\\objects detection app\\project\\dnn_model\\classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name=class_name.strip()
        classes.append(class_name)       
# print("objects list")
# print(classes)
#intialize camera
cap = cv2.VideoCapture(0)
#ENHANCE QUALITY OF CAMERA
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)


def click_button(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        button.button_click(x,y)
        

#CREATE window 
cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", click_button)

while True:
    #get frames
    ret, frame = cap.read()

    #get active button list
    active_buttons= button.active_buttons_list()
    #print("active buttons",active_buttons)

    #object detection
    (class_ids, scores, bboxes)=model.detect(frame)
    for class_id, score, bbox in zip(class_ids, scores, bboxes):
        (x, y, w, h)=bbox
        class_name=classes[class_id]
        
        if class_name in active_buttons:
            #to write label over the frame
            cv2.putText(frame, class_name,(x,y-10),cv2.FONT_HERSHEY_PLAIN, 2,(200,10,50),2)
            #print(x,y,w,h)
            #to draw a rect over the frame (x,y) to get the top left, (x+w),(y+h)to get the bottom right
            cv2.rectangle(frame, (x,y),(x+w,y+h),(200,10,50),3)

    #display button
    button.display_buttons(frame)

    cv2.imshow("Frame",frame)
    key=cv2.waitKey(1)
    if key == 27:
        break
        
cap.release()
cv2.destroyAllWindows()