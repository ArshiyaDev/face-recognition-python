import cv2
from simple_face_recognition import SimpleFaceReognition

# start use our webcam
cap = cv2.VideoCapture(0)

# Encode faces from folder
sfr = SimpleFaceReognition()
sfr.load_encoding_images("images/")

# show frame by frame
while True:
    #ret is a boolean regarding whether or not there was a return at all
    # at the frame is each frame that is returned
    ret , frame = cap.read()
    # Detect Faces
    face_place,face_names = sfr.detect_known_faces(frame)
    for face_pl , name in zip(face_place,face_names):
        y1 , x2 , y2 , x1 = face_pl[0] , face_pl[1] , face_pl[2],face_pl[3]
        # put text on top of rectangle
        cv2.putText(frame,name,(x1,y1 - 10),cv2.FONT_HERSHEY_DUPLEX,1,(0,0,0),2)
        # Draw rectangle
        cv2.rectangle(frame,(x1,y1),(x2,y2),(255, 0, 255),4)

    cv2.imshow("image",frame)

    key = cv2.waitKey(1)


cap.release()
cv2.destroyAllWindows()