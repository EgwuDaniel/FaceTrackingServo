import cv2
import sys
import numpy

detectorYN = cv2.FaceDetectorYN.create(
    "face_detection_yunet_2026may.onnx",
    config = "",
    input_size = (150,150),
    score_threshold = 0.65,
    nms_threshold = 0.3,
    top_k = 5000,
)
videoCam = cv2.VideoCapture(0)

if not videoCam.isOpened():
    print("Could not open videocam.")
    sys.exit()

hasQuit = False

while hasQuit == False:
    ret, frame = videoCam.read()

    if ret == True:
        height, width = frame.shape[:2]
        detectorYN.setInputSize((width, height))
       
        amountofFaces, detectedFaces = detectorYN.detect(frame)

        if detectedFaces is not None:
            for face in detectedFaces:
                x_pos , y_pos, width, height = face[:4].astype(int)
                startPos = (x_pos, y_pos)
                endPos = (x_pos + width, y_pos + height)
                cv2.rectangle(frame, startPos, endPos, (113,113,255), 2)
                
                center_x_pos = (x_pos + width/2).astype(int)
                center_y_pos  = (y_pos + height/2).astype(int)
                cv2.circle(frame, (center_x_pos, center_y_pos), 2, (214, 112, 218), -1)

       ## 
        cv2.imshow("Face Detection for Servo", frame)

        if cv2.waitKey(1) == ord('k'):
            hasQuit = True

videoCam.release()
cv2.destroyAllWindows()