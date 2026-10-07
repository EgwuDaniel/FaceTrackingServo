import cv2
import sys
import numpy

class faceDetector:
    def __init__(self):
        self.DetectorYN = cv2.FaceDetectorYN.create(
            "face_detection_yunet_2026may.onnx",
            config = "",
            input_size = (150,150),
            score_threshold = 0.65,
            nms_threshold = 0.3,
            top_k = 5000,
        )

        self.videoCam = cv2.VideoCapture(0)

        if not self.videoCam.isOpened():
            print("Could not open videocam.")
            sys.exit()

    def detectFirstFace(self):
        ret, self.frame = self.videoCam.read()

        center_x_pos, center_y_pos = None, None

        if ret == True:
            height, width = self.frame.shape[:2]
            self.DetectorYN.setInputSize((width, height))
        
            amountofFaces, detectedFaces = self.DetectorYN.detect(self.frame)

            if detectedFaces is not None:
                face = detectedFaces[0] ## priority face

                x_pos , y_pos, width, height = face[:4].astype(int)
                startPos = (x_pos, y_pos)
                endPos = (x_pos + width, y_pos + height)
                cv2.rectangle(self.frame, startPos, endPos, (113,113,255), 2)
                
                center_x_pos = (x_pos + width/2).astype(int)
                center_y_pos  = (y_pos + height/2).astype(int)
                cv2.circle(self.frame, (center_x_pos, center_y_pos), 2, (214, 112, 218), -1)
            
            if cv2.waitKey(1) == ord('k'):
                hasQuit = True

        return ret, center_x_pos, center_y_pos

    def cleanup(self):
        self.videoCam.release()
        cv2.destroyAllWindows()
        del self.DetectorYN
