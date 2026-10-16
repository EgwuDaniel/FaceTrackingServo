import cv2
from faceDetection import faceDetector
import pyfirmata2
import numpy
import time
ynCam = faceDetector()

board = pyfirmata2.Arduino("COM5")

servo_yaw = board.get_pin('d:9:s')
servo_pitch = board.get_pin('d:8:s')

defaultServoPos = [90,90]

current_pitch = 90
current_yaw = 90

servo_pitch.write(current_pitch)
servo_yaw.write(current_yaw)

lastIdled = 0
idleForward = True

def finerPrecision(angle):
    return 1000 + ((angle * 1000)/180)

while True:
    retResult, face_center_x_pos, face_center_y_pos, yaw_offset, pitch_offset = ynCam.detectFirstFace()

    if retResult == True and yaw_offset is not None and pitch_offset is not None:
        yaw_offset = numpy.interp(yaw_offset,[-ynCam.videoCam.get(cv2.CAP_PROP_FRAME_WIDTH) // 2, ynCam.videoCam.get(cv2.CAP_PROP_FRAME_WIDTH) // 2], [-2, 2])
        pitch_offset = numpy.interp(pitch_offset,[-ynCam.videoCam.get(cv2.CAP_PROP_FRAME_HEIGHT) // 2, ynCam.videoCam.get(cv2.CAP_PROP_FRAME_HEIGHT) // 2], [-2, 2])

        current_pitch = numpy.clip(current_pitch + pitch_offset, 0, 180)
        current_yaw = numpy.clip(current_yaw + yaw_offset, 0, 180)

        servo_pitch.write(finerPrecision(current_pitch))
        servo_yaw.write(finerPrecision(current_yaw))

        cv2.putText(ynCam.frame, f"Servo Pitch: {pitch_offset:.2f}, True Pitch: {current_pitch:.2f}", (face_center_x_pos - 30, face_center_y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
        cv2.putText(ynCam.frame, f"Servo Yaw: {yaw_offset:.2f}, True Yaw: {current_yaw:.2f}", (face_center_x_pos - 30, face_center_y_pos + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
        cv2.putText(ynCam.frame, f"Face found : {True}", (0,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)

        lastIdled = time.perf_counter()
    else:
        cv2.putText(ynCam.frame, f"Face found : {False}", (0,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
        currentTime = time.perf_counter()

        if currentTime - lastIdled > 1:
            current_pitch = 90

            if idleForward == True:
                current_yaw = numpy.clip(current_yaw + 1, 0, 180)

                if current_yaw >= 180:
                    idleForward = False
            else:
                current_yaw = numpy.clip(current_yaw - 1, 0, 180)

                if current_yaw <= 0:
                    idleForward = True
            
            servo_pitch.write(finerPrecision(current_pitch))
            servo_yaw.write(finerPrecision(current_yaw))
    
    cv2.imshow("Face Detection for Servo", ynCam.frame)

    if cv2.waitKey(1) == ord('k'):  # ESC key
        break

ynCam.cleanup()
board.exit()