import cv2
from faceDetection import faceDetector
import pyfirmata2
import numpy
import time
ynCam = faceDetector()

board = pyfirmata2.Arduino("COM5")

servo_yaw = board.get_pin('d:9:s')
servo_pitch = board.get_pin('d:8:s')

defaultServoPos = [0,0]

servo_pitch.write(defaultServoPos[0])
servo_yaw.write(defaultServoPos[1])

while True:
    retResult, center_x_pos, center_y_pos = ynCam.detectFirstFace()

    if retResult == True and center_x_pos is not None and center_y_pos is not None:
        servo_yaw_translated = numpy.interp(center_x_pos, [0, ynCam.videoCam.get(cv2.CAP_PROP_FRAME_WIDTH)], [0, 180])
        servo_pitch_translated = numpy.interp(center_y_pos, [0, ynCam.videoCam.get(cv2.CAP_PROP_FRAME_HEIGHT)], [0, 180])

        servo_pitch.write(180 - servo_pitch_translated)
        servo_yaw.write(servo_yaw_translated)

        cv2.putText(ynCam.frame, f"Servo Pitch: {servo_pitch_translated:.2f}", (center_x_pos - 30, center_y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
        cv2.putText(ynCam.frame, f"Servo Yaw: {servo_yaw_translated:.2f}", (center_x_pos - 30, center_y_pos + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
        cv2.putText(ynCam.frame, f"Face found : {retResult}", (0,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)

    else:
        cv2.putText(ynCam.frame, f"Face found : {retResult}", (0,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)

    cv2.imshow("Face Detection for Servo", ynCam.frame)

    if cv2.waitKey(1) == ord('k'):  # ESC key
        break

ynCam.cleanup()
board.exit()