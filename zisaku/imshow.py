import dlib
import cv2
import numpy as np

detector = dlib.get_frontal_face_detector()
PREDICTOR_PATH = r"C:\Users\Owner\Desktop\eye_tracking\zisaku\shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)

def show_image(img, parts):
    for i in parts:
        cv2.circle(img, (i.x, i.y), 3, (255, 0, 0), -1)

    cv2.imshow("landmark", img)

cap = cv2.VideoCapture(0)
while True:
    # カメラ映像の受け取り
    ret, frame = cap.read()

    # detetorによる顔の位置の推測
    dets = detector(frame[:, :, ::-1])
    if len(dets) > 0:
        # predictorによる顔のパーツの推測
        parts = predictor(frame, dets[0]).parts()
        # 映像の描画
        show_image(frame*0, parts) # *0 を取り除くことでそのままの映像が表示されます

    # エスケープキーを押して終了します
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
