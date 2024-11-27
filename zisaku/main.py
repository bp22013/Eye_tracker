SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 660

THRESHOLD = 100 # 目元の明るさによって調整してください

# 画像比率を維持しながら縦横を指定して拡大する 余った部分は白でパディング 
def resize_with_pad(image,new_shape,padding_color=[255,255,255]):
    original_shape = (image.shape[1], image.shape[0])
    ratio = float(max(new_shape))/max(original_shape)
    new_size = tuple([int(x*ratio) for x in original_shape])
    image = cv2.resize(image, new_size)
    delta_w = new_shape[0] - new_size[0]
    delta_h = new_shape[1] - new_size[1]
    top, bottom = delta_h//2 + delta_h-(delta_h//2), 0
    left, right = delta_w//2, delta_w-(delta_w//2)
    top, bottom, left, right = max(top,0),max(bottom,0),max(right,0),max(left,0)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)
    return image

# 左目 36:左端 37, 38:上 39:右端 40,41:下
# 右目 42:左端 43, 44:上 45:右端 46,47:下
def show_eye(img, parts,xy=[None,None,None,None]):
    # 左目の左上のカット
    delta = (parts[37].y-parts[36].y)/(parts[37].x-parts[36].x)
    y = parts[36].y
    for x in range(parts[36].x,parts[37].x+1):
        img[0:round(y),x] = 255
        y += delta
        
    # 左目の左下のカット
    delta = (parts[41].y-parts[36].y)/(parts[41].x-parts[36].x)
    y = parts[36].y
    for x in range(parts[36].x,parts[41].x+1):
        img[round(y):,x] = 255
        y += delta
        
    # 左目の上部のカット
    delta = (parts[38].y-parts[37].y)/(parts[38].x-parts[37].x)
    y = parts[37].y
    for x in range(parts[37].x,parts[38].x+1):
        # print(x,round(y))
        img[0:round(y),x] = 255
        y += delta
        
    # 左目の右上部のカット
    delta = (parts[39].y-parts[38].y)/(parts[39].x-parts[38].x)
    y = parts[38].y
    for x in range(parts[38].x,parts[39].x+1):
        # print(x,round(y))
        img[0:round(y),x] = 255
        y += delta
        
    # 左目の右下部のカット
    delta = (parts[39].y-parts[40].y)/(parts[39].x-parts[40].x)
    y = parts[40].y
    for x in range(parts[40].x,parts[39].x+1):
        # print(x,round(y))
        img[round(y):,x] = 255
        y += delta
        
    # 左目の下部のカット
    delta = (parts[41].y-parts[40].y)/(parts[41].x-parts[40].x)
    y = parts[41].y
    for x in range(parts[41].x,parts[40].x+1):
        # print(x,round(y))
        img[round(y):,x] = 255
        y += delta
    
    # 目の位置を求める
    x0_right = parts[36].x
    x1_right = parts[39].x
    y0_right = min(parts[37].y, parts[38].y)
    y1_right = max(parts[40].y, parts[41].y)
    
    # 目の長方形をスライスで切り出す
    right_eye = img[y0_right:y1_right, x0_right:x1_right]
    
    # そのままの大きさでは見づらいので拡大する
    right_eye = resize_with_pad(right_eye,(600,300),padding_color=[255,255,255])
    
    # 重心を求めるために二値化する
    img_gray = cv2.cvtColor(right_eye, cv2.COLOR_BGR2GRAY)
    ret, img_bin = cv2.threshold(img_gray, THRESHOLD, 255, cv2.THRESH_BINARY)
    # ret2, img_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU)
    
    # 重心を求める
    img_rev = 255 - img_bin # 白黒を反転する
    mu = cv2.moments(img_rev, False) # 反転した画像の白い部分の重心を求める
    
    x, y = None, None
    try:
        x,y= int(mu["m10"]/mu["m00"]) , int(mu["m01"]/mu["m00"])
        # 重心(=目の中心)を描画する
        cv2.circle(img_bin, (x, y), 5, (122), -1)
        cv2.circle(img_bin, (x, y), 20, (122), 1)
        cv2.circle(right_eye, (x, y), 5, (0,0,255), -1)
        cv2.circle(right_eye, (x, y), 20, (0,0,255), 1)
    except:
        pass
    
    # 目線の向きに関する処理
    if None not in xy and x != None and y != None:
        # 画面比率
        x_r, x_l, y_t, y_b = xy
        x_ratio = SCREEN_WIDTH / (x_l - x_r)
        y_ratio = SCREEN_HEIGHT / (y_b - y_t)

        # x_r = 0, x_l = SCREEN_WIDTH  (x-x_r)*x_ratio
        x = int((x - x_r) * x_ratio)
        y = int((y - y_t) * y_ratio)

        x = SCREEN_WIDTH - x  # 左右反転

        screen = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3)) + 255

        cv2.line(screen, (x_r, y_t), (x_l, y_t), (0, 255, 0), thickness=3, lineType=cv2.LINE_4)
        cv2.line(screen, (x_r, y_b), (x_l, y_b), (0, 255, 0), thickness=3, lineType=cv2.LINE_4)
        cv2.line(screen, (x_r, y_b), (x_r, y_t), (0, 255, 0), thickness=3, lineType=cv2.LINE_4)
        cv2.line(screen, (x_l, y_b), (x_l, y_t), (0, 255, 0), thickness=3, lineType=cv2.LINE_4)

        cv2.putText(screen, "Direction", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
        cv2.putText(screen, f"({x}, {y})", (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
        cv2.circle(screen, (x, y), 50, (0, 0, 255), -1)

        # `screen`ウィンドウを表示
        cv2.imshow("Screen", screen)
    
    cv2.imshow("Right Eye (bin)", img_bin)
    cv2.imshow("Right Eye", right_eye)
    return (x,y)

def set_xy(x_max,x_min,y_max,y_min):
    if None not in [x_max,x_min,y_max,y_min]:
        return round(x_min),round(x_max),round(y_min),round(y_max)
    else:
        return None,None,None,None

import dlib
import cv2
import numpy as np
import winsound

detector = dlib.get_frontal_face_detector()
PREDICTOR_PATH = r"C:\Users\Owner\Desktop\eye_tracking\zisaku\shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)

cap = cv2.VideoCapture(0)
print(cap.get(cv2.CAP_PROP_FPS))

x_max = None
x_min = None
y_max = None
y_min = None

x_r = None
x_l = None
y_t = None
y_b = None

while True:
    # カメラ映像の受け取り
    ret, frame = cap.read()

    # detetorによる顔の位置の推測
    dets = detector(frame[:, :, ::-1])
    if len(dets) > 0:
        # predictorによる顔のパーツの推測
        parts = predictor(frame, dets[0]).parts()
        # 映像の描画
        eye_pos = show_eye(frame, parts,xy=[x_r,x_l,y_t,y_b])
    
    key = cv2.waitKey(1)
        
    # rを押して右端のx座標を取ります
    if key == ord("r"):
        right_pos = []
        cnt = 0
        while cnt <= cap.get(cv2.CAP_PROP_FPS):
            # カメラ映像の受け取り
            ret, frame = cap.read()

            # detetorによる顔の位置の推測
            dets = detector(frame[:, :, ::-1])
            if len(dets) > 0:
                # predictorによる顔のパーツの推測
                parts = predictor(frame, dets[0]).parts()
                # 映像の描画
                eye_pos = show_eye(frame, parts)
                key = cv2.waitKey(1)
                if eye_pos[0] != None:
                    right_pos.append(eye_pos[0])
                    cnt += 1
                
        x_min = sum(right_pos)/len(right_pos)
        print(f"x_min: {x_min}")
        winsound.Beep(440, 500) # 記録が終わったらビープ音を鳴らす 
        x_r,x_l,y_t,y_b = set_xy(x_max,x_min,y_max,y_min) # 端の設定
        
    # lを押して左端のx座標を取ります
    if key == ord("l"):
        left_pos = []
        cnt = 0
        while cnt <= cap.get(cv2.CAP_PROP_FPS):
            # カメラ映像の受け取り
            ret, frame = cap.read()

            # detetorによる顔の位置の推測
            dets = detector(frame[:, :, ::-1])
            if len(dets) > 0:
                # predictorによる顔のパーツの推測
                parts = predictor(frame, dets[0]).parts()
                # 映像の描画
                eye_pos = show_eye(frame, parts)
                key = cv2.waitKey(1)
                if eye_pos[0] != None:
                    left_pos.append(eye_pos[0])
                    cnt += 1
                
        x_max = sum(left_pos)/len(left_pos)
        print(f"x_max: {x_max}")
        winsound.Beep(440, 500) # 記録が終わったらビープ音を鳴らす 
        x_r,x_l,y_t,y_b = set_xy(x_max,x_min,y_max,y_min) # 端の設定
        
    # tを押して上端のy座標を取ります
    if key == ord("t"):
        top_pos = []
        cnt = 0
        while cnt <= cap.get(cv2.CAP_PROP_FPS):
            # カメラ映像の受け取り
            ret, frame = cap.read()

            # detetorによる顔の位置の推測
            dets = detector(frame[:, :, ::-1])
            if len(dets) > 0:
                # predictorによる顔のパーツの推測
                parts = predictor(frame, dets[0]).parts()
                # 映像の描画
                eye_pos = show_eye(frame, parts)
                key = cv2.waitKey(1)
                if eye_pos[0] != None:
                    top_pos.append(eye_pos[1])
                    cnt += 1
                
        y_min = sum(top_pos)/len(top_pos)
        print(f"y_min: {y_min}")
        winsound.Beep(440, 500) # 記録が終わったらビープ音を鳴らす 
        x_r,x_l,y_t,y_b = set_xy(x_max,x_min,y_max,y_min) # 端の設定
        
    # tを押して上端のy座標を取ります
    if key == ord("b"):
        bottom_pos = []
        cnt = 0
        while cnt <= cap.get(cv2.CAP_PROP_FPS):
            # カメラ映像の受け取り
            ret, frame = cap.read()

            # detetorによる顔の位置の推測
            dets = detector(frame[:, :, ::-1])
            if len(dets) > 0:
                # predictorによる顔のパーツの推測
                parts = predictor(frame, dets[0]).parts()
                # 映像の描画
                eye_pos = show_eye(frame, parts)
                key = cv2.waitKey(1)
                if eye_pos[0] != None:
                    bottom_pos.append(eye_pos[1])
                    cnt += 1
                
        y_max = sum(bottom_pos)/len(bottom_pos)
        print(f"y_max: {y_max}")
        winsound.Beep(440, 500) # 記録が終わったらビープ音を鳴らす 
        x_r,x_l,y_t,y_b = set_xy(x_max,x_min,y_max,y_min) # 端の設定
        
    # エスケープキーを押して終了します
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()