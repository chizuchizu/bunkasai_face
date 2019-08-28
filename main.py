import cv2
import time
from face_detection import detection
from make_rectangle import make

if __name__ == '__main__':
    # 定数定義
    ESC_KEY = 27  # Escキー
    INTERVAL = 33  # 待ち時間
    FRAME_RATE = 1  # fps
    time_limit = 30  # limit

    ORG_WINDOW_NAME = "org"
    GAUSSIAN_WINDOW_NAME = "gaussian"

    DEVICE_ID = 0

    # 分類器の指定
    cascade_file = "haarcascade_frontalface_alt.xml"
    cascade = cv2.CascadeClassifier(cascade_file)

    # カメラ映像取得
    cap = cv2.VideoCapture(DEVICE_ID)

    # 初期フレームの読込
    end_flag, c_frame = cap.read()
    height, width, channels = c_frame.shape

    # ウィンドウの準備
    cv2.namedWindow(ORG_WINDOW_NAME)

    face_flag = True

    # weight, height
    W = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    H = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # cv2.namedWindow(GAUSSIAN_WINDOW_NAME)

    rec = make(H, W)

    color = (200, 200, 200)
    memo = 3

    time_1 = time.time()
    a = time_1

    count = b = 0

    # 変換処理ループ
    while end_flag:
        # 画像の取得と顔の検出
        img = c_frame
        img2 = c_frame
        if face_flag:
            rec = make(H, W)
            face_flag = False
            b = time.time() - a
            a = time.time()
            count += 1

        x, y, w, h = rec[0], rec[1], rec[2], rec[3]
        color = (0, 200, 225)
        pen_w = 3
        cv2.rectangle(img2, (x, y), (w, h), color, thickness=pen_w)

        cv2.putText(img2, "{0: .2f}sec  {1}count".format(b, count), (20, 60), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=2,
                    color=(0, 0, 0), thickness=6)

        img3, face_list = detection(img2, cascade)
        # print(face_list)
        if len(face_list) != 0:
            # print(x, y, w, h)
            if face_list[0][0] > x and face_list[0][1] > y and face_list[0][2] + face_list[0][0] < w \
                    and face_list[0][3] + face_list[0][1] < h:
                face_flag = True

        # Frame
        cv2.imshow(ORG_WINDOW_NAME, img3)

        # Escキーで終了 or Time Limit
        key = cv2.waitKey(INTERVAL)
        if key == ESC_KEY or time.time() - time_1 > time_limit:
            break

        # 次のフレーム読み込み
        end_flag, c_frame = cap.read()

    # 終了処理
    cv2.destroyAllWindows()
    cap.release()
