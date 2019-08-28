import cv2
import time
from face_detection import detection
from make_rectangle import make


class Main:
    def __init__(self):
        self.ESC_KEY = 27
        self.INTERVAL = 33
        self.time_limit = 10

        self.ORG_WINDOW_NAME = "org"
        self.DEVICE_ID = 0

        self.cascade_file = "haarcascade_frontalface_alt.xml"
        self.cascade = cv2.CascadeClassifier(self.cascade_file)

        self.cap = cv2.VideoCapture(self.DEVICE_ID)

        # 初期フレームの読込
        self.end_flag, self.c_frame = self.cap.read()
        self.height, self.width, self.channels = self.c_frame.shape

        # ウィンドウの準備
        cv2.namedWindow(self.ORG_WINDOW_NAME)

        # flag
        self.face_flag = True

        self.rec = make(self.height, self.width)

        self.time_1 = time.time()
        self.last = self.time_1

        self.count = -1
        self.interval = 0

    def loop(self):
        while self.end_flag:
            # 画像の取得と顔の検出
            img2 = self.c_frame
            if self.face_flag:
                self.rec = make(self.height, self.width)
                self.face_flag = False
                self.interval = time.time() - self.last
                self.last = time.time()
                self.count += 1

            x, y, w, h = self.rec[0], self.rec[1], self.rec[2], self.rec[3]
            img3, face_list = detection(img2, self.cascade)
            cv2.rectangle(img2, (x, y), (w, h), (0, 200, 225), thickness=3)

            cv2.putText(img3, "{0: .2f}sec {1}count".format(self.interval, self.count), (20, 60),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX,
                        fontScale=2, color=(0, 0, 0), thickness=6, lineType=cv2.LINE_4)

            # print(face_list)
            if len(face_list) != 0:
                # print(x, y, w, h)
                if face_list[0][0] > x and face_list[0][1] > y and face_list[0][2] + face_list[0][0] < w \
                        and face_list[0][3] + face_list[0][1] < h:
                    self.face_flag = True

            # Frame
            cv2.imshow(self.ORG_WINDOW_NAME, img3)

            # Escキーで終了 or Time Limit
            key = cv2.waitKey(self.INTERVAL)
            if key == self.ESC_KEY or time.time() - self.time_1 > self.time_limit:
                break

            # 次のフレーム読み込み
            self.end_flag, self.c_frame = self.cap.read()


if __name__ == '__main__':
    a = Main()
    a.loop()
