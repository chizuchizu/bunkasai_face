import cv2
import time
import random
import numpy as np
import os
from face_emotions import main

"""
/home/username/anaconda3/envs/deep36/lib/python3.6/site-packages/EmoPy/src/fermodel.py
60行目
        # self._print_prediction(prediction[0]) # past
        memo = self._print_prediction(prediction[0])  # add
        return memo  # add
60行目をコメントアウトし、後ろの2行を追加。
113行目に
        return [str(dominant_emotion), normalized_prediction[self.emotion_map[emotion]] * 100]
を追加。

実行結果が返ってくるように改良した。
"""


class Main:
    def __init__(self):
        if not os.path.isdir("image_data"):
            os.mkdir("image_data")
        self.ESC_KEY = 27
        self.INTERVAL = 33
        self.time_limit = 60

        self.odai_idx = [0, 1]
        self.odai_pred = ['anger', 'happiness']
        self.odai_list = ["Angry", "Happy"]
        self.odai_id = random.choice(self.odai_idx)

        self.ORG_WINDOW_NAME = "org"
        self.DEVICE_ID = 0

        # 顔の分類モデル(OpenCV)
        self.cascade_file = "haarcascade_frontalface_alt.xml"
        self.cascade = cv2.CascadeClassifier(self.cascade_file)

        # Web camera
        self.cap = cv2.VideoCapture(self.DEVICE_ID)

        # 初期フレームの読込
        self.end_flag, self.c_frame = self.cap.read()
        self.height, self.width, self.channels = self.c_frame.shape

        # ウィンドウの準備
        cv2.namedWindow(self.ORG_WINDOW_NAME)

        # flag
        self.face_flag = True

        self.rec = None
        # 対象の矩形のサイズ
        self.rectangle_size = 200

        self.time_1 = time.time()
        self.last = self.time_1

        # 1回目、お題が作られるときにカウントされるため-1にしている
        self.count = -1
        self.interval = 0

        self.score = 0

    def loop(self):
        ps = 40
        while self.end_flag:
            # 画像の取得と顔の検出
            self.score = round(self.score, 1)
            img = self.c_frame
            if self.face_flag or time.time() - self.last > 5:
                self.rec = self.make_rectangle()
                self.interval = time.time() - self.last
                self.last = time.time()
                if self.face_flag:
                    self.face_flag = False
                    self.count += 1
                """
                score計算
                
                (5 - インターバル（矩形が表示されてからクリアするまでの時間） * (表情の予測割合（%） - 40)
                5秒経ったら勝手に別のお題になる仕組みになっています。
                """
                self.score += round((5 - self.interval) * (ps - 40), 1)
                print(self.score)

            x, y, w, h = self.rec[0], self.rec[1], self.rec[2], self.rec[3]
            color = (0, 255, 255) if self.odai_id != 0 else (0, 240, 128)
            image_display, face_list = self.detection(img)

            cv2.rectangle(image_display, (x, y), (w, h), color, thickness=3)
            # print(face_list)
            if len(face_list) != 0:
                # print(x, y, w, h)
                if face_list[0][0] > x and face_list[0][1] > y and face_list[0][2] + face_list[0][0] < w \
                        and face_list[0][3] + face_list[0][1] < h:
                    """
                    顔の画像を切り取り、image_data/image.jpgに保存する
                    """
                    file = "image_data/image.jpg"
                    # print(np.array(img2).shape)
                    save = np.array(img)[face_list[0][1] + 5:face_list[0][1] + face_list[0][3] - 5,
                                         face_list[0][0] + 5:face_list[0][0] + face_list[0][2] - 5, :]
                    # print(save.shape)
                    # save
                    cv2.imwrite(file, save)

                    # predict emotion
                    predict, ps = main()
                    if self.odai_pred[self.odai_id] == predict:
                        self.face_flag = True
                        self.odai_id = random.choice(self.odai_idx)
                        # self.odai = self.odai_list[self.odai_id]

            image_display = cv2.flip(image_display, 1)

            cv2.putText(image_display,
                        "Time Limit{0: .2f}sec Count:{1} {2} SCORE:{3}".format(
                            self.time_limit - (time.time() - self.time_1), self.count,
                            self.odai_list[self.odai_id], self.score),
                        (20, 40),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX,
                        fontScale=0.6, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_4)

            # Frame
            cv2.imshow(self.ORG_WINDOW_NAME, image_display)

            # Escキーで終了 or Time Limit
            key = cv2.waitKey(self.INTERVAL)
            # print(time.time() - self.time_1)
            if key == self.ESC_KEY or time.time() - self.time_1 > self.time_limit:
                print("あなたの主観表情力は{}点でした！\nお疲れ様でした".format(self.score))
                break

            # 次のフレーム読み込み
            self.end_flag, self.c_frame = self.cap.read()
        print(self.score)

    def make_rectangle(self):
        h = random.randint(0, self.width - self.rectangle_size)
        w = random.randint(0, self.height - self.rectangle_size)
        return [h, w, h + self.rectangle_size, w + self.rectangle_size]

    def detection(self, img, color=(128, 0, 0)):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_list = self.cascade.detectMultiScale(img_gray, minSize=(100, 100))
        # print(face_list)
        # 検出した顔に印を付ける
        for (x, y, w, h) in face_list:
            pen_w = 3
            cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness=pen_w)

        return img, face_list


if __name__ == '__main__':
    main_loop = Main()
    main_loop.loop()
