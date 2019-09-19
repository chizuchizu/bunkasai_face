import cv2
import time
import random
import numpy as np
import os
import argparse
from core.face_emotions import main

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
    def __init__(self, gpu=False):
        """
        1.Webカメラがあるかどうか（無いとエラー）
        2.名前を取得
        3.Window作成

        2と3を入れ替えるとWindowが引っ込んでしまう
        :param gpu: TF-gpuかcpuか
        """
        # gpuを使用するかしないか
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("-g", "--gpu", action="store_true", help="TF gpu版を使用しているかどうか")
        self.args = self.parser.parse_args()
        print(self.args.gpu)
        if gpu:
            """gpuを使用する場合はgpuの指定をしなければならないので"""
            from core.face_emotions import use_gpu
            use_gpu()

        self.ESC_KEY = 27
        self.INTERVAL = 33
        self.time_limit = 60  # seconds

        self.theme_idx = [0, 1]
        self.theme_pred = ['anger', 'happiness']
        self.theme_list = ["Angry", "Happy"]
        self.theme_id = random.choice(self.theme_idx)

        self.ORG_WINDOW_NAME = "org"
        self.DEVICE_ID = 0

        # 顔の分類モデル(OpenCV)
        # コマンドラインで動かす場合はここのパス（相対パス）を絶対パスにしてください
        self.cascade_file = "core/haarcascade_frontalface_alt.xml"
        # core/haarcascade_frontalface_alt.xml
        self.cascade = cv2.CascadeClassifier(self.cascade_file)

        # Web camera
        self.cap = cv2.VideoCapture(self.DEVICE_ID)

        # お名前
        self.name = input("あなたの名前を教えてください: ")

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

        # スコア計算については下のほうに
        self.score = 0

        self.image_path = "core/data/image.jpg"
        if not os.path.isdir("core/data"):
            os.mkdir("core/data")

    def loop(self):
        """
        mainloop
        :return: None
        """
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
            color = (0, 255, 255) if self.theme_id != 0 else (0, 240, 128)
            image_display, face_list = self.detection(img)

            cv2.rectangle(image_display, (x, y), (w, h), color, thickness=3)
            # print(face_list)
            if len(face_list) != 0:
                # print(x, y, w, h)
                if face_list[0][0] > x and face_list[0][1] > y and face_list[0][2] + face_list[0][0] < w \
                        and face_list[0][3] + face_list[0][1] < h:
                    """
                    顔の画像を切り取り、data/image.jpgに保存する
                    """
                    # file = "data/image.jpg"
                    # print(np.array(img2).shape)
                    save = np.array(img)[face_list[0][1] + 5:face_list[0][1] + face_list[0][3] - 5,
                           face_list[0][0] + 5:face_list[0][0] + face_list[0][2] - 5, :]
                    # print(save.shape)
                    # save
                    cv2.imwrite(self.image_path, save)

                    # predict emotion
                    predict, ps = main()
                    if self.theme_pred[self.theme_id] == predict:
                        self.face_flag = True
                        self.theme_id = random.choice(self.theme_idx)

            image_display = cv2.flip(image_display, 1)

            """
            上に表示されるバー
            TimeLimit: float: 残り時間
            Count: int: 正解数
            SCORE: float: スコア（点数は上に計算式が載っている）
            """
            cv2.putText(image_display,
                        "Time Limit{0: .2f}sec Count:{1} {2} SCORE:{3}".format(
                            self.time_limit - (time.time() - self.time_1), self.count,
                            self.theme_list[self.theme_id], self.score),
                        (20, 40),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX,
                        fontScale=0.6, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_4)

            # Frame
            cv2.imshow(self.ORG_WINDOW_NAME, image_display)

            # Escキーで終了 or Time Limit
            key = cv2.waitKey(self.INTERVAL)
            # print(time.time() - self.time_1)
            if key == self.ESC_KEY or time.time() - self.time_1 > self.time_limit:
                self.make_ranking()
                break

            # 次のフレーム読み込み
            self.end_flag, self.c_frame = self.cap.read()
        # print(self.score)

    def tutorial(self):
        """未定"""
        return 0

    def make_rectangle(self):
        """
        ランダムに矩形を生成
        :return: list: 矩形の4点の座標
        """
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

    def make_ranking(self):
        ranking_path = "core/data/ranking.npy"
        result = np.array([[self.name, self.score]])

        if not os.path.isfile(ranking_path):
            """Fileが存在しなければ作成"""
            np.save(ranking_path, result)
        else:
            ranking = np.load(ranking_path)
            result = np.array([[self.name, self.score]])
            ranking = np.concatenate([ranking, result])
            # ranking = np.append(ranking, result, axis=0)
            ranking = ranking[np.argsort(ranking[:, 1])[::-1]]
            # print(ranking)
            np.save(ranking_path, ranking)
        print("{0}さんの瞬間表情力は{1}点でした！\nお疲れ様でした".format(self.name, self.score))


if __name__ == "__main__":
    app = Main(gpu=False)
    app.loop()
