from EmoPy.src.fermodel import FERModel

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

# https://github.com/thoughtworksarts/EmoPy

# TensorFlowをCPUで使用している場合は以下の7行(set_sessionまで)をコメントアウトして実行してください。
config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(
        visible_device_list="0",  # specify GPU number
        allow_growth=True
    )
)
set_session(tf.Session(config=config))

target_cmotions = ['anger', 'happiness']  # 怒り、嬉しい
model = FERModel(target_cmotions, verbose=True)
file = "image_data/image.jpg"


def main():
    frameString, ps = model.predict(file)
    # print(FERModel)
    print(frameString, ps)
    return frameString, round(ps, 1)

    # print(FERModel.predict(model, "a"))
    # print(type(frameString))
    # print(str(frameString or "0"))

