from EmoPy.src.fermodel import FERModel
import os

target = ['anger', 'happiness']  # 怒り、嬉しい
model = FERModel(target, verbose=True)
file = "core/data/image.jpg"
assert os.path.isfile(file)


# https://github.com/thoughtworksarts/EmoPy

def use_gpu():
    """
    gpuを使用する場合はgpuの指定のためにコードを実行する必要があります。
    :return: None
    """
    print("USING GPU")
    from keras.backend.tensorflow_backend import set_session
    import tensorflow as tf

    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            visible_device_list="0",  # specify GPU number
            allow_growth=True
        )
    )
    set_session(tf.Session(config=config))


def main():
    global file
    res, ps = model.predict(file)
    print(res, ps)
    return res, round(ps, 1)
