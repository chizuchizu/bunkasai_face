from EmoPy.src.fermodel import FERModel
from pkg_resources import resource_filename

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(
        visible_device_list="0",  # specify GPU number
        allow_growth=True
    )
)
set_session(tf.Session(config=config))

target_cmotions = ["calm", "anger", "happiness"]
model = FERModel(target_cmotions, verbose=True)
file = "image_data/image.jpg"


def main():
    frameString = model.predict(file)
    print(frameString)
