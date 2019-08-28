import cv2
import numpy as np
from PIL import ImageDraw, Image, ImageFont


class PutText:
    def __init__(self):
        pass

    @classmethod
    def puttext(selv, cls, cv_image, text, point, font_size, color=(0, 0, 0)):
        font = ImageFont.truetype()