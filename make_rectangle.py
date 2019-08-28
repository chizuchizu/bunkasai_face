import random


def make(H, W, size=200):
    h = random.randint(0, W-size)
    w = random.randint(0, H-size)
    return [h, w, h + size, w + size]
