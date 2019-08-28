import cv2


def detection(img, model, color=(128, 0, 0)):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_list = model.detectMultiScale(img_gray, minSize=(100, 100))
    # print(face_list)
    # 検出した顔に印を付ける
    for (x, y, w, h) in face_list:
        pen_w = 3
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness=pen_w)

    return img, face_list
