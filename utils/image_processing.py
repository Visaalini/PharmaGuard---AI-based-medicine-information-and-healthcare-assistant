import cv2

def preprocess_image(image_path):

    img = cv2.imread(image_path)
    img = cv2.resize(img, (224,224))

    return img