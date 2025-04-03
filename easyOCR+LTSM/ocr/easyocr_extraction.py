import cv2
import easyocr

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
    img = cv2.resize(img, (800, 800))  # Resize
    _, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)  # Binarize (increase contrast)
    return img

def extract_text_from_image(image_path):
    reader = easyocr.Reader(['en'])
    preprocessed_img = preprocess_image(image_path)
    return reader.readtext(preprocessed_img, detail=0)
