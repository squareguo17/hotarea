from config import getConfig
from predictor import FeatureExtractionDemo
import cv2

if __name__ == "__main__":
    cfg =  getConfig()
    engine = FeatureExtractionDemo(cfg, parallel=True)

    img = cv2.imread("./data/gallery/0_0.jpg")
    feature = engine.run_on_image(img)
    print("Hello")