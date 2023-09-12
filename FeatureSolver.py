from config import getConfig
from predictor import FeatureExtractionDemo
import Singleton
import cv2
import torch.nn.functional as F

# 特征提取器是全局唯一的, 所以设置成单例模式, 保证全局访问的都是同一个engine
@Singleton
class FeatureSolver:
    def __init__(self):
        self.index_ip = None
        self.engine = FeatureExtractionDemo(getConfig(), parallel=True)

    # 通过cv读出来的img获取特征值
    def getFeatFromImg(self, img):
        feature = self.engine.run_on_image(img)
        feature = self.postprocess(feature)
        return feature

    # 通过文件路径获取特征值
    def getFeatFromFile(self, fileName):
        img = cv2.imread(fileName)
        return self.getFeatureFromImg(img)

    # 特征后处理, 这个方法对外是不暴露的, 所以弄成private函数. 对外暴露的只有getFeatFromImg方法
    def __postprocess(self, features):
        # Normalize feature to compute cosine distance
        features = F.normalize(features)
        features = features.cpu().data.numpy()
        return features