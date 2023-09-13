import Singleton
import cv2
import torch.nn.functional as F
from fastreid.engine import DefaultPredictor
from config import getConfig
import torch

# 特征提取器是全局唯一的, 所以设置成单例模式, 保证全局访问的都是同一个engine
class FeatSolver:
    def __init__(self):
        self.index_ip = None
        self.engine = DefaultPredictor(getConfig())

    # 通过cv读出来的img获取特征值
    def getFeatFromImg(self, img):
        # the model expects RGB inputs
        img = img[:, :, ::-1]
        # Apply pre-processing to image.
        img = cv2.resize(img, tuple(getConfig().INPUT.SIZE_TEST[::-1]), interpolation=cv2.INTER_CUBIC)
        # Make shape with a new batch dimension which is adapted for
        # network input
        np_img = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))[None]

        feature = self.engine(np_img)
        feature = self.__postprocess(feature)
        return feature

    # 通过文件路径获取特征值
    def getFeatFromFile(self, fileName):
        img = cv2.imread(fileName)
        return self.getFeatFromImg(img)

    # 特征后处理, 这个方法对外是不暴露的, 所以弄成private函数. 对外暴露的只有getFeatFromImg方法
    def __postprocess(self, features):
        # Normalize feature to compute cosine distance
        features = F.normalize(features)
        features = features.cpu().data.numpy()
        return features