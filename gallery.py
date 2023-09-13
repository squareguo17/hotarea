import Singleton
import faiss
import os
import numpy as np
from featuresolver import FeatSolver
import cv2
map = {0:"XY", 1:"XD", 2:"YH", 3:"WG"}
color = {0:(255,0,0), 1:(0,255,0), 2:(0,0,255), 3:(255,255,0)}
class Gallery:
    def __init__(self):
        self.solver = FeatSolver()
        self.__initGallery()

    # 通过cv2读取出来的img来搜索
    def searchImg(self, img):
        query_feature = self.solver.getFeatFromImg(img)
        dis, id = self.index_ip.search(query_feature, 3)
        # print(str(dis) + str(id))
        if dis.any():
            if dis[0][0] >= 0.98:
                return map.get(int(id[0][0])), color.get(int(id[0][0]))
            else:
                return None, None
        else:
            return None, None

    # 通过文件来搜索其在特征数据库里的id
    def searchFile(self, fileName):
        img = cv2.imread(fileName)
        return self.searchImg(img)

    # 初始化特征数据全局库
    def __initGallery(self):
        dim = 2048
        self.index_ip = faiss.IndexFlatIP(dim)
        self.index_ip = faiss.IndexIDMap(self.index_ip)

        for root, dirList, fileList in os.walk("./data/gallery"):
            for fileName in fileList:
                id = int(fileName.split("_")[0])
                feature = self.solver.getFeatFromFile(os.path.join(root, fileName))
                self.index_ip.add_with_ids(feature, np.array((id,)).astype('int64'))