import Singleton
import faiss
import os
import numpy as np
import FeatureSolver
import cv2

@Singleton
class Gallery:
    def __init__(self):
        self.solver = FeatureSolver()
        self.__initGallery()

    # 通过cv2读取出来的img来搜索
    def search(self, img):
        query_feature = self.solver.getFeatureFromImg(img)
        dis, id = self.index_ip.search(query_feature, 3)
        if dis.any():
            if dis[0][0] >= 0.975:
                return int(id[0][0])
            else:
                return -1
        else:
            return -1

    # 通过文件来搜索其在特征数据库里的id
    def search(self, fileName):
        img = cv2.imread(fileName)
        return self.search(img)


    # 初始化特征数据全局库
    def __initGallery(self):
        dim = 2048
        self.index_ip = faiss.IndexFlatIP(dim)
        self.index_ip = faiss.IndexIDMap(self.index_ip)

        for root, dirList, fileList in os.walk("./data/gallery"):
            for fileName in fileList:
                id = int(fileName.split("_")[0])
                feature = self.solver.getFeature(os.path.join(root, fileName))
                self.index_ip.add_with_ids(feature, np.array((id,)).astype('int64'))