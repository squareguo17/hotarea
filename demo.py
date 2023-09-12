# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import argparse
import glob
import os
import sys

import faiss
import torch.nn.functional as F
import cv2
import numpy as np
import tqdm
from torch.backends import cudnn

sys.path.append('.')

from fastreid.config import get_cfg
from fastreid.utils.logger import setup_logger
from fastreid.utils.file_io import PathManager

from predictor import FeatureExtractionDemo

from ultralytics import YOLO


# import some modules added in project like this below
# sys.path.append("projects/PartialReID")
# from partialreid import *

cudnn.benchmark = True
setup_logger(name="fastreid")


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # add_partialreid_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Feature extraction with reid models")
    parser.add_argument(
        "--config-file",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--parallel",
        action='store_true',
        help='If use multiprocess for feature extraction.'
    )
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
             "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        default='demo_output',
        help='path to save features'
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


class Solver:
    def postprocess(self, features):
        # Normalize feature to compute cosine distance
        features = F.normalize(features)
        features = features.cpu().data.numpy()
        return features

    def __init__(self):
        self.index_ip = None
        args = get_parser().parse_args()
        cfg = setup_cfg(args)
        self.engine = FeatureExtractionDemo(cfg, parallel=args.parallel)

        self.initGallery()

    def initGallery(self):
        dim = 2048
        self.index_ip = faiss.IndexFlatIP(dim)
        self.index_ip = faiss.IndexIDMap(self.index_ip)

        for root, dirList, fileList in os.walk("./data/gallery"):
            for fileName in fileList:
                id = int(fileName.split("_")[0])
                feature = self.getFeature(os.path.join(root, fileName))
                self.index_ip.add_with_ids(feature, np.array((id,)).astype('int64'))

    def getFeature(self, fileName):
        img = cv2.imread(fileName)
        return self.getFeatureFromImg(img)

    def getFeatureFromImg(self, img):
        feature = self.engine.run_on_image(img)
        feature = self.postprocess(feature)
        return feature

    def search(self, img):
        query_feature = self.getFeatureFromImg(img)
        dis, id = self.index_ip.search(query_feature, 3)
        if dis.any():
            if dis[0][0] >= 0.975:
                return int(id[0][0])
            else:
                return -1
        else:
            return -1

    def searchFile(self, fileName):
        img = cv2.imread(fileName)
        feature = self.getFeature(fileName)
        dis, id = self.index_ip.search(feature, 5)
        print(str(dis), str(id))

class VideoGenerator:
    def __init__(self, fourcc, size):
        fps = 30  # 视频帧率
        self.video = cv2.VideoWriter("./data/result/output.avi", fourcc, fps, size)

    def appendImg(self, img):
        self.video.write(img)

    def release(self):
        cv2.destroyAllWindows()
        self.video.release()

if __name__ == '__main__':
    print("Begin solve")
    solver = Solver()
    video_capture = cv2.VideoCapture("./test.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    videoWidth = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    videoHeight = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_write = VideoGenerator(fourcc, (videoWidth, videoHeight))

    # solver.searchFile("./data/gallery/0_0.jpg")
    # solver.searchFile("./data/gallery/0_4.jpg")
    # solver.searchFile("./data/gallery/1_5.jpg")
    # solver.searchFile("./data/query/2_0.jpg")
    step = 0
    while True:
        flag, frame = video_capture.read()
        if not flag:
            break

        model = YOLO("./model/yolov8n.pt")
        results = model(frame)
        boxes = results[0].boxes.cpu()
        # cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        for box in boxes:
            if box.cls == 0:
                xyxy = box.xyxy.numpy()[0]
                startX = int(xyxy[0])
                endX = int(xyxy[2])
                startY = int(xyxy[1])
                endY = int(xyxy[3])
                # singleImg = frame[startY : endY, startX : endX]
                cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
        video_write.appendImg(frame)
        print(str(step))
        if step == 100:
            break
        step = step  + 1
                # id = solver.search(singleImg)
                # if id >= 0:
                #     color = (255, 0, 0)
                #     if id == 0:
                #         color = (255, 0 , 0)
                #     elif id == 1:
                #         color = (0, 255, 0)
                #     else:
                #         color = (0, 0, 255)
                #     cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                #     result_file_name = "./data/result/" + str(step) + "_" + str(id) + ".jpg"
                #     cv2.imwrite(result_file_name, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                #     step = step + 1
                #     if step == 20:
                #         break
    video_write.release()