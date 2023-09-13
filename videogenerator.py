import cv2

class VideoGenerator:
    def __init__(self, size):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')

        fps = 30  # 视频帧率
        self.video_write = cv2.VideoWriter("./data/result/output.avi", fourcc, fps, size)

    def append(self, img):
        self.video_write.write(img)

    def release(self):
        self.video_write.release()