from featuresolver import FeatSolver
from gallery import Gallery
from videogenerator import VideoGenerator
from ultralytics import YOLO
import cv2
if __name__ == '__main__':
    gallery = Gallery()

    video_capture = cv2.VideoCapture("./data/3MIN.mp4")
    videoWidth = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    videoHeight = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_output = VideoGenerator((videoWidth, videoHeight))

    total_frame_cnt = video_capture.get(7)
    step = 0
    while True:
        flag, frame = video_capture.read()
        if not flag:
            break

        model = YOLO("./models/yolov8n.pt")
        results = model(frame)
        boxes = results[0].boxes.cpu()
        for box in boxes:
            if box.cls == 0:
                xyxy = box.xyxy.numpy()[0]
                startX = int(xyxy[0])
                endX = int(xyxy[2])
                startY = int(xyxy[1])
                endY = int(xyxy[3])
                singleImg = frame[startY : endY, startX : endX]
                person_id, color = gallery.searchImg(singleImg)
                if person_id is None:
                    continue
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                cv2.putText(frame, person_id, (startX, startY), 2, 2, color, 2)
        video_output.append(frame)
        print(str(step) + "/" + str(total_frame_cnt))
        step = step + 1