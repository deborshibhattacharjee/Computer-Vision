import cv2
import mediapipe as mp
import time


class faceDetector():
    def __init__(self, minDetectionCon=0.5):
        self.minDetectionCon = minDetectionCon
        self.mpFace = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetect = self.mpFace.FaceDetection(self.minDetectionCon)

    def findFaces(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.faceDetect.process(imgRGB)

        bboxList = []

        if results.detections:
            for ID, detection in enumerate(results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                bboxList.append([bbox, detection.score])

                if draw:
                    img = self.advanceDraw(img, bbox)
                    cv2.putText(img, f': {int(detection.score[0]*100)}%', (bbox[0], bbox[1]-20),
                                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)

        return img, bboxList

    @staticmethod
    def advanceDraw(img, bbox, length=20, t=3, rt=1):
        x, y, w, h = bbox
        x1, y1 = x+w, y+h

        cv2.rectangle(img, bbox, (255, 0, 255), rt)
        # Top left x, y
        cv2.line(img, (x, y), (x+length, y), (255, 0, 255), t)
        cv2.line(img, (x, y), (x, y+length), (255, 0, 255), t)

        # Top Right x1, y
        cv2.line(img, (x1, y), (x1 - length, y), (255, 0, 255), t)
        cv2.line(img, (x1, y), (x1, y + length), (255, 0, 255), t)

        # Bottom left x, y1
        cv2.line(img, (x, y1), (x + length, y1), (255, 0, 255), t)
        cv2.line(img, (x, y1), (x, y1 - length), (255, 0, 255), t)

        # Bottom right x1, y1
        cv2.line(img, (x1, y1), (x1 - length, y1), (255, 0, 255), t)
        cv2.line(img, (x1, y1), (x1, y1 - length), (255, 0, 255), t)

        return img


def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = faceDetector()
    while True:
        success, img = cap.read()

        img, bbox = detector.findFaces(img)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                    3, (255, 0, 0), 2)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()