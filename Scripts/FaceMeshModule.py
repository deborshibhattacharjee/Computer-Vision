import cv2
import mediapipe as mp
import time


class faceMeshDetector():
    def __init__(self, mode=False, num_faces=1, detectionCon=0.5,
                 trackCon=0.5):

        self.mode = mode
        self.num_faces = num_faces
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.mode, self.num_faces,
                                                 self.detectionCon,
                                                 self.trackCon)
        self.drawSpecs = self.mpDraw.DrawingSpec((0, 255, 0),
                                                 thickness=1,
                                                 circle_radius=1)

    def findFaceMesh(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.faceMesh.process(imgRGB)

        faces = []
        if results.multi_face_landmarks:
            for faceLm in results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLm, self.mpFaceMesh.FACEMESH_CONTOURS,
                                           self.drawSpecs, self.drawSpecs)
            face = []
            for ID, lm in enumerate(faceLm.landmark):

                ih, iw, ic = img.shape
                x, y = int(lm.x*iw), int(lm.y*ih)
                #cv2.putText(img, str(ID), (x, y), cv2.FONT_HERSHEY_PLAIN,
                            #0.5, (0, 0, 255), 1)

                face.append([x, y])
            faces.append(face)
        return img, faces


def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = faceMeshDetector()
    while True:
        success, img = cap.read()
        img, faces = detector.findFaceMesh(img)

        if faces != 0:
            print(len(faces))

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f'fps:{int(fps)}', (5, 40), cv2.FONT_HERSHEY_PLAIN,
                    2, (0, 255, 0), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()