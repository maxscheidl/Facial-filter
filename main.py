import cv2
from faceDetector import FaceDetector
from filter import Filter


def main():

    cap = cv2.VideoCapture(0)
    detector = FaceDetector()
    filter = Filter()

    glasses = cv2.imread('glasses.jpg')
    mouth = cv2.imread('mouth.png')

    while cap.isOpened():

        success, image = cap.read()

        if not success:
            print("Ignoring empty camera frame.")
            continue

        height, width, _ = image.shape

        results, image = detector.detect_landmarks(cv2.flip(image, 1), False)

        if results:
            for landmarks in results:
                for id, landmark in enumerate(landmarks.landmark):

                    #cv2.putText(image, str(id), (int(landmark.x * width), int(landmark.y * height)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                    if id == 0 or id == 432 or id == 57:
                        pass
                        #cv2.circle(image, (int(landmark.x * width), int(landmark.y * height)), 5, (255, 50, 38), cv2.FILLED)

                image = filter.draw_glasses(image, glasses, landmarks.landmark)
                image = filter.draw_mouth(image, mouth, landmarks.landmark)

        cv2.imshow('Image', image)

        key = cv2.waitKey(5)
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
