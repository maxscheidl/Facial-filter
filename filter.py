import cv2


def in_range(x, y, overlay, image):

    img_width = image.shape[1]
    img_height = image.shape[0]
    overlay_width = overlay.shape[1]
    overlay_height = overlay.shape[0]

    if (x + overlay_width) >= img_width or x <= 0:
        return False

    if (y + overlay_height) >= img_height or y <= 0:
        return False

    return True


class Filter:

    def __init__(self):
        pass

    def draw_glasses(self, image, glasses, landmarks):

        scale = glasses.shape[0] / glasses.shape[1]

        glasses_width = (landmarks[446].x - landmarks[226].x) * image.shape[1] * 1.7
        glasses_resized = cv2.resize(glasses, (int(glasses_width), int(glasses_width * scale)))

        rows, cols, _ = glasses_resized.shape

        x = int(landmarks[6].x * image.shape[1] - cols/2)
        y = int(landmarks[6].y * image.shape[0] - rows/2)

        if in_range(x, y, glasses_resized, image):

            glasses_gray = cv2.cvtColor(glasses_resized, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(glasses_gray, 65, 255, cv2.THRESH_BINARY)

            area = image[y:y+rows, x:x+cols]
            image[y:y+rows, x:x+cols] = cv2.bitwise_and(area, area, mask=mask)
            #image[y:y+rows, x:x+cols] = cv2.add(new_area, glasses_resized)

        return image

    def draw_mouth(self, image, mouth, landmarks):

        scale = mouth.shape[0] / mouth.shape[1]

        mouth_width = (landmarks[432].x - landmarks[57].x) * image.shape[1] * 1.2
        mouth_resized = cv2.resize(mouth, (int(mouth_width), int(mouth_width * scale)))

        rows, cols, _ = mouth_resized.shape

        x = int(landmarks[0].x * image.shape[1] - cols/2)
        y = int(landmarks[0].y * image.shape[0] - rows/4)

        if in_range(x, y, mouth_resized, image):

            mouth_gray = cv2.cvtColor(mouth_resized, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(mouth_gray, 25, 255, cv2.THRESH_BINARY_INV)

            area = image[y:y+rows, x:x+cols]
            new_area = cv2.bitwise_and(area, area, mask=mask)
            image[y:y+rows, x:x+cols] = cv2.add(new_area, mouth_resized)

        return image