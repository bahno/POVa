import cv2


def perform_segmentation(frame, prev_frame):
    cv2.imshow("current frame", frame)
    if prev_frame:
        cv2.imshow("prev frame", prev_frame)
    return frame
