import cv2
import segmentation

# Replace 'video_file_path' with the path to your video file
video_file_path = '../data/travolta.gif'

# Open the video file
cap = cv2.VideoCapture(video_file_path)

# Check if the video file is opened successfully
if not cap.isOpened():
    print("Error opening video file")
    exit()

prev_frame = None

# Read and display frames until the video is over
while True:
    # Read a frame from the video
    ret, frame = cap.read()

    """
    if not prev_frame:
        #TODO manually create initial mask for the first processed frame
        pass
    """

    # Check if the frame is read successfully
    if not ret:
        print("End of video")
        break

    display_frame = segmentation.perform_segmentation(frame, prev_frame)

    # Display the frame
    cv2.imshow('Original gif', display_frame)

    prev_frame = display_frame

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(60) & 0xFF == ord('q'):
        break

# Release the video capture object and close the window
cap.release()
cv2.destroyAllWindows()
