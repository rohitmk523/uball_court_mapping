import cv2

# Extract frame 100 from video
video_path = "GX020018.MP4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video")
    exit(1)

cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
ret, frame = cap.read()
cap.release()

if ret:
    cv2.imwrite("video_frame_100.jpg", frame)
    print(f"Frame 100 saved to: video_frame_100.jpg")
    print(f"Frame size: {frame.shape[1]}x{frame.shape[0]} pixels")
else:
    print("Error: Could not read frame 100")
