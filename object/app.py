import cv2
import numpy as np

# Start webcam capture (0 is the default camera)
cap = cv2.VideoCapture(0)

# Check if the camera is accessible
if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

# Set the resolution of the video to reduce processing load
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set width to 640
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set height to 480

# Initialize the background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If frame is not read properly, break the loop
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Apply background subtraction to get the foreground mask
    fgmask = fgbg.apply(frame)

    # Optionally, apply some morphological operations to clean up the mask
    fgmask = cv2.dilate(fgmask, None, iterations=2)  # Dilation to fill in holes
    fgmask = cv2.erode(fgmask, None, iterations=1)   # Erosion to remove noise

    # Find contours in the foreground mask
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop over each contour found
    for contour in contours:
        # Ignore small contours to avoid detecting noise or small movements
        if cv2.contourArea(contour) < 500:  # You can adjust this threshold
            continue
        
        # Get the bounding box for the contour
        (x, y, w, h) = cv2.boundingRect(contour)
        
        # Draw a bounding box around the detected object
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the resulting frame with detected objects
    cv2.imshow('Object Detection - Live', frame)

    # Exit the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close any open windows
cap.release()
cv2.destroyAllWindows()
