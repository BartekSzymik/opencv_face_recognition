from cv2 import cv2
import numpy as np
import math

# Mouth dimensions
MOUTH_WIDTH = 150
MOUTH_HEIGHT = 20

# Load image
img = cv2.imread('assets/face.jpg', cv2.IMREAD_GRAYSCALE)

# Convert to BGR image
gray = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# Initialize face detector
face_cascade = cv2.CascadeClassifier('venv/lib64/python3.11/site-packages/cv2/data/haarcascade_frontalface_default.xml')

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# Initialize algorithm for facial landmark detection
protoFile = "assets/deploy.prototxt"
weightsFile = "assets/res10_300x300_ssd_iter_140000_fp16.caffemodel"
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

# Iterate over detect face
for (x, y, w, h) in faces:
    # Define the region of interest (ROI) containing the face
    face_roi = gray[y:y + h, x:x + w]

    # Get the dimensions of the ROI
    roi_h, roi_w = face_roi.shape[:2]

    # Use neural network to predict facial landmarks
    blob = cv2.dnn.blobFromImage(face_roi, 1.0, (300, 300), [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()

    # Select the first (usually the best) prediction and get the facial landmarks
    detection = detections[0, 0, :, :]
    points = np.zeros((68, 2), dtype=int)
    for i in range(68):
        x_i = int(detection[i * 2][0] * w)
        y_i = int(detection[i * 2 + 1][0] * h)
        points[i] = (x_i, y_i)

    # Assign the coordinates of the mouth
    mouth_top = points[62]
    mouth_bottom = points[66]
    mouth_left = points[60]
    mouth_right = points[64]

    # Get the mouth points
    mouth_points = np.array([(mouth_left[0] - x, mouth_left[1] - y),
                             (mouth_top[0] - x, mouth_top[1] - y),
                             (mouth_right[0] - x, mouth_right[1] - y),
                             (mouth_bottom[0] - x, mouth_bottom[1] - y)], dtype=np.int32)

    # Display the image with facial landmarks
    cv2.imshow('Facial Landmarks Detection', img)

    # Calculate the mouth position
    MOUTH_SPEED = 0.3
    MOUTH_AMPLITUDE = 10
    t = 0
    while True:
        # Calculate the mouth position
        x = MOUTH_WIDTH / 2
        y = MOUTH_HEIGHT / 2 + MOUTH_AMPLITUDE * math.sin(MOUTH_SPEED * t)

        # Generate the mouth animation image
        mouth_img = np.zeros((roi_h, roi_w, 3), dtype=np.uint8)
        cv2.rectangle(mouth_img, (int(x), int(y)), (int(x + MOUTH_WIDTH), int(y + MOUTH_HEIGHT)), (0, 0, 255), -1)

        # Apply the mouth animation to the ROI
        mask = np.zeros_like(face_roi)
        cv2.fillConvexPoly(mask, (mouth_points - (x, y)).astype('int32'), (255, 255, 255))

        masked_mouth = cv2.bitwise_and(mouth_img, mask)
        roi_with_mouth = cv2.add(face_roi, masked_mouth)

        # Assignment of modified ROI to the original image
        img[int(y):int(y + roi_h), int(x):int(x + roi_w)] = roi_with_mouth[:, :, 0]

        #  Display the result
        cv2.imshow('Mouth Animation', img)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        # Increase the animation time
        t += 0.5

    cv2.destroyAllWindows()
