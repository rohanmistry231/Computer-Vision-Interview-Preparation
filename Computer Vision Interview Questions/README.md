# OpenCV (cv2) Interview Questions for AI/ML Roles

## Core OpenCV Foundations

### Image Basics

#### Basic
1. **What is an image in OpenCV, and how is it represented? Provide an example of loading an image.**  
   An image in OpenCV is a NumPy array where each pixel is represented by intensity values (grayscale) or RGB/BGR channels (color). Used for all computer vision tasks like object detection.  
   ```python
   import cv2
   img = cv2.imread('image.jpg')  # Load image as BGR
   ```

2. **How do you access and modify pixel values in an OpenCV image? Give an example.**  
   Pixel values are accessed via NumPy array indexing, useful for image preprocessing (e.g., adjusting brightness).  
   ```python
   import cv2
   img = cv2.imread('image.jpg')
   pixel = img[100, 100]  # Access BGR at (100, 100)
   img[100, 100] = [0, 255, 0]  # Set to green
   ```

3. **What is the difference between grayscale and color images in OpenCV?**  
   Grayscale images are single-channel (intensity), while color images (BGR in OpenCV) have three channels. Grayscale is used for simpler tasks like edge detection.  
   ```python
   import cv2
   img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)  # Grayscale
   img_color = cv2.imread('image.jpg')  # BGR
   ```

4. **Explain how to save an image in OpenCV.**  
   Saving images preserves processed results (e.g., filtered images) for further analysis or model input.  
   ```python
   import cv2
   img = cv2.imread('image.jpg')
   cv2.imwrite('output.jpg', img)
   ```

#### Intermediate
5. **How do you convert an image between color spaces (e.g., BGR to grayscale or HSV)? Provide an example.**  
   Color space conversion is critical for tasks like color-based segmentation (HSV) or simplifying processing (grayscale).  
   ```python
   import cv2
   img = cv2.imread('image.jpg')
   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
   ```

6. **Write a function to resize an image in OpenCV while maintaining aspect ratio.**  
   Resizing is essential for preparing images for neural networks with fixed input sizes.  
   ```python
   import cv2
   def resize_image(img, width):
       ratio = width / img.shape[1]
       height = int(img.shape[0] * ratio)
       return cv2.resize(img, (width, height))
   img = cv2.imread('image.jpg')
   resized = resize_image(img, 300)
   ```

7. **How do you crop an image in OpenCV? Give an example.**  
   Cropping isolates regions of interest (ROI) for focused analysis, like face detection.  
   ```python
   import cv2
   img = cv2.imread('image.jpg')
   cropped = img[50:150, 100:200]  # Crop (y1:y2, x1:x2)
   ```

#### Advanced
8. **Explain image channels and how to split and merge them in OpenCV.**  
   Channels (e.g., B, G, R) represent color components, split for individual analysis or merged to reconstruct images.  
   ```python
   import cv2
   img = cv2.imread('image.jpg')
   b, g, r = cv2.split(img)  # Split channels
   merged = cv2.merge([b, g, r])  # Merge back
   ```

9. **Write a function to normalize pixel values in an image.**  
   Normalization scales pixel values (e.g., to [0, 1]) for consistent input to ML models.  
   ```python
   import cv2
   import numpy as np
   def normalize_image(img):
       return img / 255.0
   img = cv2.imread('image.jpg')
   normalized = normalize_image(img)
   ```

10. **Implement a function to rotate an image without cropping using OpenCV.**  
    Rotation is used in data augmentation for training robust vision models.  
    ```python
    import cv2
    import numpy as np
    def rotate_image(img, angle):
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        return cv2.warpAffine(img, M, (new_w, new_h))
    img = cv2.imread('image.jpg')
    rotated = rotate_image(img, 45)
    ```

### Image Processing

#### Basic
11. **What is image thresholding in OpenCV? Provide an example.**  
    Thresholding converts images to binary (e.g., for segmenting objects).  
    ```python
    import cv2
    img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    ```

12. **How do you apply a Gaussian blur to an image in OpenCV? Give an example.**  
    Gaussian blur reduces noise, aiding feature detection.  
    ```python
    import cv2
    img = cv2.imread('image.jpg')
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    ```

13. **What is the purpose of the `cv2.imshow` function, and how is it used?**  
    Displays images for debugging or visualization during development.  
    ```python
    import cv2
    img = cv2.imread('image.jpg')
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ```

#### Intermediate
14. **Explain adaptive thresholding and provide an example.**  
    Adaptive thresholding adjusts thresholds locally, ideal for uneven lighting in images.  
    ```python
    import cv2
    img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
    adaptive = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    ```

15. **Write a function to apply a sharpening filter to an image.**  
    Sharpening enhances edges for better feature detection.  
    ```python
    import cv2
    import numpy as np
    def sharpen_image(img):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(img, -1, kernel)
    img = cv2.imread('image.jpg')
    sharpened = sharpen_image(img)
    ```

16. **How do you perform histogram equalization in OpenCV? Provide an example.**  
    Histogram equalization enhances contrast, useful for low-light images.  
    ```python
    import cv2
    img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
    equalized = cv2.equalizeHist(img)
    ```

#### Advanced
17. **Implement a function to apply a custom convolution kernel.**  
    Custom kernels enable specialized filtering (e.g., edge detection).  
    ```python
    import cv2
    import numpy as np
    def apply_kernel(img, kernel):
        return cv2.filter2D(img, -1, kernel)
    img = cv2.imread('image.jpg')
    kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    filtered = apply_kernel(img, kernel)
    ```

18. **Write a function to perform morphological operations (e.g., dilation).**  
    Morphological operations like dilation enhance shapes for segmentation.  
    ```python
    import cv2
    import numpy as np
    def dilate_image(img, kernel_size=3):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.dilate(img, kernel, iterations=1)
    img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
    dilated = dilate_image(img)
    ```

19. **Explain how to handle image noise using median filtering in OpenCV.**  
    Median filtering removes salt-and-pepper noise while preserving edges.  
    ```python
    import cv2
    img = cv2.imread('image.jpg')
    denoised = cv2.medianBlur(img, 5)
    ```

### Feature Detection

#### Basic
20. **What is edge detection in OpenCV, and how is it performed using Canny?**  
    Edge detection identifies boundaries, crucial for object recognition.  
    ```python
    import cv2
    img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(img, 100, 200)
    ```

21. **How do you detect corners in an image using OpenCV? Give an example.**  
    Corner detection (e.g., Harris) identifies keypoints for tracking or matching.  
    ```python
    import cv2
    import numpy as np
    img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
    corners = cv2.cornerHarris(img, 2, 3, 0.04)
    ```

22. **What are contours in OpenCV, and how do you find them?**  
    Contours are curves joining continuous points, used for shape analysis.  
    ```python
    import cv2
    img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ```

#### Intermediate
23. **Write a function to detect and draw contours in an image.**  
    Visualizes object boundaries for segmentation tasks.  
    ```python
    import cv2
    def draw_contours(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return cv2.drawContours(img.copy(), contours, -1, (0, 255, 0), 2)
    img = cv2.imread('image.jpg')
    contoured = draw_contours(img)
    ```

24. **How do you use SIFT for feature detection in OpenCV? Provide an example.**  
    SIFT detects scale-invariant keypoints for image matching.  
    ```python
    import cv2
    img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    ```

25. **Explain Hough Transform for line detection and provide an example.**  
    Hough Transform detects lines, useful for structural analysis.  
    ```python
    import cv2
    import numpy as np
    img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(img, 100, 200)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    ```

#### Advanced
26. **Implement a function to match features between two images using ORB.**  
    Feature matching aligns images for tasks like panorama stitching.  
    ```python
    import cv2
    def match_features(img1, img2):
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        return sorted(matches, key=lambda x: x.distance)
    img1 = cv2.imread('image1.jpg', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('image2.jpg', cv2.IMREAD_GRAYSCALE)
    matches = match_features(img1, img2)
    ```

27. **Write a function to detect circles using Hough Transform.**  
    Circle detection is used in applications like pupil tracking.  
    ```python
    import cv2
    import numpy as np
    def detect_circles(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 5)
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
        return img
    img = cv2.imread('image.jpg')
    circled = detect_circles(img)
    ```

28. **Explain how to use SURF for robust feature detection.**  
    SURF is faster than SIFT and robust to scale/rotation, used for object recognition (requires `opencv-contrib-python`).  
    ```python
    import cv2
    img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
    surf = cv2.xfeatures2d.SURF_create()
    keypoints, descriptors = surf.detectAndCompute(img, None)
    ```

### Image Transformations

#### Basic
29. **What is an affine transformation in OpenCV? Provide an example.**  
    Affine transformations (e.g., translation, rotation) preserve lines, used for image alignment.  
    ```python
    import cv2
    import numpy as np
    img = cv2.imread('image.jpg')
    h, w = img.shape[:2]
    M = np.float32([[1, 0, 50], [0, 1, 20]])  # Translate
    translated = cv2.warpAffine(img, M, (w, h))
    ```

30. **How do you flip an image in OpenCV? Give an example.**  
    Flipping augments data for training vision models.  
    ```python
    import cv2
    img = cv2.imread('image.jpg')
    flipped = cv2.flip(img, 1)  # Horizontal flip
    ```

31. **Explain perspective transformation in OpenCV.**  
    Perspective transformation corrects distortions (e.g., for document scanning).  
    ```python
    import cv2
    import numpy as np
    img = cv2.imread('image.jpg')
    pts1 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
    pts2 = np.float32([[0, 0], [400, 0], [0, 400], [400, 400]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(img, M, (400, 400))
    ```

#### Intermediate
32. **Write a function to apply a shear transformation to an image.**  
    Shear transformations augment data or simulate distortions.  
    ```python
    import cv2
    import numpy as np
    def shear_image(img, shear_factor=0.2):
        h, w = img.shape[:2]
        M = np.float32([[1, shear_factor, 0], [0, 1, 0]])
        return cv2.warpAffine(img, M, (w, h))
    img = cv2.imread('image.jpg')
    sheared = shear_image(img)
    ```

33. **How do you perform image translation in OpenCV? Provide an example.**  
    Translation shifts images, used in data augmentation.  
    ```python
    import cv2
    import numpy as np
    img = cv2.imread('image.jpg')
    h, w = img.shape[:2]
    M = np.float32([[1, 0, 100], [0, 1, 50]])
    translated = cv2.warpAffine(img, M, (w, h))
    ```

34. **Implement a function to scale an image non-uniformly.**  
    Non-uniform scaling adjusts dimensions differently, useful for specific model inputs.  
    ```python
    import cv2
    def scale_image(img, scale_x, scale_y):
        return cv2.resize(img, None, fx=scale_x, fy=scale_y)
    img = cv2.imread('image.jpg')
    scaled = scale_image(img, 1.5, 0.8)
    ```

#### Advanced
35. **Write a function to apply a homography transformation between two images.**  
    Homography aligns images with different perspectives (e.g., for stitching).  
    ```python
    import cv2
    import numpy as np
    def apply_homography(img1, img2, pts1, pts2):
        H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC)
        h, w = img2.shape[:2]
        return cv2.warpPerspective(img1, H, (w, h))
    img1 = cv2.imread('image1.jpg')
    img2 = cv2.imread('image2.jpg')
    pts1 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
    pts2 = np.float32([[10, 10], [310, 0], [10, 310], [310, 310]])
    warped = apply_homography(img1, img2, pts1, pts2)
    ```

36. **Explain how to use optical flow for motion tracking in OpenCV.**  
    Optical flow tracks pixel motion between frames, used in video analysis.  
    ```python
    import cv2
    import numpy as np
    frame1 = cv2.imread('frame1.jpg', cv2.IMREAD_GRAYSCALE)
    frame2 = cv2.imread('frame2.jpg', cv2.IMREAD_GRAYSCALE)
    flow = cv2.calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    ```

37. **Implement a function to stabilize video frames using feature matching.**  
    Stabilization aligns frames for smoother video processing.  
    ```python
    import cv2
    import numpy as np
    def stabilize_frame(prev_frame, curr_frame):
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(prev_frame, None)
        kp2, des2 = orb.detectAndCompute(curr_frame, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC)
        h, w = prev_frame.shape[:2]
        return cv2.warpPerspective(curr_frame, H, (w, h))
    ```

### Video Processing

#### Basic
38. **How do you read a video file in OpenCV? Provide an example.**  
    Video reading enables frame-by-frame analysis for tasks like object tracking.  
    ```python
    import cv2
    cap = cv2.VideoCapture('video.mp4')
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    ```

39. **What is the `cv2.VideoWriter` class, and how is it used?**  
    `VideoWriter` saves processed video frames (e.g., for annotated outputs).  
    ```python
    import cv2
    cap = cv2.VideoCapture('video.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    cap.release()
    out.release()
    ```

40. **How do you capture webcam input in OpenCV?**  
    Webcam input is used for real-time vision applications like face detection.  
    ```python
    import cv2
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        cv2.imshow('Webcam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    ```

#### Intermediate
41. **Write a function to extract every nth frame from a video.**  
    Frame extraction reduces data for efficient processing.  
    ```python
    import cv2
    def extract_frames(video_path, n):
        cap = cv2.VideoCapture(video_path)
        frames = []
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if count % n == 0:
                frames.append(frame)
            count += 1
        cap.release()
        return frames
    ```

42. **How do you compute the difference between consecutive video frames?**  
    Frame differencing detects motion for tracking or event detection.  
    ```python
    import cv2
    cap = cv2.VideoCapture('video.mp4')
    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_gray, gray)
        prev_gray = gray
    cap.release()
    ```

43. **Implement a function to convert a video to grayscale.**  
    Grayscale conversion simplifies video processing for tasks like edge detection.  
    ```python
    import cv2
    def grayscale_video(input_path, output_path):
        cap = cv2.VideoCapture(input_path)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))), isColor=False)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            out.write(gray)
        cap.release()
        out.release()
    ```

#### Advanced
44. **Write a function to track an object in a video using background subtraction.**  
    Background subtraction isolates moving objects for tracking.  
    ```python
    import cv2
    def track_object(video_path):
        cap = cv2.VideoCapture(video_path)
        fgbg = cv2.createBackgroundSubtractorMOG2()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            fgmask = fgbg.apply(frame)
            cv2.imshow('Tracking', fgmask)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    ```

45. **Implement a function to detect faces in a video stream using Haar cascades.**  
    Face detection is a common real-time vision task.  
    ```python
    import cv2
    def detect_faces(video_path):
        cap = cv2.VideoCapture(video_path)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.imshow('Faces', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    ```

46. **Explain how to use Deep Neural Networks (DNN) module in OpenCV for object detection.**  
    OpenCV’s DNN module runs pre-trained models (e.g., YOLO) for real-time detection, integrating with frameworks like TensorFlow.  
    ```python
    import cv2
    net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
    img = cv2.imread('image.jpg')
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True)
    net.setInput(blob)
    outputs = net.forward(net.getUnconnectedOutLayersNames())
    ```

### Object Detection and Recognition

#### Basic
47. **What is Haar cascade in OpenCV, and how is it used for face detection?**  
    Haar cascades are classifiers for detecting objects like faces using features.  
    ```python
    import cv2
    img = cv2.imread('image.jpg')
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    ```

48. **How do you draw bounding boxes around detected objects? Give an example.**  
    Bounding boxes visualize detected objects for annotation.  
    ```python
    import cv2
    img = cv2.imread('image.jpg')
    boxes = [(50, 50, 100, 100)]  # (x, y, w, h)
    for (x, y, w, h) in boxes:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    ```

49. **Explain template matching in OpenCV.**  
    Template matching finds a small image (template) in a larger image, used for pattern recognition.  
    ```python
    import cv2
    img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
    template = cv2.imread('template.jpg', cv2.IMREAD_GRAYSCALE)
    result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    ```

#### Intermediate
50. **Write a function to perform template matching and draw the result.**  
    Visualizes matched regions for object localization.  
    ```python
    import cv2
    import numpy as np
    def template_match(img, template):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        temp_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        result = cv2.matchTemplate(gray, temp_gray, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        top_left = max_loc
        h, w = temp_gray.shape
        cv2.rectangle(img, top_left, (top_left[0] + w, top_left[1] + h), (0, 255, 0), 2)
        return img
    img = cv2.imread('image.jpg')
    template = cv2.imread('template.jpg')
    matched = template_match(img, template)
    ```

51. **How do you use HOG descriptors for pedestrian detection?**  
    HOG (Histogram of Oriented Gradients) extracts features for detecting objects like pedestrians.  
    ```python
    import cv2
    img = cv2.imread('image.jpg')
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    boxes, _ = hog.detectMultiScale(img, winStride=(8, 8))
    ```

52. **Implement a function to detect eyes in an image using Haar cascades.**  
    Eye detection is used in facial analysis applications.  
    ```python
    import cv2
    def detect_eyes(img):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(img, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 2)
        return img
    img = cv2.imread('image.jpg')
    eyed = detect_eyes(img)
    ```

#### Advanced
53. **Write a function to perform YOLO object detection using OpenCV DNN.**  
    YOLO provides real-time object detection with high accuracy.  
    ```python
    import cv2
    import numpy as np
    def yolo_detect(img, config_path, weights_path, classes_path):
        net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        with open(classes_path, 'r') as f:
            classes = [line.strip() for line in f]
        blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True)
        net.setInput(blob)
        outputs = net.forward(net.getUnconnectedOutLayersNames())
        boxes, confidences, class_ids = [], [], []
        h, w = img.shape[:2]
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    box = detection[0:4] * np.array([w, h, w, h])
                    (center_x, center_y, width, height) = box.astype("int")
                    x = int(center_x - width / 2)
                    y = int(center_y - height / 2)
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        for i in indices:
            box = boxes[i]
            x, y, w, h = box
            label = str(classes[class_ids[i]])
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return img
    img = cv2.imread('image.jpg')
    detected = yolo_detect(img, 'yolov3.cfg', 'yolov3.weights', 'coco.names')
    ```

54. **Explain how to fine-tune a pre-trained model using OpenCV DNN.**  
    Fine-tuning adapts models like MobileNet for specific tasks, using OpenCV to load and run inference.  
    ```python
    import cv2
    net = cv2.dnn.readNet('model.onnx')
    img = cv2.imread('image.jpg')
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (224, 224), swapRB=True)
    net.setInput(blob)
    output = net.forward()
    ```

55. **Implement a function for real-time object tracking using OpenCV trackers.**  
    Trackers like CSRT follow objects across frames in videos.  
    ```python
    import cv2
    def track_object(video_path, bbox):
        cap = cv2.VideoCapture(video_path)
        tracker = cv2.TrackerCSRT_create()
        ret, frame = cap.read()
        tracker.init(frame, bbox)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            success, box = tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.imshow('Tracking', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    ```

### Image Segmentation

#### Basic
56. **What is image segmentation, and how is it performed in OpenCV?**  
    Segmentation partitions images into regions (e.g., objects vs. background).  
    ```python
    import cv2
    img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
    _, segmented = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    ```

57. **How do you use watershed algorithm for segmentation in OpenCV?**  
    Watershed separates touching objects based on markers.  
    ```python
    import cv2
    import numpy as np
    img = cv2.imread('image.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    markers = cv2.watershed(img, np.zeros_like(gray, dtype=np.int32))
    ```

58. **Explain GrabCut for foreground-background segmentation.**  
    GrabCut iteratively separates foreground from background using a bounding box.  
    ```python
    import cv2
    import numpy as np
    img = cv2.imread('image.jpg')
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (50, 50, 200, 200)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    ```

#### Intermediate
59. **Write a function to perform k-means clustering for color-based segmentation.**  
    K-means groups pixels by color, segmenting images into regions.  
    ```python
    import cv2
    import numpy as np
    def kmeans_segmentation(img, k=3):
        pixels = img.reshape((-1, 3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        segmented = centers[labels.flatten()]
        return segmented.reshape(img.shape)
    img = cv2.imread('image.jpg')
    segmented = kmeans_segmentation(img)
    ```

60. **How do you use contour-based segmentation in OpenCV?**  
    Contours define object boundaries for segmentation.  
    ```python
    import cv2
    img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(img)
    cv2.drawContours(mask, contours, -1, 255, -1)
    ```

61. **Implement a function for semantic segmentation using a pre-trained model.**  
    Semantic segmentation assigns class labels to pixels using DNN models.  
    ```python
    import cv2
    import numpy as np
    def semantic_segmentation(img, model_path):
        net = cv2.dnn.readNet(model_path)
        blob = cv2.dnn.blobFromImage(img, 1/255.0, (512, 512), swapRB=True)
        net.setInput(blob)
        output = net.forward()
        return np.argmax(output[0], axis=0)
    img = cv2.imread('image.jpg')
    segmented = semantic_segmentation(img, 'model.onnx')
    ```

#### Advanced
62. **Write a function to combine watershed and GrabCut for robust segmentation.**  
    Combining methods improves accuracy for complex scenes.  
    ```python
    import cv2
    import numpy as np
    def combined_segmentation(img, rect):
        mask = np.zeros(img.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        markers = cv2.watershed(img, mask2.astype(np.int32))
        img[markers == -1] = [255, 0, 0]
        return img
    img = cv2.imread('image.jpg')
    rect = (50, 50, 200, 200)
    segmented = combined_segmentation(img, rect)
    ```

63. **Explain how to use DeepLab for segmentation in OpenCV.**  
    DeepLab models provide high-accuracy semantic segmentation, run via OpenCV’s DNN module.  
    ```python
    import cv2
    net = cv2.dnn.readNet('deeplabv3.onnx')
    img = cv2.imread('image.jpg')
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (513, 513), swapRB=True)
    net.setInput(blob)
    output = net.forward()
    ```

64. **Implement a function for instance segmentation using Mask R-CNN.**  
    Instance segmentation identifies and masks individual objects.  
    ```python
    import cv2
    import numpy as np
    def mask_rcnn_segmentation(img, config_path, weights_path):
        net = cv2.dnn.readNetFromTensorflow(weights_path, config_path)
        blob = cv2.dnn.blobFromImage(img, swapRB=True)
        net.setInput(blob)
        boxes, masks = net.forward(['detection_out_final', 'detection_masks'])
        return boxes, masks
    img = cv2.imread('image.jpg')
    boxes, masks = mask_rcnn_segmentation(img, 'mask_rcnn.cfg', 'mask_rcnn.weights')
    ```

### Camera Calibration

#### Basic
65. **What is camera calibration in OpenCV, and why is it important?**  
    Calibration corrects lens distortions, critical for accurate 3D reconstruction.  
    ```python
    import cv2
    import numpy as np
    objp = np.zeros((6*7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)
    ```

66. **How do you find chessboard corners for calibration? Provide an example.**  
    Chessboard corners provide points for computing camera parameters.  
    ```python
    import cv2
    img = cv2.imread('chessboard.jpg', cv2.IMREAD_GRAYSCALE)
    ret, corners = cv2.findChessboardCorners(img, (7, 6), None)
    ```

67. **Explain the `cv2.calibrateCamera` function.**  
    Computes intrinsic/extrinsic parameters from object points and image points.  
    ```python
    import cv2
    import numpy as np
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([objp], [corners], img.shape[::-1], None, None)
    ```

#### Intermediate
68. **Write a function to undistort an image using camera calibration parameters.**  
    Undistortion corrects lens effects for accurate analysis.  
    ```python
    import cv2
    import numpy as np
    def undistort_image(img, mtx, dist):
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        undistorted = cv2.undistort(img, mtx, dist, None, newcameramtx)
        x, y, w, h = roi
        return undistorted[y:y+h, x:x+w]
    img = cv2.imread('image.jpg')
    mtx = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    dist = np.array([k1, k2, p1, p2, k3])
    undistorted = undistort_image(img, mtx, dist)
    ```

69. **How do you compute the reprojection error in camera calibration?**  
    Reprojection error measures calibration accuracy.  
    ```python
    import cv2
    import numpy as np
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    print("Mean reprojection error:", mean_error / len(objpoints))
    ```

70. **Implement a function to calibrate a camera using multiple images.**  
    Uses multiple chessboard images for robust calibration.  
    ```python
    import cv2
    import numpy as np
    def calibrate_camera(images, pattern_size=(7, 6)):
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        objpoints, imgpoints = [], []
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        return mtx, dist
    ```

#### Advanced
71. **Explain stereo calibration in OpenCV.**  
    Stereo calibration computes relative positions of two cameras for 3D reconstruction.  
    ```python
    import cv2
    ret, mtx1, dist1, mtx2, dist2, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints1, imgpoints2, mtx1, dist1, mtx2, dist2, gray.shape[::-1])
    ```

72. **Write a function for 3D reconstruction using stereo images.**  
    Reconstructs 3D points from stereo pairs.  
    ```python
    import cv2
    import numpy as np
    def reconstruct_3d(img1, img2, mtx1, dist1, mtx2, dist2, R, T):
        img1 = cv2.undistort(img1, mtx1, dist1)
        img2 = cv2.undistort(img2, mtx2, dist2)
        stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
        disparity = stereo.compute(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY))
        Q = cv2.stereoRectify(mtx1, dist1, mtx2, dist2, img1.shape[:2], R, T)[4]
        points_3d = cv2.reprojectImageTo3D(disparity, Q)
        return points_3d
    ```

73. **Implement a function to estimate camera pose using solvePnP.**  
    Pose estimation determines camera orientation relative to an object.  
    ```python
    import cv2
    import numpy as np
    def estimate_pose(obj_points, img_points, mtx, dist):
        ret, rvec, tvec = cv2.solvePnP(obj_points, img_points, mtx, dist)
        return rvec, tvec
    ```

### Machine Learning Integration

#### Basic
74. **How do you prepare OpenCV images for machine learning models?**  
    Images are resized, normalized, and converted to arrays for ML input.  
    ```python
    import cv2
    import numpy as np
    img = cv2.imread('image.jpg')
    img = cv2.resize(img, (224, 224))  # Resize
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    ```

75. **What is the role of OpenCV in data augmentation for ML?**  
    OpenCV applies transformations like rotation, flipping, and cropping to augment datasets.  
    ```python
    import cv2
    img = cv2.imread('image.jpg')
    flipped = cv2.flip(img, 1)
    rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    ```

76. **How do you extract features from images for traditional ML models?**  
    Features like HOG or SIFT are extracted for models like SVM.  
    ```python
    import cv2
    img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    ```

#### Intermediate
77. **Write a function to generate augmented images for ML training.**  
    Augmentation increases dataset diversity.  
    ```python
    import cv2
    import numpy as np
    def augment_image(img):
        augmented = []
        augmented.append(cv2.flip(img, 1))  # Horizontal flip
        augmented.append(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))
        h, w = img.shape[:2]
        M = np.float32([[1, 0, 50], [0, 1, 20]])
        augmented.append(cv2.warpAffine(img, M, (w, h)))  # Translate
        return augmented
    img = cv2.imread('image.jpg')
    augmented = augment_image(img)
    ```

78. **How do you integrate OpenCV with scikit-learn for image classification?**  
    Extract features with OpenCV and train with scikit-learn.  
    ```python
    import cv2
    from sklearn.svm import SVC
    img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
    hog = cv2.HOGDescriptor()
    features = hog.compute(img)
    clf = SVC()
    clf.fit([features], [1])  # Example training
    ```

79. **Implement a function to preprocess images for a CNN.**  
    Prepares images for deep learning frameworks like TensorFlow.  
    ```python
    import cv2
    import numpy as np
    def preprocess_for_cnn(img, size=(224, 224)):
        img = cv2.resize(img, size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        img = img / 255.0  # Normalize
        return np.expand_dims(img, axis=0)
    img = cv2.imread('image.jpg')
    processed = preprocess_for_cnn(img)
    ```

#### Advanced
80. **Write a function to use OpenCV with a pre-trained TensorFlow model.**  
    Runs inference on images using OpenCV and TensorFlow.  
    ```python
    import cv2
    import tensorflow as tf
    def run_inference(img, model_path):
        model = tf.keras.models.load_model(model_path)
        img = preprocess_for_cnn(img)
        return model.predict(img)
    img = cv2.imread('image.jpg')
    predictions = run_inference(img, 'model.h5')
    ```

81. **Explain how to use OpenCV for real-time inference with deep learning models.**  
    OpenCV’s DNN module enables real-time inference with low latency.  
    ```python
    import cv2
    net = cv2.dnn.readNet('model.onnx')
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (224, 224), swapRB=True)
        net.setInput(blob)
        output = net.forward()
        cv2.imshow('Inference', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    ```

82. **Implement a function for active learning with OpenCV.**  
    Selects uncertain samples for labeling using model predictions.  
    ```python
    import cv2
    import numpy as np
    def active_learning(images, model):
        uncertainties = []
        for img in images:
            processed = preprocess_for_cnn(img)
            pred = model.predict(processed)
            uncertainty = -np.sum(pred * np.log(pred), axis=1)  # Entropy
            uncertainties.append(uncertainty)
        return np.argsort(uncertainties)[-10:]  # Top 10 uncertain
    ```

### Performance Optimization

#### Basic
83. **How do you optimize image processing in OpenCV?**  
    Use efficient functions (e.g., `cv2.resize` over loops) and smaller image sizes.  
    ```python
    import cv2
    img = cv2.imread('image.jpg')
    resized = cv2.resize(img, (100, 100))  # Faster processing
    ```

84. **What is the role of NumPy in OpenCV performance?**  
    OpenCV uses NumPy for fast array operations, avoiding slow Python loops.  
    ```python
    import cv2
    import numpy as np
    img = cv2.imread('image.jpg')
    img = np.clip(img + 50, 0, 255).astype(np.uint8)  # Brighten
    ```

85. **How do you use multi-threading with OpenCV for video processing?**  
    Multi-threading parallelizes frame processing for real-time applications.  
    ```python
    import cv2
    import threading
    cap = cv2.VideoCapture('video.mp4')
    def process_frame(frame):
        return cv2.GaussianBlur(frame, (5, 5), 0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        t = threading.Thread(target=process_frame, args=(frame,))
        t.start()
    cap.release()
    ```

#### Intermediate
86. **Write a function to process large images in chunks.**  
    Chunking reduces memory usage for high-resolution images.  
    ```python
    import cv2
    import numpy as np
    def process_in_chunks(img, chunk_size=1000):
        h, w = img.shape[:2]
        for y in range(0, h, chunk_size):
            for x in range(0, w, chunk_size):
                chunk = img[y:y+chunk_size, x:x+chunk_size]
                chunk = cv2.GaussianBlur(chunk, (5, 5), 0)
                img[y:y+chunk_size, x:x+chunk_size] = chunk
        return img
    img = cv2.imread('image.jpg')
    processed = process_in_chunks(img)
    ```

87. **How do you use OpenCV with GPU acceleration?**  
    OpenCV’s CUDA module accelerates operations on compatible GPUs.  
    ```python
    import cv2
    img = cv2.imread('image.jpg')
    gpu_img = cv2.cuda_GpuMat()
    gpu_img.upload(img)
    gpu_blurred = cv2.cuda_GaussianBlur(gpu_img, (5, 5), 0)
    blurred = gpu_blurred.download()
    ```

88. **Implement a function to parallelize feature detection across multiple images.**  
    Parallel processing speeds up batch tasks.  
    ```python
    import cv2
    from multiprocessing import Pool
    def detect_features(img):
        orb = cv2.ORB_create()
        kp, des = orb.detectAndCompute(img, None)
        return kp, des
    def parallel_feature_detection(images):
        with Pool() as pool:
            results = pool.map(detect_features, images)
        return results
    images = [cv2.imread(f'image{i}.jpg', cv2.IMREAD_GRAYSCALE) for i in range(1, 5)]
    features = parallel_feature_detection(images)
    ```

#### Advanced
89. **Explain how to optimize real-time video processing in OpenCV.**  
    Use lower resolutions, skip frames, and leverage GPU or multi-threading.  
    ```python
    import cv2
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (320, 240))  # Lower resolution
        cv2.imshow('Optimized', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    ```

90. **Write a function to use OpenCV’s CUDA module for fast filtering.**  
    CUDA accelerates image filtering for large datasets.  
    ```python
    import cv2
    def cuda_filter(img):
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(img)
        gpu_filtered = cv2.cuda_bilateralFilter(gpu_img, 5, 50, 50)
        return gpu_filtered.download()
    img = cv2.imread('image.jpg')
    filtered = cuda_filter(img)
    ```

91. **Implement a function to benchmark OpenCV operations.**  
    Measures performance for optimization decisions.  
    ```python
    import cv2
    import time
    def benchmark_operation(img, func, iterations=100):
        start = time.time()
        for _ in range(iterations):
            func(img)
        return (time.time() - start) / iterations
    img = cv2.imread('image.jpg')
    time_taken = benchmark_operation(img, lambda x: cv2.GaussianBlur(x, (5, 5), 0))
    print(f"Average time: {time_taken} seconds")
    ```

### Integration with Other Libraries

#### Basic
92. **How do you use OpenCV with NumPy for efficient image processing?**  
    NumPy enables fast array operations for OpenCV images.  
    ```python
    import cv2
    import numpy as np
    img = cv2.imread('image.jpg')
    img = np.clip(img * 1.5, 0, 255).astype(np.uint8)  # Increase brightness
    ```

93. **What is the role of Matplotlib in visualizing OpenCV results?**  
    Matplotlib displays images and plots for analysis.  
    ```python
    import cv2
    import matplotlib.pyplot as plt
    img = cv2.imread('image.jpg')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.savefig('plot.png')
    ```

94. **How do you integrate OpenCV with Pandas for dataset preparation?**  
    Pandas organizes image metadata for ML pipelines.  
    ```python
    import cv2
    import pandas as pd
    images = ['image1.jpg', 'image2.jpg']
    data = {'path': images, 'width': [cv2.imread(img).shape[1] for img in images]}
    df = pd.DataFrame(data)
    ```

#### Intermediate
95. **Write a function to visualize OpenCV edge detection with Matplotlib.**  
    Visualizes edges for analysis.  
    ```python
    import cv2
    import matplotlib.pyplot as plt
    def visualize_edges(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        plt.imshow(edges, cmap='gray')
        plt.savefig('edges.png')
    img = cv2.imread('image.jpg')
    visualize_edges(img)
    ```

96. **How do you use OpenCV with scikit-learn for clustering image pixels?**  
    Clusters pixels for segmentation.  
    ```python
    import cv2
    from sklearn.cluster import KMeans
    img = cv2.imread('image.jpg')
    pixels = img.reshape(-1, 3)
    kmeans = KMeans(n_clusters=3)
    labels = kmeans.fit_predict(pixels)
    segmented = kmeans.cluster_centers_[labels].reshape(img.shape).astype(np.uint8)
    ```

97. **Implement a function to load and preprocess images with OpenCV and TensorFlow.**  
    Prepares images for deep learning.  
    ```python
    import cv2
    import tensorflow as tf
    def load_and_preprocess(img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        return tf.convert_to_tensor(img)
    ```

#### Advanced
98. **Write a function to combine OpenCV and Dlib for facial landmark detection.**  
    Integrates OpenCV with Dlib for precise face analysis.  
    ```python
    import cv2
    import dlib
    def facial_landmarks(img):
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        for face in faces:
            landmarks = predictor(gray, face)
            for n in range(0, 68):
                x, y = landmarks.part(n).x, landmarks.part(n).y
                cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
        return img
    img = cv2.imread('image.jpg')
    landmarked = facial_landmarks(img)
    ```

99. **Explain how to use OpenCV with PyTorch for real-time inference.**  
    OpenCV preprocesses frames, and PyTorch runs the model.  
    ```python
    import cv2
    import torch
    model = torch.load('model.pt')
    model.eval()
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img = preprocess_for_cnn(frame)
        with torch.no_grad():
            output = model(torch.tensor(img, dtype=torch.float32))
        cv2.imshow('Inference', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    ```

100. **Implement a function to generate a dataset with OpenCV and Pandas.**  
     Creates structured datasets for ML.  
     ```python
     import cv2
     import pandas as pd
     def create_dataset(image_paths, labels):
         data = []
         for path, label in zip(image_paths, labels):
             img = cv2.imread(path)
             h, w, c = img.shape
             data.append({'path': path, 'label': label, 'width': w, 'height': h})
         return pd.DataFrame(data)
     ```

### Error Handling

#### Basic
101. **How do you handle file loading errors in OpenCV?**  
     Check if the image is loaded correctly to avoid crashes.  
     ```python
     import cv2
     img = cv2.imread('image.jpg')
     if img is None:
         raise FileNotFoundError("Image not found")
     ```

102. **What happens if you pass an invalid parameter to an OpenCV function?**  
     OpenCV raises exceptions (e.g., `cv2.error`).  
     ```python
     import cv2
     try:
         img = cv2.imread('image.jpg')
         cv2.resize(img, (0, 0))  # Invalid size
     except cv2.error as e:
         print("OpenCV error:", e)
     ```

103. **How do you handle video capture errors in OpenCV?**  
     Verify capture initialization and frame reading.  
     ```python
     import cv2
     cap = cv2.VideoCapture('video.mp4')
     if not cap.isOpened():
         raise RuntimeError("Cannot open video")
     ```

#### Intermediate
104. **Write a function with error handling for image processing.**  
     Ensures robust processing pipelines.  
     ```python
     import cv2
     def process_image(img_path):
         try:
             img = cv2.imread(img_path)
             if img is None:
                 raise FileNotFoundError("Image not found")
             return cv2.GaussianBlur(img, (5, 5), 0)
         except Exception as e:
             print(f"Error: {e}")
             return None
     ```

105. **How do you handle memory issues with large images in OpenCV?**  
     Use smaller resolutions or chunking to manage memory.  
     ```python
     import cv2
     try:
         img = cv2.imread('large_image.jpg')
         img = cv2.resize(img, (1000, 1000))  # Reduce size
     except MemoryError:
         print("Image too large")
     ```

106. **Implement a function to retry failed video frame reads.**  
     Retries handle transient errors in video streams.  
     ```python
     import cv2
     def read_frame_with_retry(cap, max_attempts=3):
         for _ in range(max_attempts):
             ret, frame = cap.read()
             if ret:
                 return frame
         raise RuntimeError("Failed to read frame")
     ```

#### Advanced
107. **Create a custom exception class for OpenCV errors.**  
     Defines specific errors for vision tasks.  
     ```python
     class OpenCVError(Exception):
         pass
     def process_image(img_path):
         img = cv2.imread(img_path)
         if img is None:
             raise OpenCVError("Failed to load image")
         return img
     ```

108. **Write a function to handle cascading errors in a processing pipeline.**  
     Manages multiple potential failures.  
     ```python
     import cv2
     def process_pipeline(img_path):
         try:
             img = cv2.imread(img_path)
             if img- is None:
                 raise OpenCVError("Image not found")
             img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
             edges = cv2.Canny(img, 100, 200)
             return edges
         except OpenCVError as e:
             print(f"Pipeline error: {e}")
             return None
         except cv2.error as e:
             print(f"OpenCV error: {e}")
             return None
     ```

109. **Explain how to log errors in OpenCV applications.**  
     Use Python’s `logging` module to track issues in vision pipelines.  
     ```python
     import cv2
     import logging
     logging.basicConfig(level=logging.ERROR)
     try:
         img = cv2.imread('image.jpg')
         cv2.resize(img, (0, 0))
     except cv2.error as e:
         logging.error(f"OpenCV error: {e}")
     ```

### Advanced Techniques

#### Basic
110. **What is image inpainting in OpenCV, and how is it used?**  
     Inpainting repairs damaged image regions, useful for preprocessing.  
     ```python
     import cv2
     img = cv2.imread('image.jpg')
     mask = cv2.imread('mask.jpg', cv2.IMREAD_GRAYSCALE)
     inpainted = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
     ```

111. **How do you perform image stitching in OpenCV?**  
     Stitching combines images into panoramas.  
     ```python
     import cv2
     images = [cv2.imread(f'image{i}.jpg') for i in range(1, 3)]
     stitcher = cv2.Stitcher_create()
     status, pano = stitcher.stitch(images)
     ```

112. **Explain the `cv2.calcHist` function for histogram computation.**  
     Computes histograms for analyzing pixel distributions.  
     ```python
     import cv2
     img = cv2.imread('image.jpg')
     hist = cv2.calcHist([img], [0], None, [256], [0, 256])  # Blue channel
     ```

#### Intermediate
113. **Write a function to perform image inpainting with a dynamic mask.**  
     Repairs specific regions based on user input.  
     ```python
     import cv2
     import numpy as np
     def inpaint_image(img, points):
         mask = np.zeros(img.shape[:2], np.uint8)
         for (x, y) in points:
             cv2.circle(mask, (x, y), 5, 255, -1)
         return cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)
     img = cv2.imread('image.jpg')
     points = [(100, 100), (150, 150)]
     inpainted = inpaint_image(img, points)
     ```

114. **How do you use OpenCV for augmented reality?**  
     Overlay virtual objects using pose estimation and transformations.  
     ```python
     import cv2
     import numpy as np
     img = cv2.imread('image.jpg')
     obj_points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32)
     img_points = np.array([[100, 100], [200, 100], [100, 200], [200, 200]], dtype=np.float32)
     ret, rvec, tvec = cv2.solvePnP(obj_points, img_points, mtx, dist)
     ```

115. **Implement a function to compute and visualize image histograms.**  
     Analyzes color/intensity distributions.  
     ```python
     import cv2
     import matplotlib.pyplot as plt
     def plot_histogram(img):
         colors = ('b', 'g', 'r')
         for i, col in enumerate(colors):
             hist = cv2.calcHist([img], [i], None, [256], [0, 256])
             plt.plot(hist, color=col)
         plt.savefig('histogram.png')
     img = cv2.imread('image.jpg')
     plot_histogram(img)
     ```

#### Advanced
116. **Write a function for real-time augmented reality with OpenCV.**  
     Overlays objects in video streams.  
     ```python
     import cv2
     import numpy as np
     def ar_overlay(video_path, obj_img, obj_points, img_points, mtx, dist):
         cap = cv2.VideoCapture(video_path)
         while cap.isOpened():
             ret, frame = cap.read()
             if not ret:
                 break
             ret, rvec, tvec = cv2.solvePnP(obj_points, img_points, mtx, dist)
             imgpts, _ = cv2.projectPoints(np.float32([[0, 0, 0]]), rvec, tvec, mtx, dist)
             cv2.warpPerspective(obj_img, frame, imgpts)
             cv2.imshow('AR', frame)
             if cv2.waitKey(1) & 0xFF == ord('q'):
                 break
         cap.release()
         cv2.destroyAllWindows()
     ```

117. **Explain how to use OpenCV for SLAM (Simultaneous Localization and Mapping).**  
     SLAM combines feature detection, tracking, and pose estimation for robotic navigation.  
     ```python
     import cv2
     orb = cv2.ORB_create()
     cap = cv2.VideoCapture(0)
     while True:
         ret, frame = cap.read()
         if not ret:
             break
         kp, des = orb.detectAndCompute(frame, None)
         # Process for SLAM
     ```

118. **Implement a function for depth estimation using stereo vision.**  
     Estimates depth from stereo image pairs.  
     ```python
     import cv2
     def depth_map(img1, img2):
         stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
         disparity = stereo.compute(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY))
         return disparity / 16.0
     img1 = cv2.imread('left.jpg')
     img2 = cv2.imread('right.jpg')
     depth = depth_map(img1, img2)
     ```

### Additional Coding Questions

119. **Write a function to detect and remove a green screen background.**  
     Chroma keying for video editing.  
     ```python
     import cv2
     import numpy as np
     def remove_green_screen(img, lower_green=(0, 100, 0), upper_green=(100, 255, 100)):
         hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
         mask = cv2.inRange(hsv, lower_green, upper_green)
         mask = cv2.bitwise_not(mask)
         return cv2.bitwise_and(img, img, mask=mask)
     img = cv2.imread('image.jpg')
     result = remove_green_screen(img)
     ```

120. **Implement a function to compute image gradients using Sobel operators.**  
     Gradients highlight edges for feature extraction.  
     ```python
     import cv2
     def compute_gradients(img):
         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
         sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
         sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
         return sobelx, sobely
     img = cv2.imread('image.jpg')
     grad_x, grad_y = compute_gradients(img)
     ```

121. **Write a function to detect text in an image using EAST detector.**  
     Text detection for OCR applications.  
     ```python
     import cv2
     import numpy as np
     def detect_text(img, model_path):
         net = cv2.dnn.readNet(model_path)
         blob = cv2.dnn.blobFromImage(img, 1.0, (320, 320), (123.68, 116.78, 103.94), swapRB=True)
         net.setInput(blob)
         (scores, geometry) = net.forward(['feature_fusion/Conv_7/Sigmoid', 'feature_fusion/concat_3'])
         return scores, geometry
     img = cv2.imread('image.jpg')
     scores, geometry = detect_text(img, 'east_text_detection.pb')
     ```

122. **Implement a function to create a mosaic effect on an image.**  
     Mosaics anonymize or stylize images.  
     ```python
     import cv2
     def mosaic_effect(img, block_size=10):
         h, w = img.shape[:2]
         for y in range(0, h, block_size):
             for x in range(0, w, block_size):
                 block = img[y:y+block_size, x:x+block_size]
                 avg_color = block.mean(axis=(0, 1)).astype(np.uint8)
                 img[y:y+block_size, x:x+block_size] = avg_color
         return img
     img = cv2.imread('image.jpg')
     mosaiced = mosaic_effect(img)
     ```

123. **Write a function to perform image blending using OpenCV.**  
     Blending combines images for effects or overlays.  
     ```python
     import cv2
     def blend_images(img1, img2, alpha=0.5):
         return cv2.addWeighted(img1, alpha, img2, 1-alpha, 0.0)
     img1 = cv2.imread('image1.jpg')
     img2 = cv2.imread('image2.jpg')
     blended = blend_images(img1, img2)
     ```

124. **Implement a function to detect and count objects in an image.**  
     Counts segmented objects for analysis.  
     ```python
     import cv2
     import numpy as np
     def count_objects(img):
         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
         _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
         contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
         return len(contours)
     img = cv2.imread('image.jpg')
     count = count_objects(img)
     ```