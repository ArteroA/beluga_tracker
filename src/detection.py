import cv2  # OpenCV for image/video processing
from ultralytics import YOLO  # Import the YOLOv8 class


def load_model(model_path):
    """Loads the YOLOv8 model from the specified path.

    Args:
        model_path (str): The path to the YOLOv8 model file (.pt).

    Returns:
        ultralytics.YOLO: The loaded YOLOv8 model, or None if an error occurred.
    """
    try:
        model = YOLO(model_path)  # Load the YOLOv8 model
        return model
    except Exception as e:
        print(f"Error loading model: {e}")  # Print any error that occurs
        return None


def detect_belugas(model, image, conf_thres=0.5, iou_thres=0.45):
    """Detects beluga whales in a single image frame.

    Args:
        model (ultralytics.YOLO): The loaded YOLOv8 model.
        image (np.ndarray):  The input image frame (a NumPy array).
        conf_thres (float): Confidence threshold. Detections below this are ignored.
        iou_thres (float): Intersection-over-Union threshold for Non-Maximum Suppression.

    Returns:
        list: A list of detections. Each detection is a list:
              [x_min, y_min, x_max, y_max, confidence, class_id].
              Returns None if there's an error or no detections.
    """
    try:
        # Perform inference.  The results object contains all the information
        # about the detections.
        results = model.predict(image, conf=conf_thres, iou=iou_thres, verbose=False)

        # results is a list, where each element corresponds to an image in a batch.
        # Since we're processing a single image, we only care about the first element.
        detections = []
        boxes  = results[0].boxes.data.tolist()
        if boxes != None:
            for box in boxes:
                x1, y1, x2, y2, confidence, class_id = box #get variables from box
                # Convert to integer coordinates
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                detections.append([x1, y1, x2, y2, confidence, int(class_id)])

        return detections

    except Exception as e:
        print(f"Error during detection: {e}")
        return None