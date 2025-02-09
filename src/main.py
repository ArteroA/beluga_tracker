# main.py
import cv2  # Import the OpenCV library for computer vision tasks.
from src.detection import load_model, detect_belugas  # Import functions from your detection module.
from src.tracking import track_belugas, Track  # Import tracking functions and the Track class.

def draw_boxes(image, detections, tracks=None):
    """Draws bounding boxes and labels on the image for both detections and tracks.

    Args:
        image (np.ndarray): The input image (a NumPy array).
        detections (list): A list of detections, where each detection is a list:
                           [x_min, y_min, x_max, y_max, confidence, class_id].
        tracks (list, optional): A list of Track objects. Defaults to None.

    Returns:
        np.ndarray: The image with bounding boxes and labels drawn.
    """
    if detections:  # Check if there are any detections
        for x1, y1, x2, y2, conf, cls in detections:  # Iterate through each detection
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])  # Convert coordinates to integers
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw a green rectangle (detection)
            label = f"Beluga: {conf:.2f}"  # Create a label with the confidence score
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)  # Put text

    if tracks:  # Check if there are any tracks (for video processing)
        for track in tracks:  # Iterate through each track
            x1, y1, x2, y2 = map(int, track.tlbr)  # Get bounding box from Track object
            track_id = track.track_id  # Get the track ID
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Draw a blue rectangle (track)
            cv2.putText(image, str(track_id), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)  # Put ID

    return image  # Return the image with boxes and labels


if __name__ == '__main__':
    
    model_path = 'models/yolov8n.pt'  # Path to your YOLOv8 model file
    model = load_model(model_path)  # Load the YOLOv8 model

    if model is None:  # Check if model loading failed
        print("Failed to load model.")
        exit() 

    image_path = 'test_image4.jpg'  # Path to test image
    detections = detect_belugas(model, image_path)  # Perform detection on the image

    if detections is not None:  # Check if detections were found
        print(f"Detections: {detections}")  # Print the raw detection data

        # Create Track objects for each detection (for this single-frame test)
        # This simulates tracking for a single frame.  Each detection becomes a track.
        tracks = [Track(track_id=i + 1, tlbr=[x1, y1, x2, y2]) for i, (x1, y1, x2, y2, _, _) in enumerate(detections)]

        image = cv2.imread(image_path)  # Read the image using OpenCV
        image_with_boxes = draw_boxes(image, detections, tracks)  # Draw boxes on the image

        cv2.imshow("Detections and Tracks", image_with_boxes)  # Display the image
        cv2.waitKey(0)  # Wait for a key press (keeps the window open)
        cv2.destroyAllWindows()  # Close the display window
    else:
        print("No detections found or error during detection.")  # Handle cases with no detections





        # Things to do: Train on our own data once fully annotated. Right now it is using the COCO dataset which contains 80 common object categories. Adding our on data will improve the accuracy of the model.
            # Data collection
            # Annotation and Labeling using roboflow around 192 videos (Should we utilize the roboflow training feature todriectly train our Yolov8 model)
            # Creating the dataset of label images
            # Training the current detection model. Either fine tune the existing model or train a new model
                # Changing hyperparameters
                # Running the training script
                # Monitoring the progress
                # Validation to get estimate of performance
            # Evaluate on a test set to meausure accuracy
            # Final deployment!