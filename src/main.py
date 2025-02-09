import cv2
from src.detection import load_model, detect_belugas # import model and functions to detect
from src.tracking import track_belugas, Track # import functions to integrate tracking with Bytetrack
from src.bytetrack.byte_tracker import BYTETracker # import ByteTracker functions to track 
import os
import argparse

"""Draws bounding boxes with different colors for adults, calves, and unknowns."""
def draw_boxes(image, detections, tracks=None):
    
    # if an object is detected, then...
    if detections:
        for x1, y1, x2, y2, conf, cls in detections:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            # based on the class, assign corresponding label and color
            if cls == 0:  # 0 is adult
                color = (0, 0, 255)  # Red for adults
                label = f"Adult Beluga: {conf:.2f}"

            elif cls == 1:  # 1 is calf
                color = (206, 255, 0)  # Cyan for calves
                label = f"Calf: {conf:.2f}"

            # --- At the moment, our footage does not include any unknowns, if we find data with other objects like boats, other animals, swimmers, etc. we can then add unkowns as another class ---   
            else:
                color = (255, 255, 255)  # White for unknown
                label = f"Unknown: {conf:.2f}"

            # annotates the frame with a bounding box and label 
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # if the object is tracked across multiple frames, then...
    if tracks:
        for track in tracks:
            x1, y1, x2, y2 = map(int, track.tlbr)
            track_id = track.track_id
            # blue for tracked objects 
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(image, str(track_id), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    return image


def process_video(model, video_path, args, output_path=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    tracker = BYTETracker(args)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detect_belugas(model, frame)

        if detections is not None:
            tracks = track_belugas(detections, tracker)
            frame_with_boxes = draw_boxes(frame, detections, tracks)
        else:
            frame_with_boxes = draw_boxes(frame, detections)


        if out:
            out.write(frame_with_boxes)
        else:
            cv2.imshow("Beluga Detection and Tracking", frame_with_boxes)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    model_path = 'models/yolov8n.pt'  # Path to YOLOv8 model
    model = load_model(model_path)

    if model is None:
        print("Failed to load model.")
        exit()

    video_path = 'data/videos/test_video2.mp4'  # path to video file 
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        exit()

    args = argparse.Namespace()
    args.track_thresh = 0.5  # Adjustable
    args.num_queries = 100  # Adjustable


    output_video_path = 'data/results/output_video.mp4'  # Choose an output path
    process_video(model, video_path, args, output_video_path)
    print(f"Output video saved to: {output_video_path}")


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
        # Final deployment
    # data/videos/test_video2.mp4


        