# main.py (Complete Demo Version)
import cv2
from src.detection import load_model, detect_belugas
from src.tracking import track_belugas, Track
from src.bytetrack.byte_tracker import BYTETracker
import os
import argparse

def draw_boxes(image, detections, tracks=None):
    """Draws bounding boxes with different colors for adults and calves."""
    if detections:
        for x1, y1, x2, y2, conf, cls in detections:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            # DEMO Classification -- Defaults to "Adult" for demo purposes. Model still needs more training
            if cls == 1:  # 1 is calf
                color = (0, 0, 255)  # Red
                label = f"Calf: {conf:.2f}"
            else: # 0 defaults to Adult 
                color = (0, 255, 0)  # Green
                label = f"Adult: {conf:.2f}"

            # Actual Classification conditions, pending a model dedicated to beluga whales
            # if cls == 0:  # 0 is adult
            #    color = (0, 255, 0)  # Green
            #    label = f"Adult: {conf:.2f}"
            # elif cls == 1:  # 1 is calf
            #    color = (0, 0, 255)  # Red
            #    label = f"Calf: {conf:.2f}"
            # else:
            #    color = (255, 255, 255) # White
            #    label = f"Unknown (Class {cls}): {conf:.2f}"

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    if tracks:
        for track in tracks:
            x1, y1, x2, y2 = map(int, track.tlbr)
            track_id = track.track_id
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue for tracks
            cv2.putText(image, str(track_id), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    return image

# - - - First version allowed for single image detection and "tracking"

def process_image(model, image_path, args):
    """Processes a single image."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return

    detections = detect_belugas(model, image)
    if detections is not None:
        # Create "tracks" for single-frame display (as before)
        tracks = [Track(track_id=i + 1, tlbr=[x1, y1, x2, y2]) for i, (x1, y1, x2, y2, _, _) in enumerate(detections)]
        image_with_boxes = draw_boxes(image, detections, tracks)
        cv2.imshow("Image Detection", image_with_boxes)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No detections found.")


# - - - Second version allowed for video detection and tracking

def process_video(model, video_path, args, output_path=None):
    """Processes a video (same as before, but uses 'args')."""
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
    parser = argparse.ArgumentParser(description="Beluga Whale Detection and Tracking Demo")
    parser.add_argument("mode", choices=['image', 'video'], help="Process an image or a video")
    parser.add_argument("path", help="Path to the image or video file")
    parser.add_argument("-o", "--output", help="Output video path (for video mode)", default=None)
    parser.add_argument("-t", "--track_thresh", type=float, default=0.5, help="Tracking confidence threshold")
    parser.add_argument("-n", "--num_queries", type=int, default=100, help="Number of queries for ByteTrack")

    args = parser.parse_args()

    model_path = 'models/yolov8n.pt'  # Or your trained model
    model = load_model(model_path)

    if model is None:
        print("Failed to load model.")
        exit()

    # Create a Namespace object for BYTETracker 
    tracker_args = argparse.Namespace()
    tracker_args.track_thresh = args.track_thresh
    tracker_args.num_queries = args.num_queries

    
    if args.mode == 'image':
        if not os.path.exists(args.path):
            print(f"Error: Image file not found at {args.path}")
            exit()
        process_image(model, args.path, tracker_args)
    elif args.mode == 'video':
        if not os.path.exists(args.path):
            print(f"Error: Video file not found at {args.path}")
            exit()
        process_video(model, args.path, tracker_args, args.output)





    # DEMO TESTS

    # Process the image
    # python -m src.main image data/test_images/test_image.jpg 

    # Process the video and display the output
    # python -m src.main video data/videos/test_video1.mp4

    # Process the video and save the output to output.mp4
    # python -m src.main video data/videos/test_video1.mp4 -o data/results/output.mp4