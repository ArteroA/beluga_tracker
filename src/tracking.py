from src.bytetrack.byte_tracker import BYTETracker
import torch  # Import PyTorch

class Track:
    """Represents a single tracked object (beluga whale)."""

    def __init__(self, track_id, tlbr):
        """Initializes a Track object.

        Args:
            track_id (int): The unique ID assigned to the track.
            tlbr (list): Top-left-bottom-right bounding box coordinates: [x1, y1, x2, y2].
        """
        self.track_id = track_id
        self.tlbr = tlbr  # Store as top-left-bottom-right


def track_belugas(detections_across_frames):
    """Tracks beluga whales across multiple frames using ByteTrack.

    Args:
        detections_across_frames (list): A list of lists.  Each inner list
            contains the detections for a single frame (output of detect_belugas).

    Returns:
        list: A list of Track objects, representing the tracked whales in the *current* frame.
              Returns an empty list if there are no detections.
    """
    tracker = BYTETracker()  # Initialize a BYTETracker object
    all_tracks = []

    for frame_detections in detections_across_frames:
        # Prepare input for ByteTrack.  It expects a list of detections in the
        # format [x1, y1, x2, y2, confidence].
        bytetrack_inputs = []
        for x1, y1, x2, y2, conf, cls in frame_detections:
            bytetrack_inputs.append([x1, y1, x2, y2, conf])

        # Update the tracker with the current frame's detections.
        # online_targets is a list of STrack objects (ByteTrack's internal representation).
        online_targets = tracker.update(torch.tensor(bytetrack_inputs))

        # Extract track information from the STrack objects.
        for t in online_targets:
            tlwh = t.tlwh  # Top-left x, y, width, height
            tid = t.track_id  # The unique track ID
            tlbr = [tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]]  # Convert to tlbr
            all_tracks.append(Track(track_id=tid, tlbr=tlbr))

    return all_tracks