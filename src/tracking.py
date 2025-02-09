from src.bytetrack.byte_tracker import BYTETracker
import torch
import numpy as np

class Track:
    """Represents a single tracked object."""
    def __init__(self, track_id, tlbr):
        self.track_id = track_id
        self.tlbr = tlbr

class TensorDictWrapper:
    """Wraps a tensor to make it look like a dictionary to ByteTrack."""
    def __init__(self, tensor):
        self.tensor = tensor

    def __getitem__(self, key):
        if key == "scores":
            return self.tensor[:, 4]  # 5th element is confidence
        elif key == "boxes":
            return self.tensor[:, :4]  # First 4 elements are bbox
        elif key == "labels":
            # ByteTrack might access this, even though it doesn't use it.
            return torch.zeros_like(self.tensor[:, 4], dtype=torch.int64)
        else:
            raise KeyError(f"Invalid key: {key}")

    def cpu(self):
        return self  # Already on CPU (or handle device transfer)
    def numpy(self):
      return self.tensor.numpy()

    def to(self, device):
        return TensorDictWrapper(self.tensor.to(device))

def track_belugas(detections, tracker):
    """Tracks beluga whales using ByteTrack."""

    all_tracks = []
    bytetrack_inputs = []

    for x1, y1, x2, y2, conf, cls in detections:
        bytetrack_inputs.append([x1, y1, x2, y2, conf])

    if not bytetrack_inputs:
        return []

    bytetrack_inputs = torch.tensor(bytetrack_inputs)
    if bytetrack_inputs.ndim == 1:
        bytetrack_inputs = bytetrack_inputs.unsqueeze(0)

    if bytetrack_inputs.numel() > 0:
        wrapped_input = TensorDictWrapper(bytetrack_inputs)
        online_targets = tracker.update(wrapped_input)
        # print("online_targets:", online_targets)   <-- Keep this for debugging if needed
    else:
        online_targets = []

    for t in online_targets:
        
        bbox = t["bbox"]      # Get the bbox array
        track_id = t["tracking_id"]  # Get the tracking ID
        x1, y1, x2, y2 = bbox  # Unpack the bbox coordinates
        w = x2 - x1           # Calculate width
        h = y2 - y1           # Calculate height
        tlwh = [x1, y1, w, h]  # Create tlwh format
        tlbr = [x1, y1, x2, y2]
        all_tracks.append(Track(track_id=track_id, tlbr=tlbr))

    return all_tracks
