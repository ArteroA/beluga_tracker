# Beluga Whale Detection and Tracking

This project aims to detect and track beluga whales in video footage. It takes a video as input and creates a new video with boxes around detected objects and lines showing where they've moved.

## Project Status

**What it can do now:**

*   **Video Input:** Works with `.mp4` video files.
*   **Detection:** Uses a YOLOv8 model to find objects in each frame.
*   **Tracking:** Uses ByteTrack to follow objects as they move between frames.
*   **Output:** Makes a new video with boxes and tracking IDs.

**What it can't do well yet (Limitations):**

*   **Beluga Whale Accuracy:** Right now, it uses a *general* object detection model (YOLOv8). It's **not** trained specifically for beluga whales and calves. This means:
    *   It might miss some belugas.
    *   It might identify other things as belugas.
    *   It won't be able to tell the difference between adult belugas and calves.
* **Speed**: It may take some time to run, depending on the computer and the size of the video
* **Different Environments**: The detection and tracking might not work as well in videos with:
    *   Very dark or murky water.
    *   Lots of glare or reflections.
    *   Crowds of many belugas close together.
    *   Belugas that are very far away or very small in the video.
* **Occlusion:** If a beluga goes behind something (another whale, an object, etc.) and comes out, the tracker might think it's a new whale.

**What we're working on (Future Work):**

*   **Better Beluga Detection:** We're working on training a new model that's *specifically* for beluga whales and calves. This will make the detection much more accurate.
    **Faster Processing:**  Make the program run faster.
    **User Options:** Add settings to let users change things like how sensitive the detection is. Change the thresholds and confidence levels
    **Support Other Video Formats:** Allow the program to process videos in different file types like .avi or .mov.

## Installation and Setup

1.  **Clone the Repository:**

    ```bash
    git clone <your_repository_url>
    cd <repository_name>  # Usually beluga_tracker1
    ```
    Replace `<your_repository_url>` with your repository's URL.

2.  **Create a Virtual Environment (Important!):**

    ```bash
    python3 -m venv .venv
    ```

3.  **Activate the Virtual Environment:**

    *   **Windows:**
        ```bash
        .venv\Scripts\activate
        ```

    *   **Linux/macOS:**
        ```bash
        source .venv/bin/activate
        ```

4.  **Install PyTorch (Very Important!):**

    *   Go to the PyTorch website: [https://pytorch.org/](https://pytorch.org/)
    *   Select your computer's setup (operating system, etc.).
    *   The website will give you a command to install PyTorch.  Use that command!  It's important to get the right version.

        ```bash
        # Example (for CUDA 11.8 - CHANGE THIS TO MATCH YOUR SYSTEM!):
        pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

        # Example for CPU-only:
        # pip install torch torchvision torchaudio
        ```

5.  **Install Other Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

```bash
python src/main.py --input <input_video.mp4> --output <output_video.mp4>