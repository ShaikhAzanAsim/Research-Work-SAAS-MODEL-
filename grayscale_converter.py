import cv2
import os

# Input and output directories
input_folder = 'C:/Users/Hp/21k4500/7thSemester/LIP/frameextract/vid'
output_folder = 'C:/Users/Hp/21k4500/7thSemester/LIP/frameextract/vid_grey'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Iterate over all files in the input folder
for video_file in os.listdir(input_folder):
    input_video_path = os.path.join(input_folder, video_file)
    
    # Check if the file is a video (you can add more extensions if needed)
    if video_file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        # Output video path
        output_video_path = os.path.join(output_folder, f'gray_{video_file}')

        # Open the input video
        cap = cv2.VideoCapture(input_video_path)

        # Get the video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 file format

        # Initialize the video writer for the grayscale output
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height), isColor=False)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the frame to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Write the grayscale frame to the output video
            out.write(gray_frame)

        # Release the video objects
        cap.release()
        out.release()

        print(f"Grayscale video saved: {output_video_path}")

print("All videos have been processed.")

