import cv2
import os

def extract_frames_from_all_videos(video_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Iterate over all files in the video folder
    for video_file in os.listdir(video_folder):
        video_path = os.path.join(video_folder, video_file)
        
        # Check if the file is a video file (you can add more extensions if needed)
        if video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            # Capture the video
            video = cv2.VideoCapture(video_path)
            success, image = video.read()
            count = 0

            while success:
                # Define the filename for each frame, using the video name to ensure uniqueness
                frame_filename = os.path.join(output_folder, f"{os.path.splitext(video_file)[0]}_frame_{count:04d}.jpg")
                
                # Save the current frame as a .jpg file
                cv2.imwrite(frame_filename, image)
                
                # Read the next frame
                success, image = video.read()
                count += 1

            video.release()
            print(f"Extracted {count} frames from {video_file} into {output_folder}")

# Example usage
video_folder = 'vid_grey'  # Replace with the path to your video folder
output_dir = 'C:/Users/Hp/21k4500/7thSemester/LIP/frameextract/FaceHighlight/face_highlight/frame_right_input'  # Replace with the desired output folder

extract_frames_from_all_videos(video_folder, output_dir)

