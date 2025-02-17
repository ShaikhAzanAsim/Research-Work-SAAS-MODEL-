import cv2
import os
import numpy as np

def segment_images(input_folder, output_folder, clusters=18):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over all files in the input folder
    for image_file in os.listdir(input_folder):
        image_path = os.path.join(input_folder, image_file)
        
        # Check if the file is an image (you can add more extensions if needed)
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Read the image
            image = cv2.imread(image_path)
            
            # Convert the image to a 2D array of pixels
            pixel_values = image.reshape((-1, 3))
            pixel_values = np.float32(pixel_values)
            
            # Define the criteria and perform K-means clustering
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
            _, labels, centers = cv2.kmeans(pixel_values, clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # Convert centers to uint8 and labels back to the image
            centers = np.uint8(centers)
            segmented_image = centers[labels.flatten()]
            segmented_image = segmented_image.reshape(image.shape)
            
            # Convert the segmented image to grayscale
            grayscale_segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
            
            # Save the grayscale segmented image
            segmented_filename = os.path.join(output_folder, f"segmented_{image_file}")
            cv2.imwrite(segmented_filename, grayscale_segmented_image)
            
            print(f"Segmented and converted {image_file} to grayscale with {clusters} clusters")

# Example usage
input_dir = 'img1'  # Replace with the path to your input image folder
output_dir = 'seg2'  # Replace with the desired output folder

segment_images(input_dir, output_dir, clusters=18)