import cv2
import dlib
import numpy as np
import os

# Load the pre-trained facial landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Load the DNN face detector model
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')

def highlight_person(input_folder, output_folder, person_index):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for image_file in sorted(os.listdir(input_folder)):
        image_path = os.path.join(input_folder, image_file)
        image = cv2.imread(image_path)
        (h, w) = image.shape[:2]

        # Prepare the image for the DNN model
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        # Create a mask for the selected person
        mask = np.zeros_like(image)

        face_positions = []

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")
                face_positions.append((x, y, x1, y1))

        # Ensure there are exactly two faces detected
        if len(face_positions) == 2:
            # Sort faces based on the x-coordinate (left to right)
            face_positions.sort(key=lambda pos: pos[0])

            # Get the position of the face based on the person_index (0 for left, 1 for right)
            selected_face = face_positions[person_index]

            # Highlight the selected face
            (x, y, x1, y1) = selected_face
            mask[y:y1, x:x1] = image[y:y1, x:x1]
            cv2.rectangle(mask, (x, y), (x1, y1), (255, 255, 255), 2)

        # Save the processed image
        output_path = os.path.join(output_folder, image_file)
        cv2.imwrite(output_path, mask)
        print(f"Processed {image_file} and saved to {output_folder}")

# Example usage
input_dir = 'frame_right_input'  # Replace with the path to your input image folder
output_dir = 'highlighted_rightPerson'  # Replace with the desired output folder

person_index = 1  # Set to 0 for the left person, 1 for the right person
highlight_person(input_dir, output_dir, person_index)
