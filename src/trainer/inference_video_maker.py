import cv2
import os
import re  # Import regular expression module

# Specify the directory where your images are located
image_folder = 'vis/video_result'
output_video = 'output_result.mp4'

# Get all image file names and extract the unique base names
images = []

# Loop through all files in the folder
for img in os.listdir(image_folder):
    if img.endswith(".png"):
        if '_color' in img:
            continue
        images.append(img)

# Sort the images numerically by extracting the numbers in the filenames
def extract_number(filename):
    # Use regular expression to find all numbers in the filename
    numbers = re.findall(r'\d+', filename)
    # Convert the first number found to integer (assuming filenames contain at least one number)
    return int(numbers[0]) if numbers else 0

# Sort images using the extracted number
images.sort(key=extract_number)

# Read the first image to get the dimensions (width and height)
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, _ = frame.shape

# Set up the video writer (make sure codec is available in your system)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use 'MJPG' or 'XVID' for .avi, or 'mp4v' for .mp4
video_writer = cv2.VideoWriter(output_video, fourcc, 30, (width, height))  # 30 fps

# Loop through each image, show it, and write it to the video
for image in images:
    img = cv2.imread(os.path.join(image_folder, image))
    
    # Print the filename
    print("Displaying:", image)
    
    # Show the image
    # cv2.imshow('Image', img)
    
    # # Wait for a key press to continue (use 1000 ms = 1 second)
    # if cv2.waitKey(0) & 0xFF == ord('q'):  # Press 'q' to quit
    #     break
    
    # Write the image to the video
    video_writer.write(img)

# Release the video writer and cleanup
video_writer.release()
cv2.destroyAllWindows()

# Optional: Check the video file is created
print("Video saved at:", output_video)
