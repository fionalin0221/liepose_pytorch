import cv2
import os

# Specify the directory where your images are located
image_folder = 'vis/video_data/cyl'
output_video = 'output_video.mp4'

# Get all image file names (ensure they're sorted properly)
# Get all image file names and extract the unique base names
images = []
base_names = set()

# Loop through all files in the folder
for img in os.listdir(image_folder):
    if img.endswith(".png"):
        if '_color' in img:
            continue
        images.append(img)
            # base_names.add(base_name)

images.sort()  # Optional: sorts images alphabetically or based on your naming pattern

# Read the first image to get the dimensions (width and height)
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, _ = frame.shape

# Set up the video writer (make sure codec is available in your system)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use 'MJPG' or 'XVID' for .avi, or 'mp4v' for .mp4
video_writer = cv2.VideoWriter(output_video, fourcc, 30, (width, height))  # 30 fps

# Loop through each image and write it to the video
for image in images:
    img = cv2.imread(os.path.join(image_folder, image))
    video_writer.write(img)

# Release the video writer and cleanup
video_writer.release()

# Optional: Check the video file is created
print("Video saved at:", output_video)
