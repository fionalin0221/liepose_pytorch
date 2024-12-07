from PIL import Image
import os

# Specify the folder containing PNG files and the output GIF file name
folder_path = "vis/video_result"  # Replace with your folder path
output_gif = "output.gif"

# Get all PNG files in the folder, sorted alphabetically
png_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".png")])

# Create a list to hold the images
images = []

# Read each PNG file and append it to the list
for png_file in png_files:
    image_path = os.path.join(folder_path, png_file)
    images.append(Image.open(image_path))

# Save as a GIF
images[0].save(
    output_gif,
    save_all=True,
    append_images=images[1:],
    duration=100,  # Duration for each frame in milliseconds
    loop=0         # 0 means the GIF will loop indefinitely
)

print(f"GIF saved as {output_gif}")
