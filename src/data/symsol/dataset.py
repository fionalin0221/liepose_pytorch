import tensorflow as tf
import tensorflow_datasets as tfds

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

import json
from PIL import Image
from tqdm import tqdm
import os

class SymmetricSolidsDataset(Dataset):
    def __init__(self, split='train', transform=None):
        """
        Args:
            split (str): 'train', 'test'
            transform: 
        """
        self.transform = transform
        self.split = split

        # Directory to save the dataset
        save_dir = os.path.join("dataset/symmetric_solids_dataset_customed", split)
        metadata_file = os.path.join(save_dir, "metadata.json")

        if os.path.exists(save_dir) and os.path.exists(metadata_file):
        # if False:
            print(f"Symsol dataset has already convert from tensorflow")
            with open(metadata_file, "r") as f:
                self.metadata = json.load(f)

        else:
            print(f"Create customed symsol dataset...")

            # data_dir = '/home/pithreeone/Ben/Project/liepose_pytorch/dataset'
            data_dir = '../../../dataset'

            # Load the dataset from tensorflow_datasets (tfds)
            shapes = ["tet", "cube", "icosa", "cone", "cyl"]
            self.tf_dataset, self.info = tfds.load("symmetric_solids", split=self.split, with_info=True, data_dir=data_dir, as_supervised=True)
            # self.tf_dataset = self.tf_dataset.filter(lambda x: tf.reduce_any(tf.equal(x["label_shape"], shapes)))
            # Directory to save the dataset
            os.makedirs(save_dir, exist_ok=True)
            os.makedirs(os.path.join(save_dir, "images"), exist_ok=True)

            self.metadata = []

            # Iterate over dataset and save images/gt_label
            for idx, (image, label) in tqdm(enumerate(self.tf_dataset.as_numpy_iterator())):
            # for idx, data in tqdm(enumerate(self.tf_dataset.as_numpy_iterator())):                
                # Convert to numpy arrays
                image = np.array(image)        # Image as Numpy array
                label_matrix = label.tolist()  # Poise matrix as a nested list

                # Save the image as a PNG file
                image_path = os.path.join(save_dir, "images", f"{idx}.png")
                # Image.fromarray(np.array(image)).save(image_path)
                
                # Append metadata
                self.metadata.append({"image_path": image_path, "label": label_matrix})

            # Save metadata to a JSON file
            with open(os.path.join(save_dir, "metadata.json"), "w") as f:
                json.dump(self.metadata, f, indent=4)

            print("Dataset saved successfully!")

        self.length = len(self.metadata)
        # self.iterator = iter(self.tf_dataset)


    def __len__(self):
        return self.length
        # return len(self.tf_dataset)
    
    def __getitem__(self, idx):
        # get the sample from the dataset
        # image, label = next(self.iterator)
        # print(type(image), type(label))

        entry = self.metadata[idx]
        image_path = entry["image_path"]
        label = torch.tensor(entry["label"], dtype=torch.float32)

        # Load the image with PIL-library and convert it to a numpy array
        image = Image.open(image_path).convert("RGB")  # H, W, C
        image = np.array(image)
        # image = np.transpose(np.array(image), (2, 0, 1))

        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        
        return image, label

def load_symmetric_solids_dataset(split='train', transform=None):
    return SymmetricSolidsDataset(split=split, transform=transform)

# Main function to test the dataset loader
def main():
    print("Loading dataset...")

    dataset = load_symmetric_solids_dataset(split='train', transform=None)

    batch_size = 256
    train_loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

    print(len(dataset))
    # print(f"Dataset Info: {dataset.info}")
    print(f"Loading batches with batch size {batch_size}...")
    for i, (images, labels) in enumerate(train_loader):
        print(f"Batch {i}:")
        print(f"  Image batch shape: {images.shape}")
        print(f"  Label batch shape: {labels.shape}")

        # # Display the first image in the batch
        img = images[0].numpy()
        # img = images[0].numpy().astype(np.uint8)  # Convert to NumPy array for visualization
        # img = (img - img.min()) / (img.max() - img.min()) * 255
        print(labels[0])
        print(img.shape)
        # img = np.transpose(img, (1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
        # img = img[..., ::-1]
        plt.imshow(img)  # Display image
        plt.show()

        # print(i)
        # # Test only the first few batches (e.g., 2)
        # if i >= 2:
        #     break


if __name__ == "__main__":
    main()