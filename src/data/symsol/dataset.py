# import tensorflow as tf
import tensorflow_datasets as tfds

import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt

class SymmetricSolidsDataset(Dataset):
    def __init__(self, split='train', transform=None):
        """
        Args:
            split (str): 'train', 'test'
            transform: 
        """
        self.transform = transform
        self.split = split

        # data_dir = '/home/pithreeone/Ben/Project/liepose_pytorch/dataset'
        data_dir = '../../../dataset'

        # Load the dataset from tensorflow_datasets (tfds)
        self.tf_dataset, self.info = tfds.load("symmetric_solids", split=self.split, with_info=True, data_dir=data_dir, as_supervised=True)
        
        self.iterator = iter(self.tf_dataset)


    def __len__(self):
        car = self.tf_dataset.cardinality()
        return car
        # return len(self.tf_dataset)
    
    def __getitem__(self, idx):
        # get the sample from the dataset
        image, label = next(self.iterator)
        # print(type(image), type(label))

        # Convert image and label to numpy arrays, and then to torch tensors
        image = torch.tensor(np.array(image), dtype=torch.float32)
        label = torch.tensor(np.array(label), dtype=torch.float32)

        if self.transform:
            image = self.transform(image)
        
        return image, label

def load_symmetric_solids_dataset(split='train', transform=None):
    return SymmetricSolidsDataset(split=split, transform=transform)

# Main function to test the dataset loader
def main():
    print("Loading dataset...")
    dataset = load_symmetric_solids_dataset(split='train')


    batch_size = 32
    train_loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

    print(len(dataset))
    # print(f"Dataset Info: {dataset.info}")
    print(f"Loading batches with batch size {batch_size}...")
    for i, (images, labels) in enumerate(train_loader):
        print(f"Batch {i}:")
        print(f"  Image batch shape: {images.shape}")
        print(f"  Label batch shape: {labels.shape}")

        # # Display the first image in the batch
        img = images[0].numpy().astype(np.uint8)  # Convert to NumPy array for visualization
        print(labels[0])
        # img = np.transpose(img, (1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
        plt.imshow(img)  # Display image
        plt.show()


        # # Test only the first few batches (e.g., 2)
        if i >= 2:
            break


if __name__ == "__main__":
    main()