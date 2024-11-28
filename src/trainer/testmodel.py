from ..model import Model
import torch
import torch.nn as nn
import torch.optim as optim

# Example synthetic data generator
def generate_synthetic_data(batch_size=16):
    n_slices = 10
    images = torch.rand(batch_size, 3, 224, 224)  # Random images
    t_values = torch.randint(0, 10, (batch_size * n_slices,)).unsqueeze(1)  # Random integers for t
    input_poses = torch.rand(batch_size * n_slices, 3)  # Random 3D poses
    # print(input_poses.shape, t_values.shape)
    output_poses = input_poses * 2 + t_values * 0.1  # Example transformation
    return images, t_values, input_poses, output_poses


# Initialize model, optimizer, and loss function
model = Model(in_dim = 3,
                    out_dim = 3,
                    image_shape = [1, 3, 224, 224],
                    resnet_depth = 34,
                    mlp_layers = 1,
                    fourier_block = True,
                    activ_fn = "leaky_relu"
                    )

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# Training loop
for epoch in range(1000):  # Small number of epochs for testing
    images, t_values, input_poses, target_poses = generate_synthetic_data()
    print(images.shape, t_values.shape, input_poses.shape, target_poses.shape)
    optimizer.zero_grad()
    predictions = model(images, input_poses, t_values)
    loss = criterion(predictions, target_poses)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# Testing the model
images, t_values, input_poses, target_poses = generate_synthetic_data(batch_size=4)
with torch.no_grad():
    predictions = model(images, t_values, input_poses)
    print("Input Poses:", input_poses)
    print("Predicted Poses:", predictions)
    print("Target Poses:", target_poses)