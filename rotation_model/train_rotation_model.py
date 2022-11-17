# Rotation model build with Pytorch, based on a Keras Model:
# https://github.com/shivaverma/Angle-Detector/

import torch
import rotation_model as model
import load_data
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# parameters
epochs = 10
learning_rate = 0.001
img_size = 128
batch_size = 32
train_share = 0.9
model_path = "../data/rotation_model/rotation_model.pt"
img_path = "../data/coa_renamed"
angle_data_path = "../data/coa_rotation_angle_rounded-dict.csv"
random_seed = 42

# Set device
if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    # elif torch.backends.mps.is_available():
    #    device = "mps"
    else:
        device = "cpu"

# Load data
full_dataset = load_data.FolderDataset(img_path, angle_data_path, img_size)

train_size = int(train_share * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset,
                                                           [train_size, val_size],
                                                           generator=torch.Generator().manual_seed(random_seed))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# Declare model
model = model.Model().to(device)

# Add loss function
loss_fn = nn.MSELoss()
optimizer = optim.AdamW(list(model.parameters()), lr=learning_rate)
max_loss = 9999



print("Training started ------------")
for epoch in tqdm(range(epochs)):
    model.train()

    for batch_idx, (img, img_angle) in enumerate(train_loader):
        img = img.to(device)
        img_angle = img_angle.to(device)

        # Input img_batch into model
        model.train()
        optimizer.zero_grad()
        predicted_angle = model(img)

        # Get loss & optimizer step
        loss = loss_fn(predicted_angle, img_angle.to(torch.float32))
        loss.backward()
        optimizer.step()

    train_loss = loss.item()

    # Validation step
    model.eval()

    with torch.no_grad():
        for batch_idx, (img, img_angle) in enumerate(val_loader):
            img = img.to(device)
            img_angle = img_angle.to(device)

            predicted_angle = model(img)
            val_loss_batch = loss_fn(predicted_angle, img_angle.to(torch.float32))

    val_loss = val_loss_batch.item()

    # Simple Best Model saving
    if val_loss < max_loss:
        max_loss = val_loss
        print(f"Validation Loss decreased, saving new best model from epoch {epoch}")
        torch.save(model.state_dict(), model_path)

    print(f"Epochs = {epoch}, Training Loss : {train_loss}, Validation Loss: {val_loss}")

