import torch
# torch.load("mean.pt").numpy().reshape(-1, 1, 1)
torch.load("std.pt").numpy().reshape(-1, 1, 1)

# from torch.utils.data import DataLoader

# # Create DataLoader for full dataset to calculate mean and std
# batch_size = 1  # Load one file at a time
# full_dataloader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)

# def calculate_mean_and_std(dataset_loader):
#     total_sum = 0
#     total_sum_of_squares = 0
#     num_samples = 0

#     for train_data, _ in dataset_loader:
#         # train_data shape: (1, 45, 48, 48) since batch_size = 1
#         train_data = train_data.squeeze(0)  # Shape: (45, 48, 48)
#         num_elements_per_layer = train_data.shape[1] * train_data.shape[2]

#         # Sum and sum of squares for each layer
#         total_sum += train_data.sum(dim=(1, 2))  # Shape: (45,)
#         total_sum_of_squares += (train_data ** 2).sum(dim=(1, 2))  # Shape: (45,)
#         num_samples += 1
#         print(num_samples)

#     # Calculate mean and std for each layer
#     mean = total_sum / (num_samples * num_elements_per_layer)
#     mean_of_squares = total_sum_of_squares / (num_samples * num_elements_per_layer)
#     std = torch.sqrt(mean_of_squares - mean ** 2)

#     return mean, std

# # Calculate mean and std across the entire dataset
# mean, std = calculate_mean_and_std(full_dataloader)

# print("Mean for each layer:", mean)
# print("Std for each layer:", std)

# # Save mean and std for later use
# mean_path = os.path.join(data_dir, "mean.pt")
# std_path = os.path.join(data_dir, "std.pt")
# torch.save(mean, mean_path)
# torch.save(std, std_path)