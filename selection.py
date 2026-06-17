import pandas as pd
import numpy as np
import json

# Load the clustering results
with open('output/final_clusters.json', 'r') as f:
    first_level_results = json.load(f)

# Extract the cluster labels
# Assuming the clustering results are in dictionary form, with keys as indices and values as cluster IDs
first_level_labels = np.array([first_level_results[str(i)] for i in range(len(first_level_results))])

# Load the data
loss_df = pd.read_csv('info/loss_alpaca.csv')

# Load the instruction data
with open('datasets/alpaca_data.json', 'r') as f:
    alpaca_data = json.load(f)

# Ensure the data is sorted by 'step'
loss_df = loss_df.sort_values(by='step')

# Extract the loss values for epoch0 and epoch1
loss_data = loss_df[['epoch0', 'epoch1']].values

# Calculate the ratio of loss values for epoch0 and epoch1
loss_ratios = loss_data[:, 1] / loss_data[:, 0]

# Initialize a set to store the final selected instruction indices
selected_indices = set()

# Calculate the number of samples in each cluster
cluster_sizes = np.bincount(first_level_labels)

# Calculate the total number of samples
total_samples = len(first_level_labels)

# Calculate the number of samples to select from each cluster
total_to_select = 3000
cluster_selections = (cluster_sizes / total_samples * total_to_select).astype(int)

# Select the instructions with the highest loss ratio for epoch0 and epoch1 within each cluster
for cluster_id in range(len(cluster_sizes)):
    # Extract the indices of the current cluster
    cluster_indices = np.where(first_level_labels == cluster_id)[0]
    if len(cluster_indices) == 0:
        continue
    
    # Extract the loss ratios for the current cluster
    cluster_loss_ratios = loss_ratios[cluster_indices]
    
    # Get the indices of the top samples based on the loss ratio
    sorted_indices = np.argsort(-cluster_loss_ratios)[:cluster_selections[cluster_id]]
    
    # Map these indices back to the original data indices
    selected_indices.update(cluster_indices[sorted_indices])

# If the number of selected samples is less than total_to_select, select additional samples from the remaining ones
if len(selected_indices) < total_to_select:
    remaining_indices = set(range(len(alpaca_data))) - selected_indices
    remaining_loss_ratios = loss_ratios[list(remaining_indices)]
    # Get the indices of the remaining samples with the lowest loss ratio
    additional_indices = np.argsort(remaining_loss_ratios)[:total_to_select - len(selected_indices)]
    selected_indices.update(list(remaining_indices)[i] for i in additional_indices)

# Select the corresponding samples from the original instruction data
selected_instructions = [alpaca_data[i] for i in selected_indices]

# Save the selected instructions to a JSON file
with open('alpaca_5%.json', 'w') as f:
    json.dump(selected_instructions, f, indent=4)