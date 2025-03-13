import pickle
import numpy as np

filenames = ["centroids.pkl", "cluster_0.pkl", "cluster_1.pkl", "cluster_2.pkl"]
new_dim = 384
for filename in filenames:
    with open(filename, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, np.ndarray) and data.shape[1] >= new_dim:
        truncated_data = data[:, :new_dim]

        with open(filename, "wb") as f:
            pickle.dump(truncated_data, f)

        print(f"Successfully overwritten {filename} with shape {truncated_data.shape}")
    else:
        print("Error: Data is not a NumPy array or has insufficient dimensions.")
