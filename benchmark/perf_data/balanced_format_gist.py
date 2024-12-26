import numpy as np
import subprocess
import argparse
import os
import struct
import sys
from collections import Counter
import pickle


def check_and_clean_folder(data_dir):
    # Check if 'gist_base.fvecs' exists in the directory
    gist_base_fvecs_path = os.path.join(data_dir, 'gist_base.fvecs')
    if not os.path.isfile(gist_base_fvecs_path):
        print(f"Error: 'gist_base.fvecs' not found in {data_dir}.")
        return
    # Clean the directory of all files except '.fvec' file
    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        if os.path.isfile(file_path) and not filename.endswith('.fvecs'):
            os.remove(file_path)  
            print(f"Removed: {file_path}")

def fvecs_to_fbin_with_metadata(gist_data_directory):
    input_file = os.path.join(gist_data_directory, 'gist_base.fvecs')
    output_file = os.path.join(gist_data_directory, 'gist_base.fbin')
    with open(input_file, "rb") as f:
        data = np.fromfile(f, dtype=np.float32)
    dim = int(data.view(np.int32)[0])
    assert dim > 0, "Invalid dimension read from file."
    
    data = data.reshape(-1, 1 + dim)
    if not np.all(data[:, 0].view(np.int32) == dim):
        raise IOError(f"Non-uniform vector sizes in {input_file}")
    
    n = data.shape[0]
    vectors = data[:, 1:]
    with open(output_file, "wb") as f:
        f.write(np.array([n, dim], dtype=np.uint32).tobytes())  # Write n and d
        vectors.tofile(f)
    print(f"Converted {input_file} to {output_file} with metadata (n={n}, d={dim})")



def run_balanced_knn(gp_ann_directory, gist_directory, num_clusters):
    gist_base_fbin = os.path.join(gist_directory, 'gist_base.fbin')
    gist_partition = os.path.join(gist_directory, 'gist.partition')
    partition_executable = os.path.join(gp_ann_directory, 'release_l2', 'Partition')
    
    command = [
        partition_executable, 
        gist_base_fbin, 
        gist_partition, 
        str(num_clusters),  
        'BalancedKMeans', 
        'default'
    ]
    
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
        for stdout_line in process.stdout:
            print(stdout_line, end='')  
        for stderr_line in process.stderr:
            print(stderr_line, end='')  

        return_code = process.wait()
        if return_code != 0:
            print(f"Error: Command returned non-zero exit code {return_code}")
        else:
            print("Command executed successfully.")



def load_partition(filepath):
    with open(filepath, 'rb') as f:
        n = struct.unpack('I', f.read(4))[0]  # 'I' means an unsigned 32-bit integer
        partition_result = struct.unpack(f'{n}i', f.read(n * 4))  # 'i' means signed 32-bit integers
        return partition_result, n


def read_embeddings(embeddings_path):
    with open(embeddings_path, 'rb') as f:
        n, d = np.frombuffer(f.read(8), dtype=np.uint32)
        embeddings = np.frombuffer(f.read(n * d * 4), dtype=np.float32)
        embeddings = embeddings.reshape((n, d))
        return embeddings, n, d


def write_cluster_embeddings(cluster_id, embeddings, partition_result, output_dir):
    cluster_embeddings = embeddings[partition_result == cluster_id]
    cluster_filepath_dat = os.path.join(output_dir, f'cluster_{cluster_id}.dat')
    cluster_filepath_pkl = os.path.join(output_dir, f'cluster_{cluster_id}.pkl')
    # Write the selected embeddings to a binary .dat file
    with open(cluster_filepath_dat, 'wb') as f:
        n = cluster_embeddings.shape[0]
        d = cluster_embeddings.shape[1]
        f.write(np.uint32(n).tobytes())
        f.write(np.uint32(d).tobytes())
        f.write(cluster_embeddings.tobytes())
    print(f"Cluster {cluster_id} saved to {cluster_filepath_dat} with {n} points.")
    
    # Write the selected embeddings to a pickle .pkl file
    with open(cluster_filepath_pkl, 'wb') as f:
        pickle.dump(cluster_embeddings, f)
    print(f"Cluster {cluster_id} saved to {cluster_filepath_pkl} with {n} points.")


def process_cluster_results(data_dir):
    partition_filepath = os.path.join(data_dir, 'gist.partition.dat')
    embeddings_path = os.path.join(data_dir, 'gist_base.fbin')
    output_dir = data_dir
    # Load partition results
    partition_result, n = load_partition(partition_filepath)
    print(f"Loaded partition with {n} points.")
    # Load embeddings
    embeddings, _, _ = read_embeddings(embeddings_path)
    print(f"Loaded {embeddings.shape[0]} embeddings with {embeddings.shape[1]} dimensions.")
    # Write cluster embeddings to files
    centroid_counts = Counter(partition_result)
    for centroid in centroid_counts.keys():
        write_cluster_embeddings(centroid, embeddings, partition_result, output_dir)


def load_centroids(filepath):
    with open(filepath, 'rb') as f:
        n, d = struct.unpack('2I', f.read(8))  
        centroids = []
        for _ in range(n):
            centroid = struct.unpack(f'{d}f', f.read(d * 4)) 
            centroids.append(centroid)
        return centroids, n, d


def save_centroids_to_pkl(centroids, filedir):
    filepath = os.path.join(filedir, "centroids.pkl")
    with open(filepath, 'wb') as f:
        pickle.dump(centroids, f)
    print(f"Centroids saved to {filepath}")


def process_centroid_results(data_dir):
    centroid_path = os.path.join(data_dir, "gist.partition_centroids.dat")
    centroids, n, d = load_centroids(centroid_path)
    print(f"Loaded {n} centroids with {d} dimensions.")
    save_centroids_to_pkl(centroids, data_dir)



def main():
    parser = argparse.ArgumentParser(description='Run Partition command with custom directories.')
    parser.add_argument('gp_ann_directory', type=str, help='Directory containing the gp_ann code (e.g., ../gp_ann)')
    parser.add_argument('gist_directory', type=str, help='Directory containing the gist files (e.g., ../gist)')
    parser.add_argument('--num_clusters', type=int, default=10, help='Number of clusters (default: 10)')
    args = parser.parse_args()

    check_and_clean_folder(args.gist_directory)

    # 1. Convert gist_base.fvecs to gist_base.fbin
    fvecs_to_fbin_with_metadata(args.gist_directory)
    # 2. Run Balanced KMeans
    run_balanced_knn(args.gp_ann_directory, args.gist_directory, args.num_clusters)
    # 3. write results to pkls
    process_cluster_results(args.gist_directory)
    process_centroid_results(args.gist_directory)


if __name__ == "__main__":
    main()