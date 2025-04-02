#!/usr/bin/env python3

import argparse
import csv
import json
import os
import sys
import time
from derecho.cascade.external_client import ServiceClientAPI
import numpy as np



SUBGROUP_TYPES = {
        "VCSS": "VolatileCascadeStoreWithStringKey",
        "PCSS": "PersistentCascadeStoreWithStringKey",
        "TCSS": "TriggerCascadeNoStoreWithStringKey"
        }

NUM_BATCHES = 50
MAX_BATCH_SIZE = 1 # TODO: fix this

EMB_DIM = 384

UDL1_PATH = "/rag/emb/encode"
# UDL1_PATH = "/pipeline2_mon"

QUERY_FILENAME = "query.csv"


def utf8_length(s: str) -> int:
    """Computes the length of a UTF-8 encoded string without actually encoding it."""
    return sum(1 + (ord(c) >= 0x80) + (ord(c) >= 0x800) + (ord(c) >= 0x10000) for c in s)


def read_csv_to_list(data_dir: str):
    file_path = os.path.join(data_dir, QUERY_FILENAME) 
    query_list = []

    with open(file_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if row:  
                query_list.append(row[0]) 

    return query_list


def serialize(queries, query_ids, emb_size):
    """
    Serializes the queries into a NumPy buffer.

    :param queries: List of query strings
    :param query_ids: List of query IDs (uint64_t)
    :param emb_size: Size of each embedding (int)
    :return: NumPy buffer containing serialized data
    """
    HEADER_SIZE = 8  # 2 * uint32_t (num_queries, embeddings_position)
    METADATA_SIZE = 28  # query_id (8 bytes) + 5 * uint32_t (20 bytes)

    num_queries = len(queries)
    text_size_mapping = {}
    total_text_size = 0
    # print(f"queries {queries}")

    # Compute sizes
    for i, query in enumerate(queries):
        text_size = utf8_length(query)  # Get byte size of string
        total_text_size += text_size
        text_size_mapping[query_ids[i]] = text_size

    total_obj_size = HEADER_SIZE + METADATA_SIZE * num_queries + total_text_size

    metadata_position = HEADER_SIZE
    text_position = metadata_position + (num_queries * METADATA_SIZE)
    embeddings_position = text_position + total_text_size

    # Initialize buffer as a NumPy array of bytes
    buffer = np.zeros(total_obj_size, dtype=np.uint8)

    # Write the header (num_queries, embeddings_position)
    buffer[:4] = np.frombuffer(np.array([num_queries], dtype=np.uint32).tobytes(), dtype=np.uint8)
    buffer[4:8] = np.frombuffer(np.array([embeddings_position], dtype=np.uint32).tobytes(), dtype=np.uint8)

    # Offsets
    metadata_ptr_offset = metadata_position
    text_ptr_offset = text_position
    embedding_ptr_offset = embeddings_position

    for i, query in enumerate(queries):
        query_id = query_ids[i]
        client_id = i  # Assuming client_id is index-based (modify if needed)
        text_bytes = np.frombuffer(query.encode("utf-8"), dtype=np.uint8)
        text_len = text_size_mapping[query_id]

        # Metadata: query_id (8 bytes) + 5x uint32_t (20 bytes)
        query_id_bytes = np.frombuffer(np.array([query_id], dtype=np.uint64).tobytes(), dtype=np.uint8)
        metadata_array = np.frombuffer(
            np.array([client_id, text_ptr_offset, text_len, embedding_ptr_offset, emb_size], dtype=np.uint32).tobytes(),
            dtype=np.uint8
        )

        buffer[metadata_ptr_offset : metadata_ptr_offset + 8] = query_id_bytes  # query_id (uint64_t)
        buffer[metadata_ptr_offset + 8 : metadata_ptr_offset + 8 + 20] = metadata_array  # 5x uint32_t

        buffer[text_ptr_offset : text_ptr_offset + text_len] = text_bytes

        # Update offsets
        metadata_ptr_offset += METADATA_SIZE
        text_ptr_offset += text_len
        embedding_ptr_offset += emb_size

    return buffer

def main(argv):
    print("Connecting to Cascade service ...")   

    parser = argparse.ArgumentParser(description="Set up and connect to the Cascade service.")
    parser.add_argument('-p', '--path', required=True, type=str, help="Path to the data folder.")

    # Parse arguments
    args = parser.parse_args()

    # Access parsed arguments
    data_dir = args.path

    print("Connecting to Cascade service ...")
    capi = ServiceClientAPI()
    my_id = capi.get_my_id()

    query_list = read_csv_to_list(data_dir)
    total_query_count = len(query_list)
    query_id = 0
    for i in range(NUM_BATCHES):
        querybatch_id = i
        key = UDL1_PATH + "/" +str(my_id) + "_" + str(querybatch_id)
        
        queries = []
        queryid_list = []
        for j in range(MAX_BATCH_SIZE):
            idx = query_id % total_query_count
            queries.append(query_list[idx])
            print(f"Put query: {idx}")
            query_id += 1
            queryid_list.append(query_id)
        serialized_requests = serialize(queries, queryid_list, EMB_DIM)
        
        _subgroup_type = SUBGROUP_TYPES["VCSS"]
        capi.put_nparray(key, serialized_requests, 
                # subgroup_type=_subgroup_type,
                # subgroup_index=0,shard_index=0, 
                trigger=True, # Use trigger put
                message_id=i)
        time.sleep(25)
        print(f"Put key:{key} to Cascade.")
    

if __name__ == "__main__":
    main(sys.argv)
