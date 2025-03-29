#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import os, sys
import struct
import torch
from easydict import EasyDict
from derecho.cascade.external_client import ServiceClientAPI
from derecho.cascade.external_client import TimestampLogger
from transformers import AutoImageProcessor
from PIL import Image
from flmr import (
    FLMRConfig,
    FLMRQueryEncoderTokenizer,
)
from datasets import load_dataset
import time
from serialize_utils import PixelValueBatcher, TextDataBatcher
from torch.utils.data import DataLoader
from preprocess import *
# import faiss

image_root_dir = "/mydata/EVQA/"
ds_dir = "/mydata/EVQA/EVQA_data/"
STEPA_SHARD_INDICES = [2]
STEPB_SHARD_INDICES = [0, 1]
STEPA_SUBGROUP_INDEX = 0
STEPB_SUBGROUP_INDEX = 0

    
        
if __name__ == "__main__":
    tl = TimestampLogger()
    capi = ServiceClientAPI()
    stepa_prefix = "/stepA/"
    stepb_prefix = "/stepB/"
    subgroup_type = "VolatileCascadeStoreWithStringKey"
    
    BS = 1
    num_batches = 1000
    
    # directories and str configs
    image_processor_name = 'openai/clip-vit-large-patch14'
    checkpoint_path = 'LinWeizheDragon/PreFLMR_ViT-L'
    
    use_split = "train"
    
    # model configs, tokenziers
    flmr_config = FLMRConfig.from_pretrained(checkpoint_path)
    query_tokenizer = FLMRQueryEncoderTokenizer.from_pretrained(checkpoint_path,
                                                                    text_config=flmr_config.text_config,
                                                                    subfolder="query_tokenizer")
    image_processor = AutoImageProcessor.from_pretrained(image_processor_name)
    
    # TODO: change to actual range at perf test
    ds = load_dataset('parquet', data_files ={  
                                            'train' : ds_dir + 'train-00000-of-00001.parquet',
                                            'test'  : ds_dir + 'test-00000-of-00001-2.parquet',
                                            })[use_split].select(i for i in range(166000, 167000, 1)) 
    # preprocess datasets so that we have 
    ds = ds.map(add_path_prefix_in_img_path, fn_kwargs={"prefix": image_root_dir})
    ds = ds.map(prepare_text_sequence)
    ds = ds.map(
        tokenize_inputs,
        fn_kwargs={"query_tokenizer": query_tokenizer, "image_processor": image_processor},
        batched=True,
        batch_size=16,
        num_proc=16,
    )
    ds.set_format(
        type="torch", 
        columns=["input_ids", "attention_mask", "pixel_values", "text_sequence", "question_id", "question"]
    )


    # Create a DataLoader for sequential access with prefetching
    loader = DataLoader(
        ds, 
        batch_size=BS, 
        shuffle=False, 
        num_workers=16,      # Use multiple workers to prefetch batches in parallel
        prefetch_factor=2,   # How many batches each worker preloads (can adjust based on your system)
        pin_memory=True      # Optionally, if you are transferring to GPU later
    )

        
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= num_batches:
            break
  
        stepa_serializer = TextDataBatcher()
        for qid in batch["question_id"]:
            uds_idx =  int(qid.find("_"))
            question_id = int(qid[uds_idx+1:])
            stepa_serializer.question_ids.append(question_id)
            tl.log(1000, question_id, 0, 0)
        
        stepa_serializer.text_sequence = batch["question"]
        stepa_serializer.input_ids = batch["input_ids"].numpy()
        stepa_serializer.attention_mask = batch["attention_mask"].numpy()
        stepa_serialized_np = stepa_serializer.serialize()
        stepa_key = stepa_prefix + f"_{batch_idx}"
        
        stepa_next_shard_idx = STEPA_SHARD_INDICES[(batch_idx) % len(STEPA_SHARD_INDICES)]
        # tl.log(10000 ,batch_idx ,0 ,0 )
        resA = capi.put_nparray(stepa_key, stepa_serialized_np,subgroup_type=subgroup_type,
                    subgroup_index=STEPA_SUBGROUP_INDEX,shard_index=stepa_next_shard_idx, message_id=1, as_trigger=True, blokcing=False)
        

        stepb_key = stepb_prefix + f"_{batch_idx}"
        serializer = PixelValueBatcher()
        serializer.question_ids = np.asarray(stepa_serializer.question_ids,dtype=np.int64)
        serializer.pixel_values = batch["pixel_values"].numpy()
        serialized_np = serializer.serialize()
        # print(f"With serializer, we got message size of: {sys.getsizeof(serialized_np.tobytes())}")
        stepb_next_shard_idx = STEPB_SHARD_INDICES[(batch_idx) % len(STEPB_SHARD_INDICES)]
        
        
        
        for qid in batch["question_id"]:
            uds_idx =  int(qid.find("_"))
            question_id = int(qid[uds_idx+1:])
            # stepa_serializer.question_ids.append(question_id)
            tl.log(10001, question_id, 0, 0)
        resB = capi.put_nparray(stepb_key, serialized_np,subgroup_type=subgroup_type,
                    subgroup_index=STEPB_SUBGROUP_INDEX,shard_index=stepb_next_shard_idx, message_id=1, as_trigger=True, blokcing=False)
        
        for qid in batch["question_id"]:
            uds_idx =  int(qid.find("_"))
            question_id = int(qid[uds_idx+1:])
            # stepa_serializer.question_ids.append(question_id)
            tl.log(10020, question_id, 0, 0)
        
        
        time.sleep(0.0002)
        
    tl.flush("client_timestamp.dat")
  



