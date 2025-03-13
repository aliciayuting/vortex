#!/usr/bin/env python3

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

NUM_BATCHES = 10

def main(argv):

    print("Connecting to Cascade service ...")
    capi = ServiceClientAPI()
    
    for i in range(NUM_BATCHES):
        querybatch_id = i
        key = "/rag/generate/check/test" + str(querybatch_id)
        # if i % 2 == 1:
        #         key = "/rag/generate/checktwo/test" + str(querybatch_id)
        query_list = ["hello world", "I am RAG"]
        json_string = json.dumps(query_list)
        encoded_bytes = json_string.encode('utf-8')
        _subgroup_type = SUBGROUP_TYPES["VCSS"]
        capi.put(key, encoded_bytes, 
                subgroup_type=_subgroup_type,
                subgroup_index=0,shard_index=0,
                message_id=i)
        # capi.put("/test/hello", array.tostring())  # deprecated
        print(f"Put key:{key} \n    value:{query_list} to Cascade.")
    

if __name__ == "__main__":
    main(sys.argv)
