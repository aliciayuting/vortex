#!/usr/bin/env python3
import time
import json
import warnings

import numpy as np

from typing import Any
from FlagEmbedding import FlagModel

import cascade_context #type: ignore
from derecho.cascade.udl import UserDefinedLogic
from derecho.cascade.member_client import TimestampLogger

warnings.filterwarnings("ignore")
from pyudl_serialize_utils import Batch



class EncodeUDL(UserDefinedLogic):
    def __init__(self, conf_str: str):
        self._conf: dict[str, Any] = json.loads(conf_str)
        self._tl = TimestampLogger()
        self._encoder = None
        self._batch = Batch()
        self._batch_id = 0

    def ocdpo_handler(self, **kwargs):
        # self._tl.log("EncodeUDL: ocdpo_handler")
        if self._encoder is None:
            # load encoder when we need it to prevent overloading
            # the hardware during startup
            self._encoder = FlagModel('BAAI/bge-small-en-v1.5', devices="cuda:0")
        message_id = kwargs["message_id"]
        # TODO: this logging only works for batch of 1
        self._tl.log(10001, message_id, 0, 0)
        data = kwargs["blob"]
        
        
        self._batch.deserialize(data)
        
        query_embeddings: np.ndarray = self._encoder.encode(
            self._batch.query_list
        )
        
        self._batch_id += 1
        

        # format should be {client}_{batch_id}
        key_str = kwargs["key"]
        output_bytes = self._batch.serialize(query_embeddings)
        cascade_context.emit(key_str, output_bytes, message_id=kwargs["message_id"])
        return None
    
    def __del__(self):
        pass
