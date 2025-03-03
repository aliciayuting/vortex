#!/usr/bin/env python3
import time
import json
import warnings
import threading
import numpy as np

from queue import Queue, Empty
from typing import Any
from FlagEmbedding import FlagModel

import cascade_context #type: ignore
from derecho.cascade.udl import UserDefinedLogic
from derecho.cascade.member_client import TimestampLogger
from derecho.cascade.member_client import ServiceClientAPI 

warnings.filterwarnings("ignore")
from pyudl_serialize_utils import Batch

# Have 3 classes:
# 1. Batcher
# 2. Compute
# 3. Sender

class Batcher(threading.Thread):
    def __init__(self, output: Queue[Batch], max_batch_size: int, capacity: int = 32, batch_time_us: int = 10_000):
        """_summary_

        Args:
            max_batch_size (int): _description_
            capacity (int, optional): _description_. Defaults to 32.
            batch_time_us (int, optional): _description_. Defaults to 10_000.
        """
        super().__init__()
        self._max_batch_size = max_batch_size
        self._batch_time_s = batch_time_us / 1_000_000
        self._output_queue: Queue[Batch] = output
        self._blob_queue: Queue[Batch] = Queue(capacity)

        self._query_ids: list[int] = []
        self._client_ids: list[int] = []
        self._text: list[str] = []
        
    def push_batch(self, batch: Batch):
        # verified shallow copy
        print("queued batch item")
        self._blob_queue.put(batch)

    def run(self):
        def send():
            self._output_queue.put(Batch(self._text, self._query_ids, self._client_ids))
            
            # since it's shallow copy, clear() will also 
            # clear out the entries in the EncoderBatch
            self._query_ids = []
            self._client_ids = []
            self._text = []

        last_reset = time.time()
        while True:
            if len(self._query_ids) == 0:
                batch = self._blob_queue.get() 
                last_reset = time.time()
            else:
                now = time.time()
                target_time = last_reset + self._batch_time_s       
                if target_time <= now:
                    send()
                    continue

                try:
                    batch = self._blob_queue.get(timeout=target_time-now)
                except Empty:
                    send()
                    continue

            for i in range(batch.size):
                self._query_ids.append(batch.query_id_list[i])
                self._client_ids.append(batch.client_id_list[i])
                self._text.append(batch.query_list[i])

                if len(self._query_ids) == self._max_batch_size:
                    send()
            
class Worker(threading.Thread):
    def __init__(self, encode_queue: Queue[Batch], send_queue: Queue[Batch], model_name: str, device: str):
        super().__init__()
        self._encode_queue = encode_queue
        self._send_queue = send_queue
        self._model: FlagModel | None = None
        self._model_name = model_name
        self._device = device

    def run(self):
        while True:
            batch = self._encode_queue.get()

            if self._model is None:
                # load encoder when we need it to prevent overloading
                # the hardware during startup

                # NOTE: use fp16 should be set to false because later udls assume f32
                # for deserialization
                self._model = FlagModel(self._model_name, device=self._device, use_fp16 = False)

            print("encoded embeddings")
            embeddings: np.ndarray = self._model.encode( # type: ignore
                batch.query_list,
                convert_to_numpy=True
            )

            batch.add_embeddings(embeddings)
            self._send_queue.put(batch)
        

class Sender(threading.Thread):
    def __init__(self, id: int, send_queue: Queue[Batch]):
        super().__init__()
        self._send_queue = send_queue
        self._batch_id = 0
        self._my_id = id

    def run(self):
        while True:
            batch = self._send_queue.get()
            output_bytes = batch.serialize()
            print("received batch")

            # format should be {client}_{batch_id} 
            # TODO: maybe change from emit to put
            key_str = f"{self._my_id}_{self._batch_id}"
            cascade_context.emit(key_str, output_bytes, message_id=1)
            self._batch_id += 1


class EncodeUDL(UserDefinedLogic):
    def __init__(self, conf_str: str):
        # TODO: parse in conf
        self._conf: dict[str, Any] = json.loads(conf_str)
        print(self._conf)

        self._tl = TimestampLogger()
        self._capi = ServiceClientAPI()
        self._my_id = self._capi.get_my_id()

        self._encode_queue: Queue[Batch] = Queue()
        self._send_queue: Queue[Batch] = Queue()

        self._batcher = Batcher(self._encode_queue, 32)
        self._batcher.start()

        self._worker = Worker(self._encode_queue, self._send_queue, "BAAI/bge-small-en-v1.5", device="cuda:0")
        self._worker.start()

        self._sender = Sender(self._my_id, self._send_queue)
        self._sender.start()

    def ocdpo_handler(self, **kwargs):
        # move data onto queue for processing
        # self._tl.log("EncodeUDL: ocdpo_handler")
        message_id = kwargs["message_id"]

        # TODO: this logging only works for batch of 1
        self._tl.log(10001, message_id, 0, 0)

        # divide incoming strings into chunks of at most self._max_batch_size
        print("received blob object")
        batch = Batch()
        batch.deserialize(kwargs["blob"])
        self._batcher.push_batch(batch)


        # return None


    def __del__(self):
        pass
