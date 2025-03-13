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

# batching queue
# encoding queue
# sending queue 

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

            timeit_start = time.time()
            embeddings: np.ndarray = self._model.encode( # type: ignore
                batch.query_list,
                convert_to_numpy=True
            )
            timeit_stop = time.time()
            print(f"Prediction time: {(timeit_stop - timeit_start)*1_000_000} us")

            batch.add_embeddings(embeddings)
            self._send_queue.put(batch)
        
class Sender(threading.Thread):
    def __init__(self, send_queue: Queue[Batch]):
        super().__init__()
        self._send_queue = send_queue
        self._batch_id = 0
        self._my_id = id
        self._capi = ServiceClientAPI()
        self._my_id = self._capi.get_my_id()

    def run(self):
        while True:
            batch = self._send_queue.get()
            timeit_start = time.time()
            output_bytes = batch.serialize()
            timeit_stop = time.time()
            print(f"Serialization time: {(timeit_stop - timeit_start)*1_000_000} us")

            # format should be {client}_{batch_id} 
            key_str = f"/rag/emb/centroids_search/{self._my_id}_{self._batch_id}"
            self._capi.put(key_str, output_bytes.tobytes())
            self._batch_id += 1

class EncodeUDL(UserDefinedLogic):
    def __init__(self, conf_str: str):
        # TODO: parse in conf
        self._conf: dict[str, Any] = json.loads(conf_str)

        self._tl = TimestampLogger()
        self._encode_queue: Queue[Batch] = Queue()
        self._send_queue: Queue[Batch] = Queue()

        self._batcher = Batcher(
            self._encode_queue,
            self._conf["max_batch_size"],
            self._conf["max_queued_entries"],
            self._conf["batch_time_us"]
        )
        self._batcher.start()

        self._worker = Worker(
            self._encode_queue,
            self._send_queue,
            self._conf["encoder_config"]["model"],
            device=self._conf["encoder_config"]["device"]
        )
        self._worker.start()

        self._sender = Sender(self._send_queue)
        self._sender.start()

    def ocdpo_handler(self, **kwargs):
        # move data onto queue for processing
        # self._tl.log("EncodeUDL: ocdpo_handler")
        message_id = kwargs["message_id"]

        # TODO: this logging only works for batch of 1
        self._tl.log(10001, message_id, 0, 0)

        batch = Batch()

        timeit_start = time.time()
        batch.deserialize(kwargs["blob"])
        timeit_stop = time.time()
        print(f"Deserialization time: {(timeit_stop - timeit_start)*1_000_000} us")

        self._batcher.push_batch(batch)

    def __del__(self):
        pass
