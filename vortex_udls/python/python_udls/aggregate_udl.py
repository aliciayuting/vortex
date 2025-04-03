#!/usr/bin/env python3
import json
import numpy as np
from typing import Any
import threading
import torch

from derecho.cascade.udl import UserDefinedLogic
from derecho.cascade.member_client import ServiceClientAPI
from derecho.cascade.member_client import TimestampLogger

from pipeline2_serialize_utils import (DocResultBatchManager, 
                                       TextCheckResultBatchManager)


class TextChecker:
    def __init__(self, device: str, model_name: str):
        self.model = None
        self.tokenizer = None
        self.device = device
        self.model_name = model_name
        
    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device)

    def model_exec(self, batch_premise: list[str]) -> np.ndarray:
        if self.model is None:
            self.load_model()
        inputs = self.tokenizer(batch_premise,
                       return_tensors='pt', padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            result = self.model(**inputs)
        logits = result.logits  # result[0] is now deprecated, use result.logits instead
        entail_contradiction_logits = logits[:, [0, 2]]
        probs = entail_contradiction_logits.softmax(dim=1)
        true_probs = probs[:, 1] * 100 
        return true_probs.tolist()



class AggregateUDL(UserDefinedLogic):
    def __init__(self, conf_str: str):
        self.conf: dict[str, Any] = json.loads(conf_str)
        self.capi = ServiceClientAPI()
        self.tl = TimestampLogger()
        self.device = self.conf["device"]
        self.model_name = self.conf["model_name"]
        # Note: batch size here is topk(5) * max_exe_batch_size, 
        self.max_exe_batch_size = int(self.conf.get("max_exe_batch_size", 16))
        self.batch_time_us = int(self.conf.get("batch_time_us", 1000))
        self.max_emit_batch_size = int(self.conf.get("max_emit_batch_size", 5))
        self.next_udl_shards = self.conf.get("next_udl_shards", [0,1])
        self.num_pending_buffer = self.conf.get("num_pending_buffer", 10)
        
        self.model_worker = None
        self.emit_worker = None
        self.sent_msg_count = 0
        
    def start_threads(self):
        '''
        Start the worker threads
        '''
        if not self.model_worker:
            self.model_worker = TextCheckWorker(self, 1)
            self.model_worker.start()
            self.emit_worker = TextCheckEmitWorker(self, 2)
            self.emit_worker.start()

    def ocdpo_handler(self, **kwargs):
        # Only start the model_worker if this UDL is triggered on this node
        if not self.model_worker:
            self.start_threads()
        data = kwargs["blob"]
        
        doc_result_batcher = DocResultBatchManager()
        doc_result_batcher.deserialize(data)
        
        for qid in doc_result_batcher.question_ids:
            self.tl.log(50000, qid, 0, 0)
            
        self.model_worker.push_to_pending_batches(doc_result_batcher)


    def __del__(self):
        '''
        Destructor
        '''
        print(f"Encoder UDL destructor")
        if self.model_worker:
            self.model_worker.signal_stop()
            self.model_worker.join()
        if self.emit_worker:
            self.emit_worker.signal_stop()
            self.emit_worker.join()
