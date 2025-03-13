import json
import queue
import time
import threading
import torch
from transformers import BartForSequenceClassification, BartTokenizer

import cascade_context
from derecho.cascade.member_client import ServiceClientAPI
from derecho.cascade.member_client import TimestampLogger
from derecho.cascade.udl import UserDefinedLogic



'''
Used for the ablation study of the pipeline
'''


class QACheckBatchUDL(UserDefinedLogic):
    '''
    AnswerCheckUDL is udl that run NLI model to check if the answer contains harmful information
    '''
    def __init__(self,conf_str):
        super(QACheckBatchUDL,self).__init__(conf_str)
        self.conf = json.loads(conf_str)
        self.tl = TimestampLogger()
        self.capi = ServiceClientAPI()
        self.my_id = self.capi.get_my_id()
        self.device_name = 'cuda:0'
        self.device =torch.device(self.device_name)
        self.model_name = "facebook/bart-large-mnli"
        self.nli_tokenizer = None
        self.model = None
        self.hypothesis = "harmful"
        
        
        
    def load_model_toGPU(self):
        '''
        Load model to GPU
        '''
        print(f"[{time.time():.2f}] QA check Loading model to GPU...")
        self.nli_tokenizer = BartTokenizer.from_pretrained(self.model_name)
        self.model = BartForSequenceClassification.from_pretrained(self.model_name)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.to(self.device)
        self.model.eval()
        print(f"[{time.time():.2f}] QA check Model loaded to GPU")


    def textcheck(self, batch_premise):
        """
        Runs batch text classification
        process batch_premise from the queue
        """
        if self.nli_tokenizer is None:
            self.load_model_toGPU()
        
        print(f"[{time.time():.2f}] QA check Processing batch of size {len(batch_premise)}...")
        # Note: this step is blocking, doesn't release the GIL  
        inputs = self.nli_tokenizer(batch_premise, [self.hypothesis] * len(batch_premise),
                        return_tensors="pt", padding=True, truncation=True).to(self.device)

        batch_start_time = time.time()

        # CUDA exec, releases GIL
        with torch.no_grad():
            result = self.model(**inputs)

        batch_latency = time.time() - batch_start_time
        self.total_latency += batch_latency

        logits = result.logits
        entail_contradiction_logits = logits[:, [0, 2]]
        probs = entail_contradiction_logits.softmax(dim=1)
        true_probs = probs[:, 1] * 100 

        # for i, prob in enumerate(true_probs.tolist()):
        #     print(f"[{time.time():.2f}] Premise {i+1}: Probability that the label is true: {prob:.2f}%")
        self.processed_tasks += len(batch_premise)
        
        print(f"[{time.time():.2f}] Processed {len(self.processed_tasks)} tasks")

        return batch_latency, true_probs


    def send_to_next_udl(self, result):
        """
        Send result to next UDL
        """
        # self.capi.put(key,result)
        # print(f"[{time.time():.2f}] Sent result to next UDL: {result}")
        pass


    def add_tasks_to_queue(self, input_texts):
        """
        Adds a text classification task to the queue with timestamp
        """
        with self.cond_var:            
            for input_text in input_texts:
                self.task_queue.put((input_text))
                print(f"[{time.time():.2f}] Task added to queue. Queue size: {self.task_queue.qsize()}")
            self.cond_var.notify()

    def ocdpo_handler(self,**kwargs):
        """
        Handles incoming tasks
        """
        key = kwargs["key"]
        blob = kwargs["blob"]
        decoded_json_string = blob.tobytes().decode('utf-8')
        query_list = json.loads(decoded_json_string)
        print(f"[{time.time():.2f}] Received query: {query_list}")
        self.add_tasks_to_queue(query_list)
        
        

    def __del__(self):
        pass
