#!/usr/bin/env python3
import json
import numpy as np
from typing import Any
import warnings

from derecho.cascade.udl import UserDefinedLogic
from derecho.cascade.member_client import ServiceClientAPI
from derecho.cascade.member_client import TimestampLogger

from pipeline2_serialize_utils import (DocResultBatchManager, 
                                       TextCheckResultBatchManager,
                                       LangDetResultBatchManager)


class CollectedResult:
    def __init__(self, question_id):
        self.question_id = question_id
        self.question_text = None
        self.doc_list = None
        self.text_check_list = None  # a list of doc_type IDs: 0 hate speech, 1 offensive language, 2 neither
        self.lang_detect_list = None

        
    def collect_all(self):
        if self.question_text is None:
            return False
        if self.doc_list is None:
            return False
        if self.text_check_list is None:
            return False
        if self.lang_detect_list is None:
            return False
        return True
    
    def print_result(self):
        print(f"Question ID: {self.question_id}")
        print(f"Question Text: {self.question_text}")
        print(f"Document List: {self.doc_list}")
        print(f"Text Check List: {self.text_check_list}")
        print(f"Language Detect List: {self.lang_detect_list}")



class AggregateUDL(UserDefinedLogic):
    def __init__(self, conf_str: str):
        self.conf: dict[str, Any] = json.loads(conf_str)
        self.capi = ServiceClientAPI()
        self.tl = TimestampLogger()
        self.collected_results = {}
        self.finished_count = 0
        self.flush_qid = self.conf["flush_qid"]
    
    def add_doc_result(self,doc_result_batch: DocResultBatchManager):
        for idx, qid in enumerate(doc_result_batch.question_ids):
            self.tl.log(70001, qid, 0, 0)
            if qid not in self.collected_results:
                self.collected_results[qid] = CollectedResult(qid)
            self.collected_results[qid].question_text = doc_result_batch.queries[idx]
            self.collected_results[qid].doc_list = doc_result_batch.doc_list[idx].copy()
            if len(self.collected_results[qid].doc_list) == 0:
                warnings.warn(f"AGG received an empty doc list for question ID {qid}")

    def add_text_check_result(self, text_check_result_batch: TextCheckResultBatchManager):
        for idx, qid in enumerate(text_check_result_batch.question_ids):
            self.tl.log(70010, qid, 0, 0)
            if qid not in self.collected_results:
                self.collected_results[qid] = CollectedResult(qid)
            self.collected_results[qid].text_check_list = text_check_result_batch.doc_types[idx].copy()
            if len(self.collected_results[qid].text_check_list) == 0:
                warnings.warn(f"AGG received an empty text check result for question ID {qid}")
            
    def add_lang_detect_result(self, lang_detect_result_batch: LangDetResultBatchManager):
        for idx, qid in enumerate(lang_detect_result_batch.question_ids):
            self.tl.log(70011, qid, 0, 0)
            if qid not in self.collected_results:
                self.collected_results[qid] = CollectedResult(qid)
            self.collected_results[qid].lang_detect_list = lang_detect_result_batch.lang_codes[idx].copy()
            if len(self.collected_results[qid].lang_detect_list) == 0:
                warnings.warn(f"AGG received an empty lang detect list for question ID {qid}")


    def ocdpo_handler(self, **kwargs):
        data = kwargs["blob"]
        key = kwargs["key"]
        
        qids = []
        if key.find("doc") != -1:
            # doc result
            doc_result_batcher = DocResultBatchManager()
            doc_result_batcher.deserialize(data)
            
            bsize = doc_result_batcher.num_queries
            for qid in doc_result_batcher.question_ids:
                qids.append(qid)
                self.tl.log(70000, qid, bsize, 0)
                if qid == self.flush_qid:
                    print(f"AGG received Doc result from No.{qid} queries")
                
            self.add_doc_result(doc_result_batcher)
            
            
            
        if key.find("check") != -1:
            # text check result
            text_check_result_batcher = TextCheckResultBatchManager()
            text_check_result_batcher.deserialize(data)
            
            bsize = text_check_result_batcher.num_queries
            for qid in text_check_result_batcher.question_ids:
                qids.append(qid)
                self.tl.log(70000, qid, bsize, 0)
                if qid == self.flush_qid:
                    print(f"AGG received TextCheck result from No.{qid} queries")
                
            self.add_text_check_result(text_check_result_batcher)
            
            
            
        if key.find("lang") != -1:
            # lang detect result
            lang_detect_result_batcher = LangDetResultBatchManager()
            lang_detect_result_batcher.deserialize(data)
            
            bsize = lang_detect_result_batcher.num_queries
            for qid in lang_detect_result_batcher.question_ids:
                qids.append(qid)
                self.tl.log(70000, qid, bsize, 0)
                if qid == self.flush_qid:
                    print(f"AGG received LangDetect result from No.{qid} queries")
                
            self.add_lang_detect_result(lang_detect_result_batcher)
            

        for qid in qids:
            if self.collected_results[qid].collect_all():
                self.finished_count += 1
                self.tl.log(70100, qid, 0, 1)
                if qid == self.flush_qid:
                    print(f"AGG received all results for question ID {qid}")
                    # self.collected_results[qid].print_result()
                    print(f"AGG finished count: {self.finished_count}")
            
            
        


    def __del__(self):
        '''
        Destructor
        '''
        print(f"Agg UDL destructor")
