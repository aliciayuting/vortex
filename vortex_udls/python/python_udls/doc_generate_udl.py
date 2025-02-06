#!/usr/bin/env python3
from collections import defaultdict
from collections import OrderedDict
import heapq
import io
import logging
import json
import numpy as np
import pickle
import re
import time
import torch
import transformers

import cascade_context
from derecho.cascade.member_client import ServiceClientAPI
from derecho.cascade.member_client import TimestampLogger
from derecho.cascade.udl import UserDefinedLogic



class DocGenerateUDL(UserDefinedLogic):
    """
    This UDL is used to retrieve documents and generate response using LLM.
    """
    
    def load_llm(self,):
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
        self.terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        

    def __init__(self,conf_str):
        '''
        Constructor
        '''
        # collect the cluster search result {(query_batch_key,query_count):{query_id: ClusterSearchResults, ...}, ...}
        self.cluster_search_res = {}
        # collect the LLM result per client_batch {(query_batch_key,query_count):{query_id: LLMResult, ...}, ...}
        self.llm_res = {}
        self.conf = json.loads(conf_str)
        self.top_k = int(self.conf["top_k"])
        self.top_clusters_count = int(self.conf["top_clusters_count"])
        self.capi = ServiceClientAPI()
        self.my_id = self.capi.get_my_id()
        self.tl = TimestampLogger()
        self.doc_file_name = './perf_data/miniset/doc_list.pickle'
        self.answer_mapping_file = './perf_data/miniset/answer_mapping.pickle'
        self.doc_list = None
        self.answer_mapping = None
        self.pipeline = None
        self.terminators = None
    
    
    def _get_doc(self, cluster_id, emb_id):
        """
        Helper method to get the document string in natural language.
        load the documents from disk if not in memory.
        @cluster_id: The id of the KNN cluster where the document falls in.
        @emb_id: The id of the document within the cluster.
        @return: The document string in natural language.
        """
        if self.answer_mapping is None:
            with open(self.answer_mapping_file, "rb") as file:
                self.answer_mapping = pickle.load(file)
        if self.doc_list is None:
            with open(self.doc_file_name, 'rb') as file:
                self.doc_list = pickle.load(file)
        return self.doc_list[self.answer_mapping[cluster_id][emb_id]]
          


    def retrieve_documents(self, search_result):
        """
        @search_result: [(cluster_id, emb_id), ...]
        @return doc_list: [document_1, document_2, ...]
        """     
        doc_list = []
        for cluster_id, emb_id in search_result.items():
            doc_list.append(self._get_doc(cluster_id, emb_id))
        return doc_list


    def llm_generate(self, query_text, doc_list):
        """
        @query: client query
        @doc_list: list of documents
        @return: llm generated response
        """    
        messages = [
            {"role": "system", "content": "Answer the user query based on this list of documents:"+" ".join(doc_list)},
            {"role": "user", "content": query_text},
        ]
        
        llm_result = self.pipeline(
            messages,
            max_new_tokens=256,
            eos_token_id=self.terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        response = llm_result[0]["generated_text"][-1]['content']
        print(f"for query:{query_text}")
        print(f"the llm generated response: {response}")
        return response
          
               
     
     
    def ocdpo_handler(self,**kwargs):
        key = kwargs["key"]
        blob = kwargs["blob"]
        
        # batch processing
        next_key = key
        search_result = json.loads(blob.decode('utf-8'))
        doc_list = self.retrieve_documents(search_result)
        llm_generated_client_batch_res = self.llm_generate(search_result["query"], doc_list)
        client_query_batch_result_json = json.dumps(llm_generated_client_batch_res)
        #  TODO: change to use emit
        self.capi.put(next_key, client_query_batch_result_json.encode('utf-8'))
        print(f"[DocGenerate] put the agg_results to key:{next_key},\
                    \n                   value: {client_query_batch_result_json}")
        return

          

    def __del__(self):
        pass