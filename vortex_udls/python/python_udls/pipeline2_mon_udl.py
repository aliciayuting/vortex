#!/usr/bin/env python3
from datasets import load_dataset
import io
import faiss
import json
import numpy as np
import pickle
import os
import pandas as pd
import pickle
import textwrap
import time
import torch
import transformers
import warnings
from FlagEmbedding import FlagModel
from transformers import BartForSequenceClassification, BartTokenizer, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from pipeline2_serialize_utils import QueryBatcher


from derecho.cascade.udl import UserDefinedLogic
from derecho.cascade.member_client import TimestampLogger

warnings.filterwarnings("ignore")

class Pipeline2monoUDL(UserDefinedLogic):

    def __init__(self,conf_str):
        '''
        Monolithic UDL for pipeline2
        '''
        super(Pipeline2monoUDL,self).__init__(conf_str)
        self.conf = json.loads(conf_str)
        # Encoder variables
        self.encoder = None
        self.emb_dim = 384
        
        # Search variables
        self.use_precomputed_index = self.conf["use_precomputed_index"]
        self.index_file = self.conf["index_file"]
        self.index_type = 'Flat'
        self.nprobe = 1
        self.index = None
        
        # DocGenerate variables
        self.doc_file_pathname = self.conf["doc_file_pathname"]
        self.emb_directory = self.conf["emb_directory"]
        self.doc_list = None
        self.pipeline = None
        self.terminators = None
        self.doc_data = None

        # Textcheck variables
        self.device =torch.device('cuda:0')
        self.nli_model_name = "facebook/bart-large-mnli"
        self.nli_tokenizer = None
        self.nli_model = None
        self.hypothesis = "harmful"

        # Text2Speech variables
        self.txt2speech_processor = None
        self.txt2speech_model = None
        self.txt2speech_vocoder = None

        self.txt2speech_embeddings_dataset = None
        self.txt2speech_speaker_embeddings = None

        self.model_loaded = False
        self.latencies = []


    def load_all(self):
        self.load_models()
        if self.use_precomputed_index:
            self.load_index()
        else:
            self.load_cluster_embeddings(self.emb_directory)
            self.build_ivf_index()
        self.model_loaded = True


    def load_models(self,):
        # Load encoder
        self.encoder = FlagModel(
                        'BAAI/bge-small-en-v1.5',
                        devices="cuda:0",
                    )
        # Load LLM model
        model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.float16},
            device_map="auto",
        )
        self.terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        # load text check model
        self.nli_tokenizer = BartTokenizer.from_pretrained(self.nli_model_name)
        self.nli_model = BartForSequenceClassification.from_pretrained(self.nli_model_name)
        for param in self.nli_model.parameters():
            param.requires_grad = False
        self.nli_model.to(self.device)
        self.nli_model.eval()
        # Load text2speech model
        self.txt2speech_processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        self.txt2speech_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to("cuda")
        self.txt2speech_vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to("cuda")
        self.txt2speech_embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        self.txt2speech_speaker_embeddings = torch.tensor(self.txt2speech_embeddings_dataset[7306]["xvector"]).unsqueeze(0).to("cuda")


    def load_cluster_embeddings(self, cluster_dir):
        self.cluster_embeddings = []
        for file in os.listdir(cluster_dir):
            if file.startswith("cluster_") and file.endswith(".pkl"):
                file_path = os.path.join(cluster_dir, file)
                with open(file_path, "rb") as f:
                        emb = pickle.load(f)
                        self.cluster_embeddings.append(emb)
        self.cluster_embeddings = np.vstack(self.cluster_embeddings).astype(np.float32)  


    def build_ivf_index(self, nlist=10):
        dim = self.cluster_embeddings.shape[1]  
        res = faiss.StandardGpuResources()  
        self.index = faiss.GpuIndexIVFFlat(
            res,  
            dim,  
            nlist,  
            faiss.METRIC_L2, 
            faiss.GpuIndexIVFFlatConfig()
        )

        self.index.train(self.cluster_embeddings)
        self.index.add(self.cluster_embeddings) 


    def load_index(self):
        self.index = faiss.read_index(self.index_file)
        gpu_res = faiss.StandardGpuResources()
        self.index = faiss.index_cpu_to_gpu(gpu_res, 0, self.index)


    def search_queries(self, query_embeddings, top_k=5):
        distances, indices = self.index.search(query_embeddings, top_k)
        return distances, indices


    def encode(self,query_list):
        query_embeddings = self.encoder.encode(query_list)
        # print(f"finished encoding shape: {query_embeddings.shape}")
        return query_embeddings


    def llm_generate(self, query_text, doc_list ):
        messages = [
            {"role": "system", "content": "Answer the user query based on this list of documents:"+" ".join(doc_list)},
            {"role": "user", "content": query_text},
        ]
        
        tmp_res = self.pipeline(
            messages,
            max_new_tokens=256,
            # eos_token_id=self.terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        raw_text = tmp_res[0]["generated_text"][-1]['content']

        return raw_text
    
    # Retrieve Documents Based on FAISS Indices
    def get_documents(self,top_k_idxs):          
        doc_list = []
        if self.doc_data is None:
            with open(self.doc_file_pathname, "rb") as f:
                self.doc_data = pickle.load(f)  
        
        for idx in top_k_idxs:
            if 0 <= idx < len(self.doc_data):
                doc_text = self.doc_data[idx]
                doc_list.append(doc_text)
            else:
                print(f"Warning: FAISS index {idx} is out of range in doc_list.pkl.")
        return doc_list
    

    def text_check(self, text):
        # run through model pre-trained on MNLI
        input_ids = self.nli_tokenizer.encode(text, self.hypothesis, return_tensors='pt').to(self.device)
        result = self.nli_model(input_ids)
        logits = result[0]

        entail_contradiction_logits = logits[:,[0,2]]
        probs = entail_contradiction_logits.softmax(dim=1)
        true_prob = probs[:,1].item() * 100
        return true_prob


        
    def split_text(self, text, chunk_size=550):
        return textwrap.wrap(text, width=chunk_size) 

    def text2speech(self, text):
        # Split text into smaller segments
        text_chunks = self.split_text(text)
        speech_outputs = []

        for idx, chunk in enumerate(text_chunks):
            inputs = self.txt2speech_processor(text=chunk, return_tensors="pt").to("cuda")  # Move inputs to GPU
            speech = self.txt2speech_model.generate_speech(inputs["input_ids"], self.txt2speech_speaker_embeddings, vocoder=self.txt2speech_vocoder)
            speech_outputs.append(speech.cpu().numpy())  # Move back to CPU before converting to NumPy
            # print(f"Chunk {idx+1}/{len(text_chunks)} processed.")
        
        full_speech = np.concatenate(speech_outputs, axis=0)
        return full_speech


    def ocdpo_handler(self,**kwargs):
        if not self.model_loaded:
            self.load_all()
        
        start_time = time.perf_counter_ns()
        data = kwargs["blob"]
        _batch = Batch()
        _batch.deserialize(data)
        
        batched_query_embeddings = self.encoder.encode(_batch.query_list)

        # print(f"query_embeddings shape: {batched_query_embeddings.shape}")
        _, batched_indices = self.search_queries(batched_query_embeddings, top_k=5)
        
        for i in range(len(_batch.query_list)):
            
            doc_list = self.get_documents(batched_indices[i])
            
            llm_res = self.llm_generate(_batch.query_list[i], doc_list)

            audio = self.text2speech(llm_res)

            result_valid = self.text_check(llm_res)
            
            end_time = time.perf_counter_ns()
            latency = (end_time - start_time)/1000
            self.latencies.append(latency)
            print(f"Latency {i}/{len(_batch.query_list)}: {latency} us")
            
        print("Pipeline2 finished")
            

    def __del__(self):
        pass
