import json
import queue
import textwrap
import time
import threading
import torch

from datasets import load_dataset
import numpy as np
import soundfile as sf
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

import cascade_context
from derecho.cascade.member_client import ServiceClientAPI
from derecho.cascade.member_client import TimestampLogger
from derecho.cascade.udl import UserDefinedLogic

from pyudl_serialize_utils import DocGenResultBatcher



'''
Used for the ablation study of the pipeline
'''


class TextToSpeechUDL(UserDefinedLogic):
    '''
    AnswerCheckUDL is udl that run NLI model to check if the answer contains harmful information
    '''
    def __init__(self,conf_str):
        super(TextToSpeechUDL,self).__init__(conf_str)
        self.conf = json.loads(conf_str)
        self.tl = TimestampLogger()
        self.capi = ServiceClientAPI()
        self.my_id = self.capi.get_my_id()
        self.device = "cuda"
        self.processor = None
        self.model = None
        self.vocoder = None
        self.speaker_embeddings = None
        self.write_to_disk = False
        
        
    def load_model_toGPU(self):
        '''
        Load model to GPU
        '''
        print(f"Txt2Speech Model about to load to GPU")
        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        self.model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(self.device)
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(self.device)

        # load xvector containing speaker's voice characteristics from a dataset
        embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        self.speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0).to(self.device)

        # print(f"Txt2Speech Model loaded to GPU")


    def split_text(self, text, chunk_size=550):
        return textwrap.wrap(text, width=chunk_size) 


    def speech_generation(self, response):
        """
        text to speech model only support single text input at a time
        """
        if self.processor is None:
            self.load_model_toGPU()
        # Split text into smaller segments
        text_chunks = self.split_text(response)
        speech_outputs = []

        for idx, chunk in enumerate(text_chunks):
            inputs = self.processor(text=chunk, return_tensors="pt").to(self.device) 
            speech = self.model.generate_speech(inputs["input_ids"], self.speaker_embeddings, vocoder=self.vocoder)
            speech_outputs.append(speech.cpu().numpy())  # Move back to CPU before converting to NumPy
            # print(f"Chunk {idx+1}/{len(text_chunks)} processed.")
        full_speech = np.concatenate(speech_outputs, axis=0)

        if self.write_to_disk:
            sf.write("speech.wav", full_speech, samplerate=16000)
        # print(f" Txt2Speech Speech generated")
        return full_speech


    def ocdpo_handler(self,**kwargs):
        """
        Handles incoming tasks
        """
        key = kwargs["key"]
        blob = kwargs["blob"]
        doc_gen_result_batch = DocGenResultBatcher()
        doc_gen_result_batch.deserialize_response(blob)
        for i in range(len(doc_gen_result_batch.responses)):
            doc_gen_result = doc_gen_result_batch.responses[i]
            query_id = int(doc_gen_result_batch.query_ids[i])
            self.speech_generation(doc_gen_result)
            self.tl.log(20050, query_id, 0, 0)
            if query_id == 20:
                self.tl.flush(f"node{self.my_id}_udls_timestamp.dat")
                print(f" text2speech flushed data")
        

    def __del__(self):
        pass
