import numpy as np
import warnings
warnings.filterwarnings("ignore")


def utf8_length(s: str) -> int:
    """Computes the length of a UTF-8 encoded string without actually encoding it."""
    return sum(1 + (ord(c) >= 0x80) + (ord(c) >= 0x800) + (ord(c) >= 0x10000) for c in s)

# ------------------------    STEP A (speech2text) UDL batcher  -------------------------

class AudioBatcher:
    def __init__(self):
        self.audio_data = [] # List of np.ndarray float64
        self.question_ids = []
        self._bytes: np.ndarray = np.array([], dtype=np.uint8)
        
        # variable used by serialization
        self.audio_pos = {}  # qid -> (offset, length)

    def serialize(self) -> np.ndarray:
        """
        Serializes the following fields into a contiguous buffer:
        - Header: 4 bytes for batch_size (uint32).
        - Metadata: For each audio_data element, store two int64 values (offset and length).
        - Fixed segments:
            * question_ids: (batch_size,) int64.
        - Variable segment:
            * audio_data: list of np.ndarray float64 with various sizes.
        """
        batch_size = len(self.audio_data)
        assert batch_size == len(self.question_ids)

        header_size = np.dtype(np.uint32).itemsize  # 4 bytes
        metadata_dtype = np.dtype([("offset", np.int64), ("length", np.int64)])
        metadata_size = batch_size * metadata_dtype.itemsize  # 16 bytes per item
        qid_size = batch_size * np.dtype(np.int64).itemsize       # 8 bytes per item

        # Fixed segment size before audio data.
        fixed_size = header_size + metadata_size + qid_size
        # Add padding so that the variable segment is aligned to 8 bytes.
        # padding = (16 - (fixed_size % 16)) % 16  
        variable_data_offset = fixed_size #+ padding

        total_audio_bytes = 0
        contiguous_audio = []
        # Ensure all audio arrays are contiguous.
        for audio in self.audio_data:
            total_audio_bytes += audio.nbytes

        total_size = variable_data_offset + total_audio_bytes
        buffer = np.zeros(total_size, dtype=np.uint8)

        # Write header.
        buffer[:header_size].view(np.uint32)[0] = batch_size
        # Write metadata.
        metadata_view = buffer[header_size : header_size + metadata_size].view(metadata_dtype)
        # Write question IDs.
        qid_view = buffer[header_size + metadata_size : fixed_size].view(np.int64)
        qid_view[:] = self.question_ids
        # # Optionally, zero the padding bytes.
        # if padding:
        #     buffer[fixed_size:variable_data_offset] = 0
        # Write variable segment: audio_data.
        write_ptr = variable_data_offset
        for i, audio_array in enumerate(self.audio_data):
            audio_bytes = audio_array.nbytes
            # Record offset relative to the start of the variable segment.
            metadata_view[i] = (write_ptr - variable_data_offset, audio_bytes)
            # Obtain a writable view into the target slice.
            dest = buffer[write_ptr : write_ptr + audio_bytes].view(audio_array.dtype)
            dest[:] = audio_array
            write_ptr += audio_bytes
        self._bytes = buffer
        return buffer

    def deserialize(self, data: np.ndarray):
        # copy data
        self._bytes = data

        header_size = np.dtype(np.uint32).itemsize
        metadata_dtype = np.dtype([("offset", np.int64), ("length", np.int64)])
        batch_size = np.frombuffer(data[:header_size], dtype=np.uint32)[0]
        metadata_size = batch_size * metadata_dtype.itemsize
        qid_size = batch_size * np.dtype(np.int64).itemsize

        fixed_size = header_size + metadata_size + qid_size
        # padding = (16 - (fixed_size % 16)) % 16  
        variable_data_offset = fixed_size #+ padding

        metadata_view = np.frombuffer(
            data[header_size : header_size + metadata_size], dtype=metadata_dtype
        )
        qid_view = np.frombuffer(
            data[header_size + metadata_size : fixed_size], dtype=np.int64
        )

        self.question_ids = qid_view.tolist()
        self.audio_data = []

        for i in range(batch_size):
            offset = metadata_view[i]["offset"]
            length = metadata_view[i]["length"]
            start = variable_data_offset + offset
            end = start + length
            audio_bytes = memoryview(data)[start:end]
            audio_array = np.frombuffer(audio_bytes, dtype=np.float64)
            self.audio_data.append(audio_array)

    def print_info(self):
        print(f"AudioBatcher: {len(self.audio_data)} audio data")
        for i, audio in enumerate(self.audio_data):
            print(f"Audio {i}: shape {audio.shape}, dtype {audio.dtype}")
        print(f"Question IDs: {self.question_ids}")


class PendingAudioRecDataBatcher:
    '''
    Super batch of AudioBatcher, used for model runtime batch process.
    '''
    def __init__(self, batch_size: int):
        self.max_batch_size = batch_size
        self.num_pending = 0
        self.question_ids = []  # List[int]
        self.audio_data = []   # List[np.ndarray]
        
    def space_left(self):
        return self.max_batch_size - self.num_pending
    
    def add_data(self, audioBatcher, start_pos):
        num_to_add = min(self.space_left(), len(audioBatcher.question_ids) - start_pos)
        end_pos = start_pos + num_to_add
        self.question_ids.extend(audioBatcher.question_ids[start_pos:end_pos])
        # Make deep copies of each audio array as it's added, 
        #  because memory of list[np array] is managed by numpy, leading to memory corruption if not copy out
        for i in range(start_pos, end_pos):
            # Create a new array with its own memory
            new_array = np.array(audioBatcher.audio_data[i], copy=True)
            
        self.audio_data.append(new_array)
        self.num_pending += num_to_add
        return end_pos
    
    def reset(self):
        self.question_ids = []
        self.audio_data = []
        self.num_pending = 0


class QueryBatcherManager:
    """
    Batches of text queries to pass results from generated by audio recognition to text encoder UDL
    """
    def __init__(self):
        self.question_ids = []   # List[int] of length batch size
        self.queries = []        # List[str] of length batch size
        self._bytes: np.ndarray = np.array([], dtype=np.uint8)
        self.num_queries = 0
        # variable used by serialization
        self.text_pos = {}  # qid -> (offset, length)
        
    def add_result(self, question_id: int, query: str):
        """
        Add one result to the batch.
        It is assumed that each query is a string.
        """
        self.question_ids.append(question_id)
        self.queries.append(query)
        self.num_queries += 1
        
    def serialize(self, start_pos: int, end_pos: int) -> np.ndarray:
        """
        Serializes a slice of the queries from start_pos to end_pos into a contiguous buffer:
        - Header: 4 bytes for batch_size (uint32).
        - Metadata: For each text_sequence element, store two int64 values (offset and length).
        - Fixed segments:
            * question_ids: (batch_size,) int64.
        - Variable segment:
            * text_sequence: concatenated UTF-8 encoded bytes.
        """
        selected_queries = self.queries[start_pos:end_pos]
        selected_question_ids = self.question_ids[start_pos:end_pos]
        num_queries = len(selected_queries)

        header_size = np.dtype(np.uint32).itemsize

        metadata_dtype = np.dtype([
            ("text_sequence_offset", np.int64),
            ("text_sequence_length", np.int64)
        ])
        metadata_size = num_queries * metadata_dtype.itemsize
        question_id_size = num_queries * np.dtype(np.int64).itemsize
        text_offset = header_size + metadata_size + question_id_size

        text_pos = {}
        total_utf8_size = 0
        for idx, q in enumerate(selected_queries):
            utf8_len = utf8_length(q)
            text_pos[selected_question_ids[idx]] = (text_offset, utf8_len)
            text_offset += utf8_len
            total_utf8_size += utf8_len

        total_size = header_size + metadata_size + question_id_size + total_utf8_size

        buffer = np.zeros(total_size, dtype=np.uint8)

        # Header
        np.frombuffer(buffer[:header_size], dtype=np.uint32)[0] = num_queries

        # Metadata
        metadata_view = np.frombuffer(buffer[header_size:header_size + metadata_size], dtype=metadata_dtype)

        # Question IDs
        question_ids_view = np.frombuffer(
            buffer[header_size + metadata_size:header_size + metadata_size + question_id_size], dtype=np.int64)
        question_ids_view[:] = selected_question_ids

        # Text sequence
        text_start = header_size + metadata_size + question_id_size
        for idx, q in enumerate(selected_queries):
            offset, length = text_pos[selected_question_ids[idx]]
            metadata_view[idx] = (offset, length)
            encoded = q.encode("utf-8")  # one copy here, still required to actually fill the buffer
            buffer[text_start:text_start + length] = np.frombuffer(encoded, dtype=np.uint8)
            text_start += length

        self._bytes = buffer
        return buffer
    
    def deserialize(self, data: np.ndarray):
        self._bytes = data
        header_size = np.dtype(np.uint32).itemsize
        metadata_dtype = np.dtype([
            ("text_sequence_offset", np.int64),
            ("text_sequence_length", np.int64)
        ])
        num_queries = np.frombuffer(data[:header_size], dtype=np.uint32)[0]
        metadata_size = num_queries * metadata_dtype.itemsize
        question_id_size = num_queries * np.dtype(np.int64).itemsize
        text_start = header_size + metadata_size + question_id_size 
        
        metadata_view = np.frombuffer(data[header_size:header_size + metadata_size], dtype=metadata_dtype)
        question_ids_view = np.frombuffer(data[header_size + metadata_size:header_size + metadata_size + question_id_size], dtype=np.int64)

        self.question_ids = question_ids_view.tolist()
        self.queries = []
        for idx in range(num_queries):
            text_offset, text_length = metadata_view[idx]
            self.queries.append(memoryview(data)[text_start + text_offset:text_start + text_offset + text_length].tobytes().decode("utf-8"))
            
    def print_info(self):
        print(f"QueryBatcherManager: {len(self.queries)} queries")
        for i, q in enumerate(self.queries):
            print(f"Query {i}: {q}")
        print(f"Question IDs: {self.question_ids}")

class PendingEncodeDataBatcher:
    def __init__(self, batch_size: int):
        self.max_batch_size = batch_size
        self.num_pending = 0
        self.question_ids = []  
        self.queries = []
    
    def space_left(self):
        return self.max_batch_size - self.num_pending
    
    def add_data(self, queryBatcher, start_pos):
        num_to_add = min(self.space_left(), len(queryBatcher.question_ids) - start_pos)
        end_pos = start_pos + num_to_add
        self.question_ids.extend(queryBatcher.question_ids[start_pos:end_pos])
        self.queries.extend(queryBatcher.queries[start_pos:end_pos])
        self.num_pending += num_to_add
        return end_pos 


    def reset(self):
        self.question_ids = []
        self.queries = []
        self.num_pending = 0

# ------------------------    STEP B (Encoder) UDL batcher  -------------------------


class EncodeResultBatchManager:
    """
    Manages the encoding results for a batch of queries and serializes the batch in the format expected by Batch.deserialize.
    
    The output layout is:
      [Header][Metadata records][Concatenated text bytes][Embeddings block]
    
    Header (dtype header_type):
      - count: uint32, the number of queries
      - embeddings_start: uint32, byte offset to the start of embeddings block
    
    Metadata (dtype metadata_type) for each query:
      - question_id: uint64
      - text_position: uint32  (absolute offset to the query text in the buffer)
      - text_length: uint32    (length in bytes of the query text)
      - embeddings_position: uint32 (absolute offset to the query embedding in the buffer)
      - embeddings_dim: uint32 (number of float32 values in the embedding)
    """
    
    def __init__(self):
        self.question_ids = []  # list[int]
        self.queries = []       # list[str]
        self.embeddings_list = []  # list[np.ndarray]
        self.emb_dim = 0
        self.num_queries = 0
        self._bytes: np.ndarray = np.array([], dtype=np.uint8)

    def add_result(self, question_id: int, query: str, embeddings: np.ndarray):
        """
        Add one result to the batch.
        It is assumed that each embedding is a numpy array of type float32
        with either shape (d,) or (1, d) and that all embeddings share the same dimension.
        """
        self.question_ids.append(question_id)
        self.queries.append(query)
        self.embeddings_list.append(embeddings)
        self.num_queries += 1
    
    def serialize(self, start_pos, end_pos) -> np.ndarray:
        batch_size = end_pos - start_pos
        header_type = np.dtype([
            ('count', np.uint32),
            ('embeddings_start', np.uint32)
        ])
        
        metadata_type = np.dtype([
            ('question_id', np.uint64),
            ('text_position', np.uint32),
            ('text_length', np.uint32),
            ('embeddings_position', np.uint32),
            ('embeddings_dim', np.uint32),
        ])
        
        count = batch_size
        header_size = header_type.itemsize
        metadata_size = count * metadata_type.itemsize
        
        # Preprocess query texts.
        text_bytes_list = [q.encode('utf-8') for q in self.queries[start_pos:end_pos]]
        text_lengths = [len(b) for b in text_bytes_list]
        total_text_size = sum(text_lengths)
        
       # Preprocess embeddings.
        first_emb = self.embeddings_list[0]
        if len(first_emb.shape) == 1:
            self.emb_dim = first_emb.shape[0]
        else:
            self.emb_dim = first_emb.shape[1]
        emb_itemsize = np.dtype(np.float32).itemsize
        embedding_bytes_per_query = self.emb_dim * emb_itemsize
        
        total_embeddings_size = count * embedding_bytes_per_query
        
        # Calculate the embeddings_start offset.
        embeddings_start = header_size + metadata_size + total_text_size
        total_size = header_size + metadata_size + total_text_size + total_embeddings_size
        
        # Allocate the output buffer.
        buffer = np.zeros(total_size, dtype=np.uint8)
        
        # Fill the header.
        header_array = np.frombuffer(buffer[:header_size], dtype=header_type)
        header_array[0] = (count, embeddings_start)
        
        # Fill the metadata records.
        metadata_array = np.frombuffer(buffer[header_size:header_size + metadata_size], dtype=metadata_type)
        text_block_start = header_size + metadata_size
        embeddings_block_start = embeddings_start
        
        current_text_offset = 0
        
        for i in range(count):
            # Calculate absolute positions.
            text_pos = text_block_start + current_text_offset
            text_len = text_lengths[i]
            emb_pos = embeddings_block_start + i * embedding_bytes_per_query
            
            # Fill metadata record.
            metadata_array[i]['question_id'] = self.question_ids[start_pos + i]
            metadata_array[i]['text_position'] = text_pos
            metadata_array[i]['text_length'] = text_len
            metadata_array[i]['embeddings_position'] = emb_pos
            metadata_array[i]['embeddings_dim'] = self.emb_dim
            
            current_text_offset += text_len
        
        # Write the text block.
        current_text_offset = 0
        for b in text_bytes_list:
            start = text_block_start + current_text_offset
            end = start + len(b)
            buffer[start:end] = np.frombuffer(b, dtype=np.uint8)
            current_text_offset += len(b)
            
        # Write the embeddings block.
        current_embedding_offset = 0
        for emb in self.embeddings_list[start_pos:end_pos]:
            start = embeddings_block_start + current_embedding_offset
            # Ensure the embedding is float32.
            if emb.dtype != np.float32:
                emb = emb.astype(np.float32)
            emb_bytes = emb.flatten().view(np.uint8)
            emb_nbytes = emb.nbytes
            end = start + emb_nbytes
            buffer[start:end] = np.frombuffer(emb_bytes, dtype=np.uint8)
            current_embedding_offset += emb_nbytes
        
        return buffer

    def deserialize(self,buffer: np.ndarray):
        """
        The layout is:
        [Header][Metadata records][Concatenated text bytes][Embeddings block]
        
        Header (dtype header_type):
        - count: uint32, the number of queries
        - embeddings_start: uint32, byte offset to the start of the embeddings block
        
        Metadata (dtype metadata_type) for each query:
        - question_id: uint64
        - text_position: uint32 (absolute offset to the query text)
        - text_length: uint32   (length in bytes of the query text)
        - embeddings_position: uint32 (absolute offset to the query embedding)
        - embeddings_dim: uint32 (number of float32 values in the embedding)
        """
        # Create a memory view for zero-copy slicing.
        mv = memoryview(buffer)
        
        # Define header and metadata dtypes.
        header_dtype = np.dtype([
            ('count', np.uint32),
            ('embeddings_start', np.uint32)
        ])
        metadata_dtype = np.dtype([
            ('question_id', np.uint64),
            ('text_position', np.uint32),
            ('text_length', np.uint32),
            ('embeddings_position', np.uint32),
            ('embeddings_dim', np.uint32),
        ])
        
        # Read header (no copy, just a view).
        header = np.frombuffer(mv, dtype=header_dtype, count=1)[0]
        count = int(header['count'])
        self.num_queries = count
        embeddings_start = int(header['embeddings_start'])
        
        # Compute sizes.
        header_size = header_dtype.itemsize
        metadata_size = count * metadata_dtype.itemsize
        
        # Read metadata as a view.
        metadata = np.frombuffer(mv, dtype=metadata_dtype, count=count, offset=header_size)
      
        
        # For each metadata record, create views for text and embeddings.
        for rec in metadata:
            self.question_ids.append(int(rec['question_id']))
            
            # Get the query text.
            text_pos = int(rec['text_position'])
            text_len = int(rec['text_length'])
            # This slice creates a new bytes object for the string, which is needed for decoding.
            text_bytes = mv[text_pos:text_pos + text_len].tobytes()
            self.queries.append(text_bytes.decode('utf-8'))
            
            # Create a view for the embedding (no copy).
            emb_pos = int(rec['embeddings_position'])
            self.emb_dim = int(rec['embeddings_dim'])
            # np.frombuffer returns a view on mv without copying the data.
            embedding = np.frombuffer(mv, dtype=np.float32, count=self.emb_dim, offset=emb_pos)
            self.embeddings_list.append(embedding)
        
    
    def print_info(self):
        print(f"EncodeResultBatchManager: {self.num_queries} queries")
        for i, q in enumerate(self.queries):
            print(f"Query {i}: {q}")
        print(f"Question IDs: {self.question_ids}")
        print(f"Embeddings dimension: {self.emb_dim}")

# ------------------------    STEP C (Searcher) UDL batcher  -------------------------


class PendingSearchDataBatcher:
    def __init__(self, batch_size: int, emb_dim: int):
        self.max_batch_size = batch_size
        self.num_pending = 0
        self.question_ids = []
        self.queries = []
        self.embeddings = np.empty((self.max_batch_size, emb_dim), dtype=np.float32)  # Placeholder for embeddings

    
    def space_left(self):
        return self.max_batch_size - self.num_pending
    
    def add_data(self, embedding_batcher, start_pos):
        num_to_add = min(self.space_left(), len(embedding_batcher.question_ids) - start_pos)
        end_pos = start_pos + num_to_add
        self.question_ids.extend(embedding_batcher.question_ids[start_pos:end_pos])
        self.queries.extend(embedding_batcher.queries[start_pos:end_pos])
        self.embeddings[self.num_pending:self.num_pending + num_to_add] = embedding_batcher.embeddings_list[start_pos:end_pos]
        self.num_pending += num_to_add
        return end_pos

    def reset(self):
        self.question_ids = []
        self.queries = []
        self.embeddings = np.empty((self.max_batch_size, self.embeddings.shape[1]), dtype=np.float32)
        self.num_pending = 0

class SearchResultBatchManager:
    def __init__(self):
        self.question_ids = []  # list[int]
        self.queries = []       # list[str]
        self.doc_ids = []       # list[list[int]]
        self.num_queries = 0
        self.text_pos = {}  # qid -> (offset, length)
        self._bytes: np.ndarray = np.array([], dtype=np.uint8)
    
    def add_result(self, question_id, query, doc_ids):
        self.question_ids.append(question_id)
        self.queries.append(query)
        self.doc_ids.append(doc_ids)
        self.num_queries += 1
    
    def serialize(self) -> np.ndarray:
        """
        Serializes the batch with the following layout:
          [Header][Metadata records][Concatenated query texts][Flattened doc_ids block]

        Header (dtype header_type):
          - count: uint32, the number of queries
          - doc_ids_start: uint32, byte offset to the start of the flattened doc_ids block

        Metadata (dtype metadata_type) for each query:
          - question_id: uint64
          - text_position: uint32 (absolute offset to the query text)
          - text_length: uint32   (length in bytes of the query text)
          - doc_ids_position: uint32 (absolute offset to the doc_ids for this query)
          - doc_ids_count: uint32    (number of doc_ids for this query)
        """
        count = self.num_queries
        header_dtype = np.dtype([
            ('count', np.uint32),
            ('doc_ids_start', np.uint32)
        ])
        metadata_dtype = np.dtype([
            ('question_id', np.uint64),
            ('text_position', np.uint32),
            ('text_length', np.uint32),
            ('doc_ids_position', np.uint32),
            ('doc_ids_count', np.uint32)
        ])
        header_size = header_dtype.itemsize
        metadata_size = count * metadata_dtype.itemsize

        # Encode queries into UTF-8 bytes.
        text_bytes_list = [q.encode('utf-8') for q in self.queries]
        text_lengths = [len(b) for b in text_bytes_list]
        total_text_size = sum(text_lengths)

        # Flatten doc_ids from each query.
        flattened_doc_ids = []
        doc_ids_counts = []
        for docs in self.doc_ids:
            doc_ids_counts.append(len(docs))
            flattened_doc_ids.extend(docs)
        total_doc_ids_count = len(flattened_doc_ids)
        # Assume each doc_id is stored as a 64-bit integer.
        doc_ids_dtype = np.uint64
        doc_ids_itemsize = np.dtype(doc_ids_dtype).itemsize
        total_doc_ids_size = total_doc_ids_count * doc_ids_itemsize

        # Calculate offsets.
        doc_ids_start = header_size + metadata_size + total_text_size
        total_size = header_size + metadata_size + total_text_size + total_doc_ids_size

        # Allocate the output buffer.
        buffer = np.zeros(total_size, dtype=np.uint8)

        # Fill header.
        header_array = np.frombuffer(buffer[:header_size], dtype=header_dtype)
        header_array[0] = (count, doc_ids_start)

        # Fill metadata records.
        metadata_array = np.frombuffer(buffer[header_size:header_size+metadata_size], dtype=metadata_dtype)
        text_block_start = header_size + metadata_size
        current_text_offset = 0
        current_doc_ids_offset = 0
        for i in range(count):
            text_pos = text_block_start + current_text_offset
            text_len = text_lengths[i]
            doc_ids_pos = doc_ids_start + current_doc_ids_offset
            doc_count = doc_ids_counts[i]
            metadata_array[i] = (self.question_ids[i], text_pos, text_len, doc_ids_pos, doc_count)
            current_text_offset += text_len
            current_doc_ids_offset += doc_count * doc_ids_itemsize

        # Write the text block.
        current_text_offset = 0
        for b in text_bytes_list:
            start = text_block_start + current_text_offset
            end = start + len(b)
            buffer[start:end] = np.frombuffer(b, dtype=np.uint8)
            current_text_offset += len(b)

        # Write the doc_ids block.
        if total_doc_ids_count > 0:
            doc_ids_array = np.array(flattened_doc_ids, dtype=doc_ids_dtype)
            doc_ids_bytes = doc_ids_array.view(np.uint8)
            start = doc_ids_start
            end = start + total_doc_ids_size
            buffer[start:end] = doc_ids_bytes

        return buffer

    def deserialize(self, buffer: np.ndarray):
        """
        Deserializes the given buffer and directly assigns the resulting data
        to the instance variables of self. The layout is:

          [Header][Metadata records][Concatenated query texts][Flattened doc_ids block]

        Uses memory views and np.frombuffer to avoid unnecessary data copies.
        """
        mv = memoryview(buffer)
        header_dtype = np.dtype([
            ('count', np.uint32),
            ('doc_ids_start', np.uint32)
        ])
        metadata_dtype = np.dtype([
            ('question_id', np.uint64),
            ('text_position', np.uint32),
            ('text_length', np.uint32),
            ('doc_ids_position', np.uint32),
            ('doc_ids_count', np.uint32)
        ])
        # Read header.
        header = np.frombuffer(mv, dtype=header_dtype, count=1)[0]
        count = int(header['count'])
        doc_ids_start = int(header['doc_ids_start'])
        header_size = header_dtype.itemsize
        metadata_size = count * metadata_dtype.itemsize

        # Read metadata as a view.
        metadata = np.frombuffer(mv, dtype=metadata_dtype, count=count, offset=header_size)

        # Directly assign deserialized values to self.
        text_block_start = header_size + metadata_size
        doc_ids_dtype = np.uint64  # as stored

        for rec in metadata:
            # Append question id.
            self.question_ids.append(int(rec['question_id']))
            # Decode and append the query text.
            text_pos = int(rec['text_position'])
            text_len = int(rec['text_length'])
            text_bytes = mv[text_pos:text_pos + text_len].tobytes()
            self.queries.append(text_bytes.decode('utf-8'))
            # Create a zero-copy view for the doc_ids.
            doc_pos = int(rec['doc_ids_position'])
            doc_count = int(rec['doc_ids_count'])
            if doc_count > 0:
                doc_ids_view = np.frombuffer(mv, dtype=doc_ids_dtype, count=doc_count, offset=doc_pos)
                self.doc_ids.append(doc_ids_view)
            else:
                self.doc_ids.append(np.array([], dtype=doc_ids_dtype))
        self.num_queries = count
        
    def print_info(self):
        print(f"SearchResultBatchManager: {self.num_queries} queries")
        for i, q in enumerate(self.queries):
            print(f"Query {i}: {q}")
        print(f"Question IDs: {self.question_ids}")
        print(f"Doc IDs: {self.doc_ids}")
        
