#!/usr/bin/env python3
import threading
import time
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
from transformers import AutoImageProcessor
from derecho.cascade.external_client import ServiceClientAPI, TimestampLogger
from serialize_utils import MonoDataBatcher, WebDataBatcher

# Global configuration variables.
BATCH_THRESHOLD = 4  # How many individual batches to accumulate before processing.
incoming_batches = []  # A list to hold received batches.
incoming_lock = threading.Lock()

# Create instances of capi and timestamp logger.
capi = ServiceClientAPI()
tl = TimestampLogger()

# Use a prefix and other parameters as before.
prefix = "/Mono/"
subgroup_type = "VolatileCascadeStoreWithStringKey"
subgroup_index = 0
MONO_SHARD_IDS = [0, 1, 2]
batch_counter = 0  # To keep track of the aggregated batches sent.




app = Flask(__name__)

@app.route('/upload_data', methods=['POST'])
def upload_data():
    """
    Endpoint to receive binary data. It decodes the bytes into a MonoDataBatcher,
    optionally pre-processes the images, and then enqueues the sample for aggregation.
    """
    data_bytes = request.data
    if not data_bytes:
        return jsonify({'error': 'No data received'}), 400

    data_array = np.frombuffer(data_bytes, dtype=np.uint8)
    
    batcher = MonoDataBatcher()
    batcher.deserialize(data_array)
    
    

    # Enqueue the received batcher.
    with incoming_lock:
        incoming_batches.append(batcher)
    
    # Respond immediately to acknowledge receipt.
    return jsonify({'message': 'Data received'}), 200

def batch_processor():
    """
    Background thread that checks the incoming_batches queue. Once enough
    samples are available (or after a timeout), it aggregates them into a single batch,
    serializes the aggregated batch, and sends it with capi.put_nparray().
    """
    global batch_counter
    # You can also implement a timeout-based flush if desired.
    while True:
        time.sleep(0.1)  # Check periodically.
        with incoming_lock:
            if len(incoming_batches) >= BATCH_THRESHOLD:
                # Create a new aggregated batch.
                aggregated_batcher = MonoDataBatcher()
                # Aggregate fields from each received batch.
                for b in incoming_batches:
                    aggregated_batcher.question_ids.extend(b.question_ids)
                    aggregated_batcher.images.extend(b.images)
                    aggregated_batcher.questions.extend(b.questions)
                    aggregated_batcher.text_sequences.extend(b.text_sequences)
                    # If you have additional fields (e.g. pixel_values), aggregate them too.
                    if hasattr(b, "pixel_values"):
                        if not hasattr(aggregated_batcher, "pixel_values"):
                            aggregated_batcher.pixel_values = []
                        aggregated_batcher.pixel_values.extend(b.pixel_values)
                # Clear the queue.
                incoming_batches.clear()
                
                print(f"Aggregated batch of size {len(aggregated_batcher.question_ids)}")
                
                # Serialize the aggregated batch.
                serialized_np = aggregated_batcher.serialize()
                
                # Determine shard id (rotating among MONO_SHARD_IDS).
                shard_id = MONO_SHARD_IDS[batch_counter % len(MONO_SHARD_IDS)]
                
                # Call capi.put_nparray() to send the aggregated batch.
                res = capi.put_nparray(prefix + f"_{batch_counter}",
                                       serialized_np,
                                       subgroup_type=subgroup_type,
                                       subgroup_index=subgroup_index,
                                       shard_index=shard_id,
                                       message_id=batch_counter,
                                       as_trigger=True,
                                       blokcing=False)
                print(f"Sent aggregated batch {batch_counter} with result: {res}")
                batch_counter += 1

if __name__ == "__main__":
    # Start the background batch processor thread.
    processor_thread = threading.Thread(target=batch_processor, daemon=True)
    processor_thread.start()
    
    # Run the Flask server (using threaded mode so that multiple requests can be handled concurrently).
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)