

inline int parse_batch_id(const std::string& key_string) {
     size_t pos = key_string.find("qb");
     if (pos == std::string::npos) {
          return -1;
     }
     pos += 2; 
     // Extract the number following "qb"
     std::string numberStr;
     while (pos < key_string.size() && std::isdigit(key_string[pos])) {
          numberStr += key_string[pos];
          ++pos;
     }
     if (!numberStr.empty()) {
          return std::stoi(numberStr);
     }
     return -1;
}


/*** 
* Helper function to cdpo_handler()
* @param bytes the bytes object to deserialize
* @param data_size the size of the bytes object
* @param nq the number of queries in the blob object, output. Used by FAISS search.
     type is uint32_t because in previous encode_centroids_search_udl, it is serialized from an unsigned "big" ordered int
* @param query_embeddings the embeddings of the queries, output. 
***/
void deserialize_embeddings_and_quries_from_bytes(const uint8_t* bytes,
                                                            const std::size_t& data_size,
                                                            uint32_t& nq,
                                                            const int& emb_dim,
                                                            float*& query_embeddings,
                                                            std::vector<std::string>& query_list) {
     if (data_size < 4) {
          throw std::runtime_error("Data size is too small to deserialize its embeddings and queries.");
     }
     
     // 0. get the number of queries in the blob object
     nq = (static_cast<uint32_t>(bytes[0]) << 24) |
                    (static_cast<uint32_t>(bytes[1]) << 16) |
                    (static_cast<uint32_t>(bytes[2]) <<  8) |
                    (static_cast<uint32_t>(bytes[3]));
     dbg_default_debug("In [{}],Number of queries: {}",__func__,nq);
     // 1. get the emebddings of the queries from the blob object
     std::size_t float_array_start = 4;
     std::size_t float_array_size = sizeof(float) * emb_dim * nq;
     std::size_t float_array_end = float_array_start + float_array_size;
     if (data_size < float_array_end) {
          std::cerr << "Data size "<< data_size <<" is too small for the expected float array end: " << float_array_end <<"." << std::endl;
          return;
     }
     query_embeddings = const_cast<float*>(reinterpret_cast<const float*>(bytes + float_array_start));

     // 2. get the queries from the blob object
     std::size_t json_start = float_array_end;
     if (json_start >= data_size) {
          std::cerr << "No space left for queries data." << std::endl;
          return;
     }
     // Create a JSON string from the remainder of the bytes object
     char* json_data = const_cast<char*>(reinterpret_cast<const char*>(bytes + json_start));
     std::size_t json_size = data_size - json_start;
     std::string json_string(json_data, json_size);

     // Parse the JSON string using nlohmann/json
     try {
          nlohmann::json parsed_json = nlohmann::json::parse(json_string);
          query_list = parsed_json.get<std::vector<std::string>>();
     } catch (const nlohmann::json::parse_error& e) {
          std::cerr << "JSON parse error: " << e.what() << std::endl;
     }
}