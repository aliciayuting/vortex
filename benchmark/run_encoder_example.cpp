/** Just a simple example of how to start at the encoder udl instead of centroid 
 * search udl :)
 */

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include <cascade/service_client_api.hpp>

#include "../vortex_udls/serialize_utils.hpp"

using namespace derecho::cascade;

#define UDL1_PATH "/rag/emb/encode"
#define UDL2_PATH "/rag/emb/centroids_search"
#define UDL3_PATH "/rag/emb/clusters_search"
#define UDL4_PATH "/rag/generate/agg"

#define UDL1_TIMESTAMP_FILE "udl1.dat"
#define UDL2_TIMESTAMP_FILE "udl2.dat"
#define UDL3_TIMESTAMP_FILE "udl3.dat"
#define UDL4_TIMESTAMP_FILE "udl4.dat" 

#define UDL1_SUBGROUP_INDEX 0
#define UDL2_SUBGROUP_INDEX 1
#define UDL3_SUBGROUP_INDEX 2
#define UDL4_SUBGROUP_INDEX 3
#define UDLS_SUBGROUP_TYPE VolatileCascadeStoreWithStringKey 

#define VORTEX_CLIENT_MAX_WAIT_TIME 60

#define UDL3_SUBGROUP_INDEX 2

const int ID = 0;
ServiceClientAPI& capi = ServiceClientAPI::get_service_client();

int main() {
    const size_t number_iterations = 128;
    const size_t number_queries = 64;
    for (size_t i = 0; i < number_iterations; i++) {

        EncoderQueryBatcher batcher(number_queries);
        for(size_t j = 0; j < number_queries; j++) {
            batcher.add_query(j + number_queries * i, ID, std::make_shared<std::string>("What is the weather today?"));
        }

        batcher.serialize();
        ObjectWithStringKey obj;
        obj.key = UDL1_PATH "/" + std::string("batch") + std::to_string(number_iterations);
        obj.blob = std::move(*batcher.get_blob());
        capi.trigger_put(obj);
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    return 0;
}