#include <cascade/user_defined_logic_interface.hpp>
#include <iostream>
#include <sys/uio.h>
#include <cstdlib>
#include <cstring>
#include <iostream>

namespace derecho{
namespace cascade{

#define MY_UUID     "88e80f9c-8800-99eb-8999-0888aa880009"
#define MY_DESC     "UDL that is the receiver of the oob."

std::string get_uuid() {
    return MY_UUID;
}

std::string get_description() {
    return MY_DESC;
}

class OOBReceiverOCDPO: public OffCriticalDataPathObserver {

    size_t      oob_mr_size     = 1ul << 20; // initialize 1MB data
    size_t      oob_data_size   = 256;      // the data size to receive
    void*       oob_mr_ptr = nullptr;


    void print_data (void* addr,size_t size) {
        std::cout << "data@0x" << std::hex << reinterpret_cast<uint64_t>(addr) << " [ " << std::endl;

        for (uint32_t cidx=0;cidx<16;cidx++) {
            std::cout << reinterpret_cast<char*>(addr)[cidx] << " ";
        }
        std::cout << std::endl;
        std::cout << "..." << std::endl;
        for (uint32_t cidx=size - 16;cidx<size;cidx++) {
            std::cout << reinterpret_cast<char*>(addr)[cidx] << " ";
        }

        std::cout << std::dec << std::endl;
        std::cout << "]" << std::endl;
    }
    
    void alloc_mem(){
        // Allocate aligned memory for OOB to write to
        this->oob_mr_ptr = aligned_alloc(4096,oob_mr_size);
        if (!this->oob_mr_ptr) {
            std::cerr << "Failed to allocate oob memory with aligned_alloc(). errno=" << errno << std::endl;
            return -1;
        }
        
        // Initialize the OOB memory
        memset(oob_mr_ptr, 0, oob_mr_size);

        std::cout << "Initialized contents of the buffer" << std::endl;
        std::cout << "==========================" << std::endl;
        print_data(oob_mr_ptr,oob_mr_size);

        derecho::memory_attribute_t attr;
        attr.type = derecho::memory_attribute_t::SYSTEM;
    }

    void send_rdma_write_info(DefaultCascadeContextType* typed_ctxt){
        """
        send the rdma writer the local info to send to
        """
        typed_ctxt->send_rdma_write_info();
        
    }

    void check_written_content(){
        std::cout << "Receiver: Received data is as follows:" << std::endl;
        print_data(oob_mr_ptr, oob_data_size);
    }

    virtual void OOBReceiverOCDPO::ocdpo_handler(const node_id_t sender,
                            const std::string& object_pool_pathname,
                            const std::string& key_string,
                            const ObjectWithStringKey& object,
                            const emit_func_t& emit,
                            DefaultCascadeContextType* typed_ctxt,
                            uint32_t worker_id) override {
        const std::string suffix = "init";
        if (str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0){
            std::cout << "[OOBReceiverOCDPO] init the send" << std::endl;

        }

        const std::string sender_finish_suffix = "finished_send";
        if (str.compare(str.size() - sender_finish_suffix.size(), sender_finish_suffix.size(), sender_finish_suffix) == 0){
            std::cout << "[OOBReceiverOCDPO] received the finished_send from sender" << std::endl;
            this->check_written_content();
        }


    }

    static std::shared_ptr<OffCriticalDataPathObserver> ocdpo_ptr;
public:
    static void initialize() {
        if(!ocdpo_ptr) {
            ocdpo_ptr = std::make_shared<OOBReceiverOCDPO>();
        }
    }
    static auto get() {
        return ocdpo_ptr;
    }
};

std::shared_ptr<OffCriticalDataPathObserver> OOBReceiverOCDPO::ocdpo_ptr;

void initialize(ICascadeContext* ctxt) {
    OOBReceiverOCDPO::initialize();
}

std::shared_ptr<OffCriticalDataPathObserver> get_observer(
        ICascadeContext*,const nlohmann::json&) {
    return OOBReceiverOCDPO::get();
}

void release(ICascadeContext* ctxt) {
    // nothing to release
    return;
}

} // namespace cascade
} // namespace derecho