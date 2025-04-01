#include <cascade/user_defined_logic_interface.hpp>
#include <iostream>
#include <string>

namespace derecho{
namespace cascade{

#define MY_UUID     "88e80f9c-8800-99eb-8999-0888aa880009"
#define MY_DESC     "UDL that is the sender of the oob."

std::string get_uuid() {
    return MY_UUID;
}

std::string get_description() {
    return MY_DESC;
}

class OOBSenderOCDPO: public OffCriticalDataPathObserver {
    size_t      oob_mr_size     = 1ul << 20; // initialize 1MB data
    size_t      oob_data_size   = 256;      // the data size to receive
    void*       oob_mr_ptr = nullptr;

    void prepare_content(derecho::node_id_t remote_node,
                              uint64_t remote_dest_addr,
                              uint64_t rkey) {
        oob_mr_ptr = aligned_alloc(4096,oob_mr_size);
        if (!oob_mr_ptr) {
            std::cerr << "Failed to allocate oob memory with aligned_alloc(). errno=" << errno << std::endl;
            return -1;
        }
        derecho::memory_attribute_t attr;
        attr.type = derecho::memory_attribute_t::SYSTEM;
        uint64_t rkey = typed_ctxt->get_service_client_ref().get_oob_memory_key(oob_mr_ptr);

        constexpr size_t data_size = 256;
        char data[data_size];
        // Fill the local data buffer (for example, with the letter 'X')
        memset(data, 'X', data_size);

        // Set up the iovec to point to our data buffer.
        struct iovec iov;
        iov.iov_base = data;
        iov.iov_len = data_size;

        // Call the oob_remote_write operation.
        // (In your Derecho code this call will internally invoke the underlying RDMA write.)
        oob_remote_write(remote_node, &iov, 1, remote_dest_addr, rkey, data_size);
        std::cout << "Sender: Called oob_remote_write to send " << data_size << " bytes." << std::endl;

    }

    bool perform_remote_write(const uint64_t& callee_addr, const uint64_t& caller_addr, const uint64_t rkey, const uint64_t size){
        // STEP 1 - validate the memory size
        if ((callee_addr < reinterpret_cast<uint64_t>(oob_mr_ptr)) ||
            ((callee_addr+size) > reinterpret_cast<uint64_t>(oob_mr_ptr) + oob_mr_size)) {
            std::cerr << "callee address:0x" << std::hex << callee_addr << " or size " << size << " is invalid." << std::dec << std::endl;
            return false;
        }
        // STEP 2 - do RDMA write to send the OOB data
        auto& subgroup_handle = group->template get_subgroup<OOBRDMA>(this->subgroup_index);
        struct iovec iov;
        iov.iov_base    = reinterpret_cast<void*>(callee_addr);
        iov.iov_len     = static_cast<size_t>(size);
        subgroup_handle.oob_remote_write(group->get_rpc_caller_id(),&iov,1,caller_addr,rkey,size);
        subgroup_handle.wait_for_oob_op(group->get_rpc_caller_id(),OOB_OP_WRITE,1000);
        return true;
    }


    virtual void OOBSenderOCDPO::ocdpo_handler(const node_id_t sender,
                            const std::string& object_pool_pathname,
                            const std::string& key_string,
                            const ObjectWithStringKey& object,
                            const emit_func_t& emit,
                            DefaultCascadeContextType* typed_ctxt,
                            uint32_t worker_id) override {
        std::cout << "[OOBSenderOCDPO] ABOUT to oob_remote_write" << std::endl;

        uint64_t remote_dest_addr = ; // TODO: Get it from the value
        uint64_t rkey = ;      // TODO: Get it from the value

        bool remote_w_succ = this->perform_remote_write(sender,remote_addr,reinterpret_cast<uint64_t>(get_buffer_laddr),rkey,oob_data_size);
        
        std::cout << "sent remote write" << std::endl;

        // TODO: call put with key /oob_receive/finished_send to the receiver to notify the receiver to check

    }

    static std::shared_ptr<OffCriticalDataPathObserver> ocdpo_ptr;
public:
    static void initialize() {
        if(!ocdpo_ptr) {
            ocdpo_ptr = std::make_shared<OOBSenderOCDPO>();
        }
    }
    static auto get() {
        return ocdpo_ptr;
    }
};

std::shared_ptr<OffCriticalDataPathObserver> OOBSenderOCDPO::ocdpo_ptr;

void initialize(ICascadeContext* ctxt) {
    OOBSenderOCDPO::initialize();
}

std::shared_ptr<OffCriticalDataPathObserver> get_observer(
        ICascadeContext*,const nlohmann::json&) {
    return OOBSenderOCDPO::get();
}

void release(ICascadeContext* ctxt) {
    // nothing to release
    return;
}

} // namespace cascade
} // namespace derecho