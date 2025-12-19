#include <cuda.h>
#include <cuda_runtime_api.h>
#include <sys/types.h>

#include <iostream>

extern "C" {
void *mc_nvlink_malloc(ssize_t size, int device, cudaStream_t stream) {
    size_t granularity = 0;
    CUdevice currentDev;
    CUmemAllocationProp prop = {};
    CUmemGenericAllocationHandle handle;
    void *ptr = nullptr;
    int cudaDev;
    int flag = 0;
    CUresult result = cuDeviceGet(&currentDev, device);
    if (result != CUDA_SUCCESS) {
        std::cerr << "cuDeviceGet failed: " << result << "\n";
        return nullptr;
    }
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
    prop.location.id = currentDev;
    result = cuDeviceGetAttribute(
        &flag, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED,
        currentDev);
    if (result != CUDA_SUCCESS) {
        std::cerr << "cuDeviceGetAttribute failed: " << result;
        return nullptr;
    }
    if (flag) prop.allocFlags.gpuDirectRDMACapable = 1;
    result = cuMemGetAllocationGranularity(&granularity, &prop,
                                           CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    if (result != CUDA_SUCCESS) {
        std::cerr << "cuMemGetAllocationGranularity failed: " << result;
        return nullptr;
    }
    // fix size
    size = (size + granularity - 1) & ~(granularity - 1);
    if (size == 0) size = granularity;
    result = cuMemCreate(&handle, size, &prop, 0);
    if (result != CUDA_SUCCESS) {
        std::cerr << "cuMemCreate failed: " << result;
        return nullptr;
    }
    result = cuMemAddressReserve((CUdeviceptr *)&ptr, size, granularity, 0, 0);
    if (result != CUDA_SUCCESS) {
        std::cerr << "cuMemAddressReserve failed: " << result;
        cuMemRelease(handle);
        return nullptr;
    }
    result = cuMemMap((CUdeviceptr)ptr, size, 0, handle, 0);
    if (result != CUDA_SUCCESS) {
        std::cerr << "cuMemMap failed: " << result;
        cuMemAddressFree((CUdeviceptr)ptr, size);
        cuMemRelease(handle);
        return nullptr;
    }
    int device_count;
    cudaGetDeviceCount(&device_count);
    CUmemAccessDesc accessDesc[device_count];
    for (int idx = 0; idx < device_count; ++idx) {
        accessDesc[idx].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        accessDesc[idx].location.id = idx;
        accessDesc[idx].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    }
    result = cuMemSetAccess((CUdeviceptr)ptr, size, accessDesc, device_count);
    if (result != CUDA_SUCCESS) {
        std::cerr << "cuMemSetAccess failed: " << result;
        cuMemUnmap((CUdeviceptr)ptr, size);
        cuMemAddressFree((CUdeviceptr)ptr, size);
        cuMemRelease(handle);
        return nullptr;
    }
    return ptr;
}

void mc_nvlink_free(void *ptr, ssize_t ssize, int device, cudaStream_t stream) {
    CUmemGenericAllocationHandle handle;
    size_t size = 0;
    if (!ptr) return;
    auto result = cuMemRetainAllocationHandle(&handle, ptr);
    if (result != CUDA_SUCCESS) {
        std::cerr << "cuMemRetainAllocationHandle failed: " << result << "\n";
        return;
    }
    result = cuMemGetAddressRange(NULL, &size, (CUdeviceptr)ptr);
    if (result == CUDA_SUCCESS) {
        cuMemUnmap((CUdeviceptr)ptr, size);
        cuMemAddressFree((CUdeviceptr)ptr, size);
    }
    cuMemRelease(handle);
}
}

int main() {
    CUresult result = cuInit(0);
    if (result != CUDA_SUCCESS) {
        std::cerr << "cuInit failed: " << result << "\n";
        return 1;
    }

    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    std::cout << "Device count: " << device_count << "\n";

    if (device_count == 0) {
        std::cerr << "No CUDA devices found\n";
        return 1;
    }

    ssize_t size = 1024 * 1024;
    int device = 0;
    cudaStream_t stream = nullptr;

    std::cout << "Calling mc_nvlink_malloc with size=" << size << ", device=" << device << "\n";
    void *ptr = mc_nvlink_malloc(size, device, stream);

    if (ptr) {
        std::cout << "mc_nvlink_malloc succeeded, ptr=" << ptr << "\n";
        std::cout << "Calling mc_nvlink_free\n";
        mc_nvlink_free(ptr, size, device, stream);
        std::cout << "mc_nvlink_free succeeded\n";
    } else {
        std::cerr << "mc_nvlink_malloc failed\n";
        return 1;
    }

    return 0;
}
