#include "nvlink_allocator.cpp"

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

    int device = 0;
    cudaError_t cudaErr = cudaSetDevice(device);
    if (cudaErr != cudaSuccess) {
        std::cerr << "cudaSetDevice failed: " << cudaGetErrorString(cudaErr) << "\n";
        return 1;
    }
    std::cout << "cudaSetDevice(" << device << ") succeeded\n";

    ssize_t size = 1024 * 1024;
    cudaStream_t stream = nullptr;

    std::cout << "Calling mc_nvlink_malloc with size=" << size << ", device=" << device << "\n";
    void *ptr = mc_nvlink_malloc(size, device, stream);

    if (ptr) {
        std::cout << "mc_nvlink_malloc succeeded, ptr=" << ptr << "\n";

        int dmabuf_fd;
        CUresult result = cuMemGetHandleForAddressRange(
            &dmabuf_fd, (CUdeviceptr)ptr, size,
            CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
        if (result != CUDA_SUCCESS) {
            const char *errStr;
            cuGetErrorString(result, &errStr);
            std::cerr << "Failed to retrieve dmabuf for " << (uintptr_t)addr
                       << " cuda error=" << errStr;
            return 1;
        }

        std::cout << "Calling mc_nvlink_free\n";
        mc_nvlink_free(ptr, size, device, stream);
        std::cout << "mc_nvlink_free succeeded\n";
    } else {
        std::cerr << "mc_nvlink_malloc failed\n";
        return 1;
    }

    return 0;
}
