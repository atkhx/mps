#include "mtl_custom_kernels.h"

void* createFillKernel(void *deviceID, const char *kernelSource) {
    KernelMTLBufferFillImpl *kernel = [[KernelMTLBufferFillImpl alloc]
        initWithDevice:(id<MTLDevice>)deviceID
        kernelSource:[NSString stringWithUTF8String:kernelSource]];

    return (__bridge void*)kernel;
}

void* createReLUFwdKernel(void *deviceID, const char *kernelSource) {
    KernelMTLBufferReluFwdImpl *kernel = [[KernelMTLBufferReluFwdImpl alloc]
        initWithDevice:(id<MTLDevice>)deviceID
        kernelSource:[NSString stringWithUTF8String:kernelSource]];

    return (__bridge void*)kernel;
}

void* createReLUBwdKernel(void *deviceID, const char *kernelSource) {
    KernelMTLBufferReluBwdImpl *kernel = [[KernelMTLBufferReluBwdImpl alloc]
        initWithDevice:(id<MTLDevice>)deviceID
        kernelSource:[NSString stringWithUTF8String:kernelSource]];

    return (__bridge void*)kernel;
}

void* createMulKernel(void *deviceID, const char *kernelSource) {
    KernelMTLBufferMulImpl *kernel = [[KernelMTLBufferMulImpl alloc]
        initWithDevice:(id<MTLDevice>)deviceID
        kernelSource:[NSString stringWithUTF8String:kernelSource]];

    return (__bridge void*)kernel;
}

void* createDropoutKernel(void *deviceID, const char *kernelSource) {
    KernelMTLBufferDropoutImpl *kernel = [[KernelMTLBufferDropoutImpl alloc]
        initWithDevice:(id<MTLDevice>)deviceID
        kernelSource:[NSString stringWithUTF8String:kernelSource]];

    return (__bridge void*)kernel;
}

void* createSoftmaxBufferTrilKernel(void *deviceID, const char *kernelSource) {
    KernelMTLBufferSoftmaxTrilImpl *kernel = [[KernelMTLBufferSoftmaxTrilImpl alloc]
        initWithDevice:(id<MTLDevice>)deviceID
        kernelSource:[NSString stringWithUTF8String:kernelSource]];

    return (__bridge void*)kernel;
}

void* createSoftmaxBufferTrilBwdKernel(void *deviceID, const char *kernelSource) {
    KernelMTLBufferSoftmaxTrilBwdImpl *kernel = [[KernelMTLBufferSoftmaxTrilBwdImpl alloc]
        initWithDevice:(id<MTLDevice>)deviceID
        kernelSource:[NSString stringWithUTF8String:kernelSource]];

    return (__bridge void*)kernel;
}