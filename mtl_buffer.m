#include "mtl_buffer.h"

void* createNewBufferWithBytes(void *deviceID, float *bytes, size_t length) {
    id<MTLDevice> device = (id<MTLDevice>)deviceID;
    id<MTLBuffer> buffer = [device
        newBufferWithBytes:bytes
        length:length * sizeof(float)
        options: MTLResourceCPUCacheModeDefaultCache
    ];
     return (__bridge void*)buffer;
}

void* createNewBufferWithLength(void *deviceID, size_t length) {
    id<MTLDevice> device = (id<MTLDevice>)deviceID;
    id<MTLBuffer> buffer = [device
        newBufferWithLength:length * sizeof(float)
        options: MTLResourceCPUCacheModeDefaultCache
    ];
     return (__bridge void*)buffer;
}

void releaseBuffer(void *bufferID) {
    id<MTLBuffer> buffer = (id<MTLBuffer>)bufferID;
    [buffer release];
}

void* getBufferContents(void *bufferID) {
    id<MTLBuffer> buffer = (id<MTLBuffer>)bufferID;
    return [buffer contents];
}

