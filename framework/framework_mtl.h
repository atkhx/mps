#include <Metal/Metal.h>

// MTLDevice

void* mtlDeviceCreate();
void mtlDeviceRelease(void *deviceID);

// MTLCommandQueue

void* mtlCommandQueueCreate(void *deviceID);
void mtlCommandQueueRelease(void *commandQueueID);

// MTLCommandBuffer

void* mtlCommandBufferCreate(void *commandQueueID);
void mtlCommandBufferRelease(void *commandBufferID);
void mtlCommandBufferCommitAndWaitUntilCompleted(void *commandBufferID);

// MTLBuffer

void* mtlBufferCreateCreateWithBytes(void *deviceID, float *bytes, size_t length);
void* mtlBufferCreateWithLength(void *deviceID, size_t length);
void* mtlBufferCreatePrivateWithLength(void *deviceID, size_t length);
void* mtlBufferGetContents(void *bufferID);
void mtlBufferRelease(void *bufferID);
