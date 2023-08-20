#include <CoreGraphics/CoreGraphics.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <Metal/Metal.h>

void* createNewBufferWithBytes(void *deviceID, float *bytes, size_t length);
void* createNewBufferWithLength(void *deviceID, size_t length);

void* getBufferContents(void *bufferID);
void releaseBuffer(void *bufferID);

