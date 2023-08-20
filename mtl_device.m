#include "mtl_device.h"

void* createDevice() {
    return MTLCreateSystemDefaultDevice();
}

void releaseDevice(void *deviceID) {
    id<MTLDevice> device = (id<MTLDevice>)deviceID;
    [device release];
}
