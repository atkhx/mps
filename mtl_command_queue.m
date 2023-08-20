#include "mtl_command_queue.h"

void* createCommandQueue(void *deviceID) {
    id<MTLDevice> device = (id<MTLDevice>)deviceID;
    id<MTLCommandQueue> commandQueue = [device newCommandQueue];

     return (__bridge void*)commandQueue;
}

void releaseCommandQueue(void *commandQueueID) {
    id<MTLCommandQueue> commandQueue = (id<MTLCommandQueue>)commandQueueID;
    [commandQueue release];
}
