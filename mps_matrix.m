// cgo_matrix.m
#include "mps_matrix.h"

void* createMPSMatrixDescriptor(int cols, int rows) {
    MPSMatrixDescriptor *result = [MPSMatrixDescriptor
        matrixDescriptorWithRows:rows
        columns:cols
        rowBytes:cols * sizeof(float)
        dataType:MPSDataTypeFloat32];

    return (__bridge void*)result;
}

void releaseMPSMatrixDescriptor(void *descriptorID) {
    MPSMatrixDescriptor *descriptor = (__bridge MPSMatrixDescriptor*)descriptorID;
    [descriptor release];
}

void* createMPSMatrix(
    void *bufferID,
    void *descriptorID,
    int offset
) {
    id<MTLBuffer> buffer = (id<MTLBuffer>)bufferID;
    MPSMatrixDescriptor *descriptor = (__bridge MPSMatrixDescriptor*)descriptorID;

    MPSMatrix *matrixA = [[MPSMatrix alloc]
        initWithBuffer:buffer
        offset:offset*sizeof(float)
        descriptor:descriptor];

    return (__bridge void*)matrixA;
}

void releaseMPSMatrix(void *matrixID) {
    MPSMatrix *matrix = (__bridge MPSMatrix*)matrixID;
    [matrix release];
}