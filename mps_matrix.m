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
    MPSMatrix *matrixA = [[MPSMatrix alloc]
        initWithBuffer:(id<MTLBuffer>)bufferID
        offset:offset*sizeof(float)
        descriptor:(__bridge MPSMatrixDescriptor*)descriptorID];

    return (__bridge void*)matrixA;
}

void releaseMPSMatrix(void *matrixID) {
    MPSMatrix *matrix = (__bridge MPSMatrix*)matrixID;
    [matrix release];
}