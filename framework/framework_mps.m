#include "framework_mps.h"

// MPSMatrixDescriptor

void* mpsMatrixDescriptorCreate(int cols, int rows, int batchSize, int batchStride) {
    return [MPSMatrixDescriptor
        matrixDescriptorWithRows:rows
        columns:cols
        matrices:batchSize
        rowBytes:cols * sizeof(float)
        matrixBytes:batchStride * sizeof(float)
        dataType:MPSDataTypeFloat32];
}

void mpsMatrixDescriptorRelease(void *descriptorID) {
    [(__bridge MPSMatrixDescriptor*)descriptorID release];
}

// MPSMatrix

void* mpsMatrixCreate(void *bufferID, void *descriptorID, int offset) {
    return [[MPSMatrix alloc]
        initWithBuffer:(id<MTLBuffer>)bufferID
        offset:offset*sizeof(float)
        descriptor:(__bridge MPSMatrixDescriptor*)descriptorID];
}

void mpsMatrixRelease(void *matrixID) {
    [(__bridge MPSMatrix*)matrixID release];
}

// MPSMatrixMultiplication

void* mpsMatrixMultiplicationCreate(
    void *deviceID,

    int resultRows,
    int resultColumns,
    int interiorColumns,

    float alpha,
    float beta,

    bool transposeLeft,
    bool transposeRight
) {
    return [[MPSMatrixMultiplication alloc]
        initWithDevice:(id<MTLDevice>)deviceID
        transposeLeft:transposeLeft
        transposeRight:transposeRight
        resultRows:resultRows
        resultColumns:resultColumns
        interiorColumns:interiorColumns
        alpha:alpha
        beta:beta];
}

void mpsMatrixMultiplicationEncode(
    void *commandBufferID,
    void *kernelID,
    void *matrixAID,
    void *matrixBID,
    void *matrixCID
) {
    [(__bridge MPSMatrixMultiplication*)kernelID encodeToCommandBuffer:(id<MTLCommandBuffer>)commandBufferID
        leftMatrix:(__bridge MPSMatrix*)matrixAID
        rightMatrix:(__bridge MPSMatrix*)matrixBID
        resultMatrix:(__bridge MPSMatrix*)matrixCID];
}

// MPSMatrixRandomDistributionDescriptor

void* mpsMatrixRandomDistributionDescriptorCreate(float min, float max) {
    return [MPSMatrixRandomDistributionDescriptor uniformDistributionDescriptorWithMinimum:min maximum:max];
}

// MPSMatrixRandomMTGP32

void* mpsMatrixRandomMTGP32Create(void *deviceID, void *distribution, NSUInteger seed) {
    return [[MPSMatrixRandomMTGP32 alloc]
        initWithDevice:(id<MTLDevice>)deviceID
        destinationDataType:MPSDataTypeFloat32
        seed:seed
        distributionDescriptor:(__bridge MPSMatrixRandomDistributionDescriptor*)distribution
    ];
}

void mpsMatrixRandomMTGP32Encode(void *kernelID, void *commandBufferID, void *dstMatrix) {
    [(__bridge MPSMatrixRandomMTGP32*)kernelID
        encodeToCommandBuffer:(id<MTLCommandBuffer>)commandBufferID
        destinationMatrix:(__bridge MPSMatrix*)dstMatrix];
}