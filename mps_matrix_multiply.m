#include "mps_matrix_multiply.h"

void matrixMultiplyOnDeviceWithOffset(
    void *deviceID,
    void *commandBufferID,

    void *matrixAID,
    void *matrixBID,
    void *matrixCID,

    int _interiorColumns,
    float _alpha, float _beta,
    bool _transposeLeft, bool _transposeRight
) {
    id<MTLDevice> device = (id<MTLDevice>)deviceID;
    id<MTLCommandBuffer> commandBuffer = (id<MTLCommandBuffer>)commandBufferID;

    MPSMatrix *matrixA = (__bridge MPSMatrix*)matrixAID;
    MPSMatrix *matrixB = (__bridge MPSMatrix*)matrixBID;
    MPSMatrix *matrixC = (__bridge MPSMatrix*)matrixCID;

    MPSMatrixMultiplication *kernel = [[MPSMatrixMultiplication alloc]
        initWithDevice:device
        transposeLeft:_transposeLeft
        transposeRight:_transposeRight
        resultRows:matrixC.rows
        resultColumns:matrixC.columns
        interiorColumns:_interiorColumns
        alpha:_alpha
        beta:_beta];

    [kernel encodeToCommandBuffer:commandBuffer leftMatrix:matrixA rightMatrix:matrixB resultMatrix:matrixC];
    [kernel release];
}