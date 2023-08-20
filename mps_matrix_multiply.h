#include <CoreGraphics/CoreGraphics.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <Metal/Metal.h>

void matrixMultiplyOnDeviceWithOffset(
    void *deviceID,
    void *commandBufferID,

    void *matrixAID,
    void *matrixBID,
    void *matrixCID,

    int _interiorColumns,
    float _alpha, float _beta,
    bool _transposeLeft, bool _transposeRight
);
