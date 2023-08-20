#include <CoreGraphics/CoreGraphics.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <Metal/Metal.h>

void* createMPSMatrixDescriptor(int cols, int rows);
void releaseMPSMatrixDescriptor(void *descriptorID);

void* createMPSMatrix(void *bufferID, void *descriptorID, int offset);
void releaseMPSMatrix(void *matrixID);
