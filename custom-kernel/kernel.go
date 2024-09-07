package custom_kernel

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation

#include "kernel.h"

void* customKernelCreate(void *deviceID, const char *kernelSource) {
    return [[MPSCustomKernelImpl alloc]
        initWithDevice:(id<MTLDevice>)deviceID
        kernelSource:[NSString stringWithUTF8String:kernelSource]];
}

void customKernelRelease(void *kernelID) {
    [(__bridge MPSCustomKernelImpl*)kernelID release];
}

void customKernelFill(
    void *kernelID,
    void *commandBuffer,
    void *dstBuffer,
    float value,
    const uint offset,
    const uint length
) {
    [(__bridge MPSCustomKernelImpl*)kernelID fill:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        value:(float)value
        offset:(uint)offset
        length:(uint)length
	];
}

void customKernelUpdateWithAdam(
    void *kernelID,
    void *commandBufferID,
    void *dataBufferID,
    void *gradBufferID,
    void *mBufferID,
    void *vBufferID,
    float beta1,
    float beta2,
    float beta1powIterationLR,
    float beta2powIteration
) {
    [(__bridge MPSCustomKernelImpl*)kernelID updateWithAdam:(id<MTLCommandBuffer>)commandBufferID
        dataBuffer:(id<MTLBuffer>)dataBufferID
        gradBuffer:(id<MTLBuffer>)gradBufferID
        mBuffer:(id<MTLBuffer>)mBufferID
        vBuffer:(id<MTLBuffer>)vBufferID
        beta1:beta1
        beta2:beta2
        beta1powIterationLR:beta1powIterationLR
        beta2powIteration:beta2powIteration];
}


*/
import "C"
import (
	_ "embed"
	"unsafe"
)

//go:embed kernel.metal
var metalFunctions string

func New(deviceID unsafe.Pointer) *Kernel {
	cKernelString := C.CString(metalFunctions)
	defer C.free(unsafe.Pointer(cKernelString))
	return &Kernel{
		deviceID: deviceID,
		kernelID: C.customKernelCreate(deviceID, cKernelString),
	}
}

type Kernel struct {
	deviceID unsafe.Pointer
	kernelID unsafe.Pointer
}

func (k *Kernel) Release() {
	C.customKernelRelease(k.kernelID)
}

func (k *Kernel) Fill(
	commandBufferID unsafe.Pointer,
	bufferID unsafe.Pointer,
	value float32,
	offset int,
	length int,
) {
	C.customKernelFill(k.kernelID, commandBufferID, bufferID, C.float(value), C.uint(offset*4), C.uint(length*4))
}

func (k *Kernel) UpdateWithAdam(
	commandBufferID unsafe.Pointer,
	dataBufferID unsafe.Pointer,
	gradBufferID unsafe.Pointer,
	mBufferID unsafe.Pointer,
	vBufferID unsafe.Pointer,
	beta1 float32,
	beta2 float32,
	beta1powIterationLR float32,
	beta2powIteration float32,
) {
	C.customKernelUpdateWithAdam(
		k.kernelID,
		commandBufferID,

		dataBufferID,
		gradBufferID,
		mBufferID,
		vBufferID,

		C.float(beta1),
		C.float(beta2),
		C.float(beta1powIterationLR),
		C.float(beta2powIteration),
	)
}
