package framework

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal
#include "framework_mtl.h"
*/
import "C"
import (
	"unsafe"
)

// MTLDevice

func MTLDeviceCreate() unsafe.Pointer {
	return unsafe.Pointer(C.mtlDeviceCreate())
}

func MTLDeviceRelease(deviceID unsafe.Pointer) {
	C.mtlDeviceRelease(deviceID)
}

// MTLCommandQueue

func MTLCommandQueueCreate(deviceID unsafe.Pointer) unsafe.Pointer {
	return C.mtlCommandQueueCreate(deviceID)
}

func MTLCommandQueueRelease(commandBufferID unsafe.Pointer) {
	C.mtlCommandQueueRelease(commandBufferID)
}

// MTLCommandBuffer

func MTLCommandBufferCreate(commandQueueID unsafe.Pointer) unsafe.Pointer {
	return C.mtlCommandBufferCreate(commandQueueID)
}

func MTLCommandBufferRelease(commandBufferID unsafe.Pointer) {
	C.mtlCommandBufferRelease(commandBufferID)
}

func MTLCommandBufferCommitAndWaitUntilCompleted(commandBufferID unsafe.Pointer) {
	C.mtlCommandBufferCommitAndWaitUntilCompleted(commandBufferID)
}

// MTLBuffer

func MTLBufferCreateCreateWithBytes(deviceID unsafe.Pointer, data []float32) unsafe.Pointer {
	return C.mtlBufferCreateCreateWithBytes(deviceID, (*C.float)(unsafe.Pointer(&data[0])), C.ulong(len(data)))
}

func MTLBufferCreateWithLength(deviceID unsafe.Pointer, bfLength int) unsafe.Pointer {
	return C.mtlBufferCreateWithLength(deviceID, C.ulong(bfLength))
}

func MTLBufferCreatePrivateWithLength(deviceID unsafe.Pointer, bfLength int) unsafe.Pointer {
	return C.mtlBufferCreatePrivateWithLength(deviceID, C.ulong(bfLength))
}

func MTLBufferGetContents(bufferID unsafe.Pointer) unsafe.Pointer {
	return unsafe.Pointer(C.mtlBufferGetContents(bufferID))
}

func MTLBufferRelease(bufferID unsafe.Pointer) {
	C.mtlBufferRelease(bufferID)
}
