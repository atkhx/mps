package mps

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics
#include "mtl_buffer.h"
*/
import "C"
import (
	"unsafe"
)

type MTLBuffer struct {
	bufferID unsafe.Pointer
	device   *MTLDevice
	contents []float32
	length   int
	released bool
}

func (device *MTLDevice) CreateBufferWithBytes(data []float32) *MTLBuffer {
	bufferID := C.createNewBufferWithBytes(device.deviceID, (*C.float)(unsafe.Pointer(&data[0])), C.ulong(len(data)))
	contents := unsafe.Pointer(C.getBufferContents(bufferID))
	bfLength := len(data)

	byteSlice := (*[1 << 30]byte)(contents)[:bfLength:bfLength]
	float32Slice := *(*[]float32)(unsafe.Pointer(&byteSlice))

	buffer := &MTLBuffer{
		bufferID: bufferID,
		device:   device,
		contents: float32Slice,
		length:   bfLength,
	}

	device.regSource(buffer)
	return buffer
}

func (device *MTLDevice) CreateNewBufferWithLength(bfLength int) *MTLBuffer {
	bufferID := C.createNewBufferWithLength(device.deviceID, C.ulong(bfLength))
	contents := unsafe.Pointer(C.getBufferContents(bufferID))

	byteSlice := (*[1 << 30]byte)(contents)[:bfLength:bfLength]
	float32Slice := *(*[]float32)(unsafe.Pointer(&byteSlice))

	buffer := &MTLBuffer{
		bufferID: bufferID,
		device:   device,
		contents: float32Slice,
		length:   bfLength,
	}

	device.regSource(buffer)
	return buffer
}

func (buffer *MTLBuffer) GetData() []float32 {
	return buffer.contents
}

func (buffer *MTLBuffer) Release() {
	if !buffer.released {
		C.releaseBuffer(buffer.bufferID)
		buffer.released = true
	}
}
