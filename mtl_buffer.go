package mps

import (
	"unsafe"

	"github.com/atkhx/mps/framework"
)

func NewMTLBufferWithBytes(device *MTLDevice, data []float32) *MTLBuffer {
	bufferID := framework.MTLBufferCreateCreateWithBytes(device.deviceID, data)
	contents := framework.MTLBufferGetContents(bufferID)
	bfLength := len(data)

	byteSlice := (*[1 << 30]byte)(contents)[:bfLength:bfLength]
	float32Slice := *(*[]float32)(unsafe.Pointer(&byteSlice))

	return &MTLBuffer{
		BufferID: bufferID,
		device:   device,
		contents: float32Slice,
		Length:   bfLength,
	}
}

func NewMTLBufferWithLength(device *MTLDevice, bfLength int) *MTLBuffer {
	bufferID := framework.MTLBufferCreateWithLength(device.deviceID, bfLength)
	contents := framework.MTLBufferGetContents(bufferID)

	byteSlice := (*[1 << 30]byte)(contents)[:bfLength:bfLength]
	float32Slice := *(*[]float32)(unsafe.Pointer(&byteSlice))

	return &MTLBuffer{
		BufferID: bufferID,
		device:   device,
		contents: float32Slice,
		Length:   bfLength,
	}
}

type MTLBuffer struct {
	BufferID unsafe.Pointer
	device   *MTLDevice
	contents []float32
	Length   int
	released bool
}

func (buffer *MTLBuffer) CreateMatrix(cols, rows, offset int) *MPSMatrix {
	return NewMPSMatrix(buffer.BufferID, cols, rows, 1, 0, offset)
}

func (buffer *MTLBuffer) CreateMatrixBatch(cols, rows, batchSize, batchStride, offset int) *MPSMatrix {
	return NewMPSMatrix(buffer.BufferID, cols, rows, batchSize, batchStride, offset)
}

func (buffer *MTLBuffer) GetData() []float32 {
	return buffer.contents
}

func (buffer *MTLBuffer) Release() {
	if !buffer.released {
		framework.MTLBufferRelease(buffer.BufferID)
		buffer.released = true
	}
}
