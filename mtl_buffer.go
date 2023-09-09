package mps

import "unsafe"

type MTLBuffer struct {
	bufferID unsafe.Pointer
	device   *MTLDevice
	contents []float32
	length   int
	released bool
}

func (device *MTLDevice) CreateBufferWithBytes(data []float32) *MTLBuffer {
	bufferID := mtlBufferCreateCreateWithBytes(device.deviceID, data)
	contents := mtlBufferGetContents(bufferID)
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
	bufferID := mtlBufferCreateWithLength(device.deviceID, bfLength)
	contents := mtlBufferGetContents(bufferID)

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
		mtlBufferRelease(buffer.bufferID)
		buffer.released = true
	}
}
