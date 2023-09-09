package mps

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics
#include "mtl_command_queue.h"
#include "mps_matrix_multiply.h"
*/
import "C"
import (
	"unsafe"
)

func (device *MTLDevice) CreateCommandQueue() *MTLCommandQueue {
	queue := &MTLCommandQueue{
		queueID:  C.createCommandQueue(device.deviceID),
		deviceID: device.deviceID,
		device:   device,
	}
	device.regSource(queue)
	return queue
}

type MTLCommandQueue struct {
	queueID  unsafe.Pointer
	deviceID unsafe.Pointer
	device   *MTLDevice
	released bool
	buffer   *MTLCommandBuffer
}

func (d *MTLCommandQueue) GetID() unsafe.Pointer {
	return d.queueID
}

func (b *MTLCommandQueue) Release() {
	if !b.released {
		b.buffer.Release()
		C.releaseCommandQueue(b.queueID)
		b.released = true
	}
}
