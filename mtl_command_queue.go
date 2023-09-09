package mps

import "C"
import "unsafe"

func (device *MTLDevice) CreateCommandQueue() *MTLCommandQueue {
	queue := &MTLCommandQueue{
		queueID: mtlCommandQueueCreate(device.deviceID),
		device:  device,
	}
	device.regSource(queue)
	return queue
}

type MTLCommandQueue struct {
	queueID  unsafe.Pointer
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
		mtlCommandQueueRelease(b.queueID)
		b.released = true
	}
}
