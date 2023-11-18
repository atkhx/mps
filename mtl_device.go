package mps

import (
	"unsafe"

	"github.com/atkhx/mps/custom-kernel"
	"github.com/atkhx/mps/framework"
)

var DefaultDevice *MTLDevice

func InitDefaultDevice() {
	if DefaultDevice == nil {
		DefaultDevice = NewMTLDevice()
	}
}

func ReleaseDefaultDevice() {
	if DefaultDevice != nil {
		DefaultDevice.Release()
	}
}

type Releasable interface {
	Release()
}

func NewMTLDevice() *MTLDevice {
	deviceID := framework.MTLDeviceCreate()
	device := &MTLDevice{
		deviceID:      deviceID,
		CustomKernels: custom_kernel.CustomKernelCreate(deviceID),
	}

	return device
}

type MTLDevice struct {
	deviceID      unsafe.Pointer
	resources     []Releasable
	CustomKernels unsafe.Pointer
}

func (device *MTLDevice) CreateCommandQueue() *MTLCommandQueue {
	queue := NewMTLCommandQueue(device)
	device.regSource(queue)
	return queue
}

func (device *MTLDevice) CreateBufferWithBytes(data []float32) *MTLBuffer {
	buffer := NewMTLBufferWithBytes(device, data)
	device.regSource(buffer)
	return buffer
}

func (device *MTLDevice) CreateBufferWithLength(bfLength int) *MTLBuffer {
	buffer := NewMTLBufferWithLength(device, bfLength)
	device.regSource(buffer)
	return buffer
}

func (device *MTLDevice) CreateMatrixMultiplyKernel(
	resultRows int,
	resultColumns int,
	interiorColumns int,
	alpha float32,
	beta float32,
	transposeLeft bool,
	transposeRight bool,
) unsafe.Pointer {
	return framework.MPSMatrixMultiplicationCreate(
		device.deviceID,
		resultRows,
		resultColumns,
		interiorColumns,
		alpha,
		beta,
		transposeLeft,
		transposeRight,
	)

}

type MatrixRandomDistribution struct {
	id unsafe.Pointer
}

func (device *MTLDevice) CreateMatrixRandomDistribution(min, max float32) *MatrixRandomDistribution {
	return &MatrixRandomDistribution{id: framework.MPSMatrixRandomDistributionDescriptorCreate(min, max)}
}

type MatrixRandomMTGP32 struct {
	id unsafe.Pointer
}

func (device *MTLDevice) CreateMatrixRandomMTGP32(
	distribution *MatrixRandomDistribution,
	seed uint64,
) *MatrixRandomMTGP32 {
	return &MatrixRandomMTGP32{id: framework.MPSMatrixRandomMTGP32Create(
		device.deviceID,
		distribution.id,
		seed,
	)}
}

func (device *MTLDevice) regSource(source Releasable) {
	device.resources = append(device.resources, source)
}

func (device *MTLDevice) Release() {
	for i := len(device.resources); i > 0; i-- {
		device.resources[i-1].Release()
	}
	device.resources = nil
	framework.MTLDeviceRelease(device.deviceID)
}
