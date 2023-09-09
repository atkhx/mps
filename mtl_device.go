package mps

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics
#include "mtl_device.h"
*/
import "C"
import (
	"unsafe"
)

var DefaultDevice *MTLDevice

func InitDefaultDevice() {
	// todo Once
	if DefaultDevice == nil {
		DefaultDevice = NewMTLDevice()
	}
}

func ReleaseDefaultDevice() {
	// todo Once
	if DefaultDevice != nil {
		DefaultDevice.Release()
	}
}

type Releasable interface {
	Release()
}

func NewMTLDevice() *MTLDevice {
	device := &MTLDevice{
		deviceID: unsafe.Pointer(C.createDevice()),
	}

	device.kernels = device.CreateCustomKernels()
	return device
}

type MTLDevice struct {
	deviceID  unsafe.Pointer
	kernels   *MTLCustomKernels
	resources []Releasable
}

func (device *MTLDevice) regSource(source Releasable) {
	device.resources = append(device.resources, source)
}

func (device *MTLDevice) Release() {
	for i := len(device.resources); i > 0; i-- {
		device.resources[i-1].Release()
	}
	device.resources = nil
	C.releaseDevice(device.deviceID)
}
