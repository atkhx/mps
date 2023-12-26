package operation

import "C"
import (
	"github.com/atkhx/mps"
	"github.com/atkhx/mps/operation/rmsnorm"
)

func NewOpNormRMS(
	device *mps.MTLDevice,
	inputData *mps.MTLBuffer,
	inputGrad *mps.MTLBuffer,
	outputData *mps.MTLBuffer,
	outputGrad *mps.MTLBuffer,
	chunkSize int,
) *OpNormRMS {
	aggBuff := device.CreateBufferWithLength(inputData.Length / chunkSize)
	aggGrad := device.CreateBufferWithLength(inputData.Length / chunkSize)

	kernel := rmsnorm.New(device.DeviceID)

	return &OpNormRMS{
		device:      device,
		kernel:      kernel,
		aggTempData: aggBuff,
		aggTempGrad: aggGrad,
		inputData:   inputData,
		inputGrad:   inputGrad,
		outputData:  outputData,
		outputGrad:  outputGrad,
		chunkSize:   chunkSize,
	}
}

type OpNormRMS struct {
	device *mps.MTLDevice
	kernel *rmsnorm.Kernel

	inputData *mps.MTLBuffer
	inputGrad *mps.MTLBuffer

	outputData *mps.MTLBuffer
	outputGrad *mps.MTLBuffer

	aggTempData *mps.MTLBuffer
	aggTempGrad *mps.MTLBuffer

	chunkSize int
}

func (op *OpNormRMS) Forward(b *mps.MTLCommandBuffer) {
	b.Exclusive(func() {
		op.kernel.Forward(
			b.ID,
			op.inputData.BufferID,
			op.outputData.BufferID,
			op.aggTempData.BufferID,
			op.chunkSize,
		)
	})
}

func (op *OpNormRMS) Backward(b *mps.MTLCommandBuffer) {
	b.Exclusive(func() {
		op.kernel.Backward(
			b.ID,
			op.inputData.BufferID,
			op.inputGrad.BufferID,
			op.outputData.BufferID,
			op.outputGrad.BufferID,
			op.aggTempData.BufferID,
			op.aggTempGrad.BufferID,
			op.chunkSize,
		)
	})
}
