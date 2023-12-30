package operation

import "C"
import (
	"github.com/atkhx/mps"
	"github.com/atkhx/mps/operation/rmsnormrows"
)

func NewOpRMSNormByRows(
	device *mps.MTLDevice,
	inputData *mps.MTLBuffer,
	inputGrad *mps.MTLBuffer,
	outputData *mps.MTLBuffer,
	outputGrad *mps.MTLBuffer,
	chunkSize int,
) *OpRMSNormByRows {
	rmsData := device.CreateBufferWithLength(inputData.Length / chunkSize)
	rmsGrad := device.CreateBufferWithLength(inputData.Length / chunkSize)

	kernel := rmsnormrows.New(device.DeviceID)

	return &OpRMSNormByRows{
		device:     device,
		kernel:     kernel,
		rmsData:    rmsData,
		rmsGrad:    rmsGrad,
		inputData:  inputData,
		inputGrad:  inputGrad,
		outputData: outputData,
		outputGrad: outputGrad,
		chunkSize:  chunkSize,
	}
}

type OpRMSNormByRows struct {
	device *mps.MTLDevice
	kernel *rmsnormrows.Kernel

	inputData *mps.MTLBuffer
	inputGrad *mps.MTLBuffer

	outputData *mps.MTLBuffer
	outputGrad *mps.MTLBuffer

	rmsData *mps.MTLBuffer
	rmsGrad *mps.MTLBuffer

	chunkSize int
}

func (op *OpRMSNormByRows) Forward(b *mps.MTLCommandBuffer) {
	b.Exclusive(func() {
		op.kernel.Forward(
			b.ID,
			op.inputData.BufferID,
			op.outputData.BufferID,
			op.rmsData.BufferID,
			op.chunkSize,
		)
	})
}

func (op *OpRMSNormByRows) Backward(b *mps.MTLCommandBuffer) {
	b.Exclusive(func() {
		op.kernel.Backward(
			b.ID,
			op.inputData.BufferID,
			op.inputGrad.BufferID,
			op.outputData.BufferID,
			op.outputGrad.BufferID,
			op.rmsData.BufferID,
			op.rmsGrad.BufferID,
			op.chunkSize,
		)
	})
}
