package operation

import "C"
import (
	"github.com/atkhx/mps"
	"github.com/atkhx/mps/operation/rmsrows"
)

func NewOpRMSByRows(
	device *mps.MTLDevice,
	inputData *mps.MTLBuffer,
	inputGrad *mps.MTLBuffer,
	outputData *mps.MTLBuffer,
	outputGrad *mps.MTLBuffer,
	chunkSize int,
) *OpRMSByRows {
	kernel := rmsrows.New(device.DeviceID)

	return &OpRMSByRows{
		device:     device,
		kernel:     kernel,
		inputData:  inputData,
		inputGrad:  inputGrad,
		outputData: outputData,
		outputGrad: outputGrad,
		chunkSize:  chunkSize,
	}
}

type OpRMSByRows struct {
	device *mps.MTLDevice
	kernel *rmsrows.Kernel

	inputData *mps.MTLBuffer
	inputGrad *mps.MTLBuffer

	outputData *mps.MTLBuffer
	outputGrad *mps.MTLBuffer

	chunkSize int
}

func (op *OpRMSByRows) Forward(b *mps.MTLCommandBuffer) {
	b.Exclusive(func() {
		op.kernel.Forward(
			b.ID,
			op.inputData.BufferID,
			op.outputData.BufferID,
			op.chunkSize,
		)
	})
}

func (op *OpRMSByRows) Backward(b *mps.MTLCommandBuffer) {
	b.Exclusive(func() {
		op.kernel.Backward(
			b.ID,
			op.inputData.BufferID,
			op.outputData.BufferID,
			op.inputGrad.BufferID,
			op.outputGrad.BufferID,
			op.chunkSize,
		)
	})
}
