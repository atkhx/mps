package operation

import (
	"unsafe"

	"github.com/atkhx/mps"
	"github.com/atkhx/mps/operation/roperows"
)

func NewOpRopeRows(
	device *mps.MTLDevice,
	inputData *mps.MTLBuffer,
	inputGrad *mps.MTLBuffer,
	outputData *mps.MTLBuffer,
	outputGrad *mps.MTLBuffer,
	featuresCount int,
	headSize int,
	contextLength int,
) *OpRopeRows {
	return &OpRopeRows{
		kernel:        roperows.New(device.DeviceID),
		inputData:     inputData.BufferID,
		inputGrad:     inputGrad.BufferID,
		outputData:    outputData.BufferID,
		outputGrad:    outputGrad.BufferID,
		featuresCount: featuresCount,
		headSize:      headSize,
		contextLength: contextLength,
	}
}

type OpRopeRows struct {
	kernel *roperows.Kernel

	inputData unsafe.Pointer
	inputGrad unsafe.Pointer

	outputData unsafe.Pointer
	outputGrad unsafe.Pointer

	featuresCount int
	headSize      int
	contextLength int
}

func (op *OpRopeRows) Forward(b *mps.MTLCommandBuffer) {
	b.Exclusive(func() {
		op.kernel.Forward(
			b.ID,
			op.inputData,
			op.outputData,
			op.featuresCount,
			op.headSize,
			op.contextLength,
		)
	})
}

func (op *OpRopeRows) Backward(b *mps.MTLCommandBuffer) {
	b.Exclusive(func() {
		op.kernel.Backward(
			b.ID,
			op.inputGrad,
			op.outputGrad,
			op.featuresCount,
			op.headSize,
			op.contextLength,
		)
	})
}
