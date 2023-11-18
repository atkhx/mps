package operation

import (
	"github.com/atkhx/mps"
	"github.com/atkhx/mps/custom-kernel"
)

func NewOpTranspose(device *mps.MTLDevice, inputData, inputGrad, outputData, outputGrad *mps.MTLBuffer, width, height int) *OpTranspose {
	return &OpTranspose{
		device:     device,
		inputData:  inputData,
		inputGrad:  inputGrad,
		outputData: outputData,
		outputGrad: outputGrad,
		width:      width,
		height:     height,
	}
}

type OpTranspose struct {
	device *mps.MTLDevice

	inputData *mps.MTLBuffer
	inputGrad *mps.MTLBuffer

	outputData *mps.MTLBuffer
	outputGrad *mps.MTLBuffer

	width  int
	height int
}

func (op *OpTranspose) Forward(b *mps.MTLCommandBuffer) {
	b.Exclusive(func() {
		custom_kernel.TransposeTo(
			op.device.CustomKernels,
			b.ID,
			op.inputData.BufferID,
			op.outputData.BufferID,
			op.width,
			op.height,
		)
	})
}

func (op *OpTranspose) Backward(b *mps.MTLCommandBuffer) {
	b.Exclusive(func() {
		custom_kernel.TransposeAndAddTo(
			op.device.CustomKernels,
			b.ID,
			op.outputGrad.BufferID,
			op.inputGrad.BufferID,
			op.width,
			op.height,
		)
	})
}
