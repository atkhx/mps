package operation

import "github.com/atkhx/mps"

func NewOpReLu(device *mps.MTLDevice, inputData, inputGrad, outputData, outputGrad *mps.MTLBuffer) *OpReLu {
	return &OpReLu{
		device:     device,
		inputData:  inputData,
		inputGrad:  inputGrad,
		outputData: outputData,
		outputGrad: outputGrad,
	}
}

type OpReLu struct {
	device *mps.MTLDevice

	inputData *mps.MTLBuffer
	inputGrad *mps.MTLBuffer

	outputData *mps.MTLBuffer
	outputGrad *mps.MTLBuffer
}

func (op *OpReLu) Forward(b *mps.MTLCommandBuffer) {
	b.ReLuMTLBuffer(op.outputData, op.inputData)
}

func (op *OpReLu) Backward(b *mps.MTLCommandBuffer) {
	b.ReLuMTLBufferBwd(op.inputGrad, op.outputGrad, op.outputData)
}
