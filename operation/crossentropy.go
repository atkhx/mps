package operation

import (
	"github.com/atkhx/mps"
)

func NewOpCrossEntropy(device *mps.MTLDevice, inputData, inputGrad, outputData, outputGrad, targets *mps.MTLBuffer, chunkSize int) *OpCrossEntropy {
	softmaxData := device.CreateBufferWithLength(inputData.Length)
	softmaxGrad := device.CreateBufferWithLength(inputData.Length)

	return &OpCrossEntropy{
		softmaxData:      softmaxData,
		softmaxGrad:      softmaxGrad,
		operationSoftmax: NewOpSoftmax(device, inputData, inputGrad, softmaxData, softmaxGrad, chunkSize),
		operationNLL:     NewOpNegLogLikelihood(device, softmaxData, softmaxGrad, outputData, outputGrad, targets, chunkSize),
	}
}

type OpCrossEntropy struct {
	softmaxData      *mps.MTLBuffer
	softmaxGrad      *mps.MTLBuffer
	operationSoftmax *OpSoftmax
	operationNLL     *OpNegLogLikelihood
}

func (op *OpCrossEntropy) Forward(b *mps.MTLCommandBuffer) {
	b.ClearMTLBuffer(op.softmaxData) // todo check if we need this
	b.ClearMTLBuffer(op.softmaxGrad) // todo check if we need this
	op.operationSoftmax.Forward(b)
	op.operationNLL.Forward(b)
}

func (op *OpCrossEntropy) Backward(b *mps.MTLCommandBuffer) {
	op.operationNLL.Backward(b)
	op.operationSoftmax.Backward(b)
}
