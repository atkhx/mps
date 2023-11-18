package operation

import "github.com/atkhx/mps"

func NewOpCrossEntropy(device *mps.MTLDevice, inputData, inputGrad, outputData, outputGrad, targets *mps.MTLBuffer, chunkSize int) *OpCrossEntropy {
	softmax := device.CreateBufferWithLength(inputData.Length)
	aggBuff := device.CreateBufferWithLength(inputData.Length / chunkSize)

	return &OpCrossEntropy{
		device: device,

		inputData: inputData,
		inputGrad: inputGrad,

		outputData: outputData,
		outputGrad: outputGrad,

		targets: targets,
		softmax: softmax,
		aggBuff: aggBuff,

		chunkSize: chunkSize,
	}
}

type OpCrossEntropy struct {
	device *mps.MTLDevice

	inputData *mps.MTLBuffer
	inputGrad *mps.MTLBuffer

	outputData *mps.MTLBuffer
	outputGrad *mps.MTLBuffer

	targets *mps.MTLBuffer
	softmax *mps.MTLBuffer
	aggBuff *mps.MTLBuffer

	chunkSize int
}

func (op *OpCrossEntropy) Forward(b *mps.MTLCommandBuffer) {
	b.CrossEntropyPos(op.outputData, op.inputData, op.softmax, op.aggBuff, op.targets, op.chunkSize)
}

func (op *OpCrossEntropy) Backward(b *mps.MTLCommandBuffer) {
	b.CrossEntropyPosBwd(op.outputGrad, op.inputGrad, op.targets, op.softmax, op.chunkSize)
}
