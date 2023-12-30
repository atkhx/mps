package operation

import "C"
import (
	"github.com/atkhx/mps"
	"github.com/atkhx/mps/operation/divcols"
	"github.com/atkhx/mps/operation/rmsrows"
)

func NewOpRMSNormByRows2(
	device *mps.MTLDevice,
	inputData *mps.MTLBuffer,
	inputGrad *mps.MTLBuffer,
	outputData *mps.MTLBuffer,
	outputGrad *mps.MTLBuffer,
	chunkSize int,
) *OpRMSNormByRows2 {
	colHeight := inputData.Length / chunkSize
	rmsData := device.CreateBufferWithLength(colHeight)
	rmsGrads := device.CreateBufferWithLength(colHeight)

	rmsRowsKernel := rmsrows.New(device.DeviceID)
	divOnColKernel := divcols.New(device.DeviceID)

	return &OpRMSNormByRows2{
		device: device,

		rmsRowsKernel:  rmsRowsKernel,
		divOnColKernel: divOnColKernel,

		rmsData:    rmsData,
		rmsGrads:   rmsGrads,
		inputData:  inputData,
		inputGrad:  inputGrad,
		outputData: outputData,
		outputGrad: outputGrad,
		colHeight:  colHeight,
		chunkSize:  chunkSize,
	}
}

type OpRMSNormByRows2 struct {
	device *mps.MTLDevice

	rmsRowsKernel  *rmsrows.Kernel
	divOnColKernel *divcols.Kernel

	inputData *mps.MTLBuffer
	inputGrad *mps.MTLBuffer

	outputData *mps.MTLBuffer
	outputGrad *mps.MTLBuffer

	rmsData  *mps.MTLBuffer
	rmsGrads *mps.MTLBuffer

	colHeight int
	chunkSize int
}

func (op *OpRMSNormByRows2) Forward(b *mps.MTLCommandBuffer) {
	b.ClearMTLBuffer(op.rmsGrads)
	b.Exclusive(func() {
		op.rmsRowsKernel.Forward(
			b.ID,
			op.inputData.BufferID,
			op.rmsData.BufferID,
			op.chunkSize,
		)

		op.divOnColKernel.Forward(
			b.ID,
			op.inputData.BufferID,
			op.rmsData.BufferID,
			op.outputData.BufferID,
			op.chunkSize,
			op.colHeight,
		)
	})
}

func (op *OpRMSNormByRows2) Backward(b *mps.MTLCommandBuffer) {
	b.Exclusive(func() {
		op.divOnColKernel.Backward(
			b.ID,
			op.inputData.BufferID,
			op.inputGrad.BufferID,
			op.rmsData.BufferID,
			op.rmsGrads.BufferID,
			op.outputData.BufferID,
			op.outputGrad.BufferID,
			op.chunkSize,
			op.colHeight,
		)

		op.rmsRowsKernel.Backward(
			b.ID,
			op.inputData.BufferID,
			op.rmsData.BufferID,
			op.inputGrad.BufferID,
			op.rmsGrads.BufferID,
			op.chunkSize,
		)
	})
}
