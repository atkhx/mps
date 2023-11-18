package mps

import (
	"unsafe"

	"github.com/atkhx/mps/framework"
)

func NewMPSMatrix(bufferID unsafe.Pointer, cols, rows, batchSize, batchStride, offset int) *MPSMatrix {
	descriptorID := framework.MPSMatrixDescriptorCreate(cols, rows, batchSize, batchStride)
	matrixID := framework.MPSMatrixCreate(bufferID, descriptorID, offset)

	return &MPSMatrix{
		matrixID:     matrixID,
		descriptorID: descriptorID,
	}
}

type MPSMatrix struct {
	matrixID     unsafe.Pointer
	descriptorID unsafe.Pointer
}

func (m *MPSMatrix) Release() {
	framework.MPSMatrixDescriptorRelease(m.descriptorID)
	framework.MPSMatrixRelease(m.matrixID)
}
