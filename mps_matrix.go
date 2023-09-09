package mps

import "unsafe"

type Matrix struct {
	matrixID     unsafe.Pointer
	descriptorID unsafe.Pointer
	mtlBuffer    *MTLBuffer

	offset int
	length int

	cols int
	rows int
}

func (buffer *MTLBuffer) CreateMatrix(cols, rows, offset int) *Matrix {
	descriptorID := mpsMatrixDescriptorCreate(cols, rows)
	matrixID := mpsMatrixCreate(buffer.bufferID, descriptorID, offset)

	return &Matrix{
		matrixID:     matrixID,
		descriptorID: descriptorID,
		mtlBuffer:    buffer,

		offset: offset,
		length: cols * rows,

		cols: cols,
		rows: rows,
	}
}

func (m *Matrix) GetData() []float32 {
	return m.mtlBuffer.GetData()[m.offset : m.offset+m.length]
}

func (m *Matrix) Release() {
	mpsMatrixDescriptorRelease(m.descriptorID)
	mpsMatrixRelease(m.matrixID)
}
