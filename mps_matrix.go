package mps

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics
#include "mps_matrix.h"
*/
import "C"
import "unsafe"

type Matrix struct {
	matrixID unsafe.Pointer
	descID   unsafe.Pointer
	buffer   *MTLBuffer

	offset int
	length int

	cols int
	rows int
}

func (buffer *MTLBuffer) CreateMatrix(cols, rows, offset int) *Matrix {
	// todo validate offset + cols * rows with buffer length

	descID := C.createMPSMatrixDescriptor(C.int(cols), C.int(rows))

	matrix := &Matrix{
		matrixID: C.createMPSMatrix(buffer.bufferID, descID, C.int(offset)),
		descID:   descID,
		buffer:   buffer,
		offset:   offset,
		length:   cols * rows,

		cols: cols,
		rows: rows,
	}

	return matrix
}

func (m *Matrix) GetData() []float32 {
	return m.buffer.GetData()[m.offset : m.offset+m.length]
}

func (m *Matrix) Release() {
	C.releaseMPSMatrixDescriptor(m.descID)
	C.releaseMPSMatrix(m.matrixID)
}
