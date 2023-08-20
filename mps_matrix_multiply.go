package mps

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics
#include "mps_matrix.h"
#include "mps_matrix_multiply.h"
*/
import "C"

func (b *MTLCommandBuffer) MatrixMultiplyAB(aM, bM, cM *Matrix, alpha, beta float32) {
	b.MatrixMultiply(aM, bM, cM, aM.cols, false, false, alpha, beta)
}

func (b *MTLCommandBuffer) MatrixMultiplyATB(aM, bM, cM *Matrix, alpha, beta float32) {
	b.MatrixMultiply(aM, bM, cM, aM.cols, false, true, alpha, beta)
}

func (b *MTLCommandBuffer) MatrixMultiplyTAB(aM, bM, cM *Matrix, alpha, beta float32) {
	b.MatrixMultiply(aM, bM, cM, bM.rows, true, false, alpha, beta)
}

func (b *MTLCommandBuffer) MatrixMultiply(
	aM, bM, cM *Matrix,
	iC int,
	aT, bT bool,
	alpha, beta float32,
) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.uncommitted++

	C.matrixMultiplyOnDeviceWithOffset(
		b.deviceID,
		b.id,

		// buffer IDs
		aM.matrixID,
		bM.matrixID,
		cM.matrixID,

		C.int(iC),

		C.float(alpha),
		C.float(beta),

		C.bool(aT),
		C.bool(bT),
	)
}
