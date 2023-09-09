package mps

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

	mpsMatrixMultiply(b.deviceID, b.id, aM.matrixID, bM.matrixID, cM.matrixID, iC, aT, bT, alpha, beta)
}
