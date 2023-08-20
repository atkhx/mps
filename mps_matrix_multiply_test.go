package mps

import (
	"testing"

	"github.com/stretchr/testify/require"
)

var (
	testMatrixA1 = []float32{
		1, 0, 1,
		0, 1, 0,
		1, 0, 1,
	}
	testMatrixA2 = []float32{
		0, 1, 0,
		1, 0, 1,
		0, 1, 0,
	}
	testMatrixA3 = []float32{
		1, 0, 1,
		0, 1, 0,
	}
	testMatrixB = []float32{
		1, 2,
		3, 4,
		5, 6,
	}
)

func TestMTLCommandBuffer_MatrixMultiply_OneOperationOnOneBuffer(t *testing.T) {
	// Simple check ability to perform matrix multiplication using common buffer for each matrix.

	// C = A @ B

	device := NewMTLDevice()
	defer device.Release()

	commandQueue := device.CreateCommandQueue()
	defer commandQueue.Release()

	commandBuffer := commandQueue.CreateCommandBuffer()
	defer commandBuffer.Release()

	aW, aH := 3, 3
	bW, bH := 2, 3
	cW, cH := bW, aH

	buffer := device.CreateNewBufferWithLength(aW*aH + bW*bH + cW*cH)
	defer buffer.Release()

	copy(buffer.GetData()[:aW*aH], testMatrixA1)
	copy(buffer.GetData()[aW*aH:aW*aH+bW*bH], testMatrixB)

	matrixA := buffer.CreateMatrix(aW, aH, 0)
	defer matrixA.Release()

	matrixB := buffer.CreateMatrix(bW, bH, aW*aH)
	defer matrixB.Release()

	matrixC := buffer.CreateMatrix(cW, cH, aW*aH+bW*bH)
	defer matrixC.Release()

	commandBuffer.MatrixMultiplyAB(matrixA, matrixB, matrixC, 1, 0)
	commandBuffer.Wait()

	require.Equal(t, []float32{
		1, 0, 1,
		0, 1, 0,
		1, 0, 1,

		1, 2,
		3, 4,
		5, 6,

		6, 8,
		3, 4,
		6, 8,
	}, buffer.GetData())
}

func TestMTLCommandBuffer_MatrixMultiply_TwoOperationsWithCommonMultiplierOnOneBuffer(t *testing.T) {
	// Check ability to use the common matrix B for two multiplications in parallel mode.

	// C1 = A1 @ B
	// C2 = A2 @ B

	device := NewMTLDevice()
	defer device.Release()

	commandQueue := device.CreateCommandQueue()
	defer commandQueue.Release()

	commandBuffer := commandQueue.CreateCommandBuffer()
	defer commandBuffer.Release()

	aW1, aH1 := 3, 3
	aW2, aH2 := 3, 3

	bW, bH := 2, 3

	cW1, cH1 := bW, aH1
	cW2, cH2 := bW, aH2

	buffer := device.CreateNewBufferWithLength(aW1*aH1 + aW2*aH2 + bW*bH + cW1*cH1 + cW2*cH2)
	defer buffer.Release()

	copy(buffer.GetData()[:aW1*aH1], testMatrixA1)
	copy(buffer.GetData()[aW1*aH1:aW1*aH1+aW2*aH2], testMatrixA2)
	copy(buffer.GetData()[aW1*aH1+aW2*aH2:aW1*aH1+aW2*aH2+bW*bH], testMatrixB)

	matrixA1 := buffer.CreateMatrix(aW1, aH1, 0)
	defer matrixA1.Release()

	matrixA2 := buffer.CreateMatrix(aW2, aH2, aW1*aH1)
	defer matrixA2.Release()

	matrixB := buffer.CreateMatrix(bW, bH, aW1*aH1+aW2*aH2)
	defer matrixB.Release()

	matrixC1 := buffer.CreateMatrix(cW1, cH1, aW1*aH1+aW2*aH2+bW*bH)
	defer matrixC1.Release()

	matrixC2 := buffer.CreateMatrix(cW2, cH2, aW1*aH1+aW2*aH2+bW*bH+cW1*cH1)
	defer matrixC2.Release()

	commandBuffer.MatrixMultiplyAB(matrixA1, matrixB, matrixC1, 1, 0)
	commandBuffer.MatrixMultiplyAB(matrixA2, matrixB, matrixC2, 1, 0)
	commandBuffer.Wait()

	require.Equal(t, []float32{
		1, 0, 1,
		0, 1, 0,
		1, 0, 1,

		0, 1, 0,
		1, 0, 1,
		0, 1, 0,

		1, 2,
		3, 4,
		5, 6,

		6, 8,
		3, 4,
		6, 8,

		3, 4,
		6, 8,
		3, 4,
	}, buffer.GetData())
}

func TestMTLCommandBuffer_MatrixMultiply_TwoOperationsWithCommonDestinationOnOneBuffer(t *testing.T) {
	// Check ability to use the common result matrix C for two multiplications in parallel mode.

	// C += A1 @ B
	// C += A2 @ B

	device := NewMTLDevice()
	defer device.Release()

	commandQueue := device.CreateCommandQueue()
	defer commandQueue.Release()

	commandBuffer := commandQueue.CreateCommandBuffer()
	defer commandBuffer.Release()

	aW1, aH1 := 3, 3
	aW2, aH2 := 3, 3

	bW, bH := 2, 3

	cW, cH := bW, aH1

	buffer := device.CreateNewBufferWithLength(aW1*aH1 + aW2*aH2 + bW*bH + cW*cH)
	defer buffer.Release()

	copy(buffer.GetData()[:aW1*aH1], testMatrixA1)
	copy(buffer.GetData()[aW1*aH1:aW1*aH1+aW2*aH2], testMatrixA2)
	copy(buffer.GetData()[aW1*aH1+aW2*aH2:aW1*aH1+aW2*aH2+bW*bH], testMatrixB)

	matrixA1 := buffer.CreateMatrix(aW1, aH1, 0)
	defer matrixA1.Release()

	matrixA2 := buffer.CreateMatrix(aW2, aH2, aW1*aH1)
	defer matrixA2.Release()

	matrixB := buffer.CreateMatrix(bW, bH, aW1*aH1+aW2*aH2)
	defer matrixB.Release()

	matrixC := buffer.CreateMatrix(cW, cH, aW1*aH1+aW2*aH2+bW*bH)
	defer matrixC.Release()

	commandBuffer.MatrixMultiplyAB(matrixA1, matrixB, matrixC, 1, 1)
	commandBuffer.MatrixMultiplyAB(matrixA2, matrixB, matrixC, 1, 1)
	commandBuffer.Wait()

	require.Equal(t, []float32{
		// A1
		1, 0, 1,
		0, 1, 0,
		1, 0, 1,

		// A2
		0, 1, 0,
		1, 0, 1,
		0, 1, 0,

		// B
		1, 2,
		3, 4,
		5, 6,

		// C
		9, 12,
		9, 12,
		9, 12,
	}, buffer.GetData())
}

func TestMTLCommandBuffer_MatrixMultiplyATB_OneOperationOnOneBuffer(t *testing.T) {
	device := NewMTLDevice()
	defer device.Release()

	commandQueue := device.CreateCommandQueue()
	defer commandQueue.Release()

	commandBuffer := commandQueue.CreateCommandBuffer()
	defer commandBuffer.Release()

	aW, aH := 3, 3
	bW, bH := 2, 3
	cW, cH := bW, aH

	buffer := device.CreateNewBufferWithLength(aW*aH + bW*bH + cW*cH)
	defer buffer.Release()

	copy(buffer.GetData()[:aW*aH], testMatrixA1)
	copy(buffer.GetData()[aW*aH:aW*aH+bW*bH], testMatrixB)

	matrixA := buffer.CreateMatrix(aW, aH, 0)
	defer matrixA.Release()

	matrixB := buffer.CreateMatrix(bW, bH, aW*aH)
	defer matrixB.Release()

	matrixC := buffer.CreateMatrix(cW, cH, aW*aH+bW*bH)
	defer matrixC.Release()

	commandBuffer.MatrixMultiplyAB(matrixA, matrixB, matrixC, 1, 0)
	commandBuffer.Wait()

	require.Equal(t, []float32{
		1, 0, 1,
		0, 1, 0,
		1, 0, 1,

		1, 2,
		3, 4,
		5, 6,

		6, 8,
		3, 4,
		6, 8,
	}, buffer.GetData())

	commandBuffer2 := commandQueue.CreateCommandBuffer()
	defer commandBuffer2.Release()

	// A = C @ Bt
	commandBuffer2.MatrixMultiplyATB(matrixC, matrixB, matrixA, 1, 1)
	commandBuffer2.Wait()

	require.Equal(t, []float32{
		// C      @   Bt        +  C
		// 6, 8,      1, 3, 5,     1, 0, 1
		// 3, 4,      2, 4, 6,     0, 1, 0,
		// 6, 8,                   1, 0, 1,

		//6+16+1, 18+32+0, 30+48+1,
		//3+8+0,  9+16+1,  15+24+0,
		//6+16+1, 18+32+0, 30+48+1,

		23, 50, 79,
		11, 26, 39,
		23, 50, 79,
	}, matrixA.GetData())
}

func TestMTLCommandBuffer_MatrixMultiplyTAB_OneOperationOnOneBuffer(t *testing.T) {
	device := NewMTLDevice()
	defer device.Release()

	commandQueue := device.CreateCommandQueue()
	defer commandQueue.Release()

	commandBuffer := commandQueue.CreateCommandBuffer()
	defer commandBuffer.Release()

	aW, aH := 3, 2
	bW, bH := 2, 3
	cW, cH := bW, aH

	buffer := device.CreateNewBufferWithLength(aW*aH + bW*bH + cW*cH)
	defer buffer.Release()

	copy(buffer.GetData()[:aW*aH], testMatrixA3)
	copy(buffer.GetData()[aW*aH:aW*aH+bW*bH], testMatrixB)

	matrixA := buffer.CreateMatrix(aW, aH, 0)
	defer matrixA.Release()

	matrixB := buffer.CreateMatrix(bW, bH, aW*aH)
	defer matrixB.Release()

	matrixC := buffer.CreateMatrix(cW, cH, aW*aH+bW*bH)
	defer matrixC.Release()

	commandBuffer.MatrixMultiplyAB(matrixA, matrixB, matrixC, 1, 0)
	commandBuffer.Wait()

	require.Equal(t, []float32{
		1, 0, 1,
		0, 1, 0,

		1, 2,
		3, 4,
		5, 6,

		6, 8,
		3, 4,
	}, buffer.GetData())

	commandBuffer2 := commandQueue.CreateCommandBuffer()
	defer commandBuffer2.Release()

	// B = At @ C
	commandBuffer2.MatrixMultiplyTAB(matrixA, matrixC, matrixB, 1, 1)
	commandBuffer2.Wait()

	require.Equal(t, []float32{
		// At     @  C      +   B
		// 1, 0,     6, 8,      1, 2
		// 0, 1,     3, 4,      3, 4,
		// 1, 0,                5, 6,

		// 6+0+1, 8+0+2,
		// 0+3+3, 0+4+4,
		// 6+0+5, 8+0+6,

		7, 10,
		6, 8,
		11, 14,
	}, matrixB.GetData())
}
