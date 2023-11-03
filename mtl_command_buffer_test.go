package mps

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestMTLCommandBuffer_FillMTLBuffer(t *testing.T) {
	device := NewMTLDevice()
	defer device.Release()

	commandQueue := device.CreateCommandQueue()
	defer commandQueue.Release()

	commandBuffer := commandQueue.GetCommandBuffer()
	defer commandBuffer.Release()

	buffer := device.CreateNewBufferWithLength(32)
	defer buffer.Release()

	for i := 0; i < 32; i++ {
		buffer.GetData()[i] = rand.Float32()
	}

	commandBuffer.FillMTLBuffer(buffer, 1)
	commandBuffer.Wait()

	s := float32(0.0)
	for i := 0; i < 32; i++ {
		s += buffer.GetData()[i]
	}
	require.Equal(t, float32(32), s)
}

func TestMTLCommandBuffer_Copy(t *testing.T) {
	device := NewMTLDevice()
	defer device.Release()

	commandQueue := device.CreateCommandQueue()
	defer commandQueue.Release()

	commandBuffer := commandQueue.GetCommandBuffer()
	defer commandBuffer.Release()

	dstBuffer := device.CreateBufferWithBytes([]float32{
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	})
	defer dstBuffer.Release()

	srcBuffer := device.CreateBufferWithBytes([]float32{
		1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
	})
	defer srcBuffer.Release()

	commandBuffer.Copy(dstBuffer, srcBuffer, 3, 2, 5)
	commandBuffer.Wait()

	require.Equal(t, []float32{
		0, 0, 0, 3, 4, 5, 6, 7, 0, 0,
	}, dstBuffer.GetData())
}

func TestMTLCommandBuffer_ClearMTLBuffer(t *testing.T) {
	device := NewMTLDevice()
	defer device.Release()

	commandQueue := device.CreateCommandQueue()
	defer commandQueue.Release()

	commandBuffer := commandQueue.GetCommandBuffer()
	defer commandBuffer.Release()

	buffer := device.CreateNewBufferWithLength(32)
	defer buffer.Release()

	for i := 0; i < 32; i++ {
		buffer.GetData()[i] = rand.Float32()
	}

	commandBuffer.ClearMTLBuffer(buffer)
	commandBuffer.Wait()

	s := float32(0.0)
	for i := 0; i < 32; i++ {
		s += buffer.GetData()[i]
	}
	require.Equal(t, float32(0.0), s)
}

func TestMTLCommandBuffer_ReLuMTLBuffer(t *testing.T) {
	device := NewMTLDevice()
	defer device.Release()

	commandQueue := device.CreateCommandQueue()
	defer commandQueue.Release()

	commandBuffer := commandQueue.GetCommandBuffer()
	defer commandBuffer.Release()

	sourceBuffer := device.CreateNewBufferWithLength(32)
	defer sourceBuffer.Release()

	destinationBuffer := device.CreateNewBufferWithLength(32)
	defer destinationBuffer.Release()

	for i := 0; i < 32; i++ {
		sourceBuffer.GetData()[i] = float32(rand.NormFloat64())
	}

	fmt.Println(sourceBuffer.GetData())
	fmt.Println(destinationBuffer.GetData())
	fmt.Println()

	commandBuffer.ReLuMTLBuffer(destinationBuffer, sourceBuffer)
	commandBuffer.Wait()

	fmt.Println(sourceBuffer.GetData())
	fmt.Println(destinationBuffer.GetData())
	fmt.Println()
}

func TestMTLCommandBuffer_ReLuMTLBufferBwd(t *testing.T) {
	device := NewMTLDevice()
	defer device.Release()

	commandQueue := device.CreateCommandQueue()
	defer commandQueue.Release()

	commandBuffer := commandQueue.GetCommandBuffer()
	defer commandBuffer.Release()

	sourceBuffer := device.CreateNewBufferWithLength(3)
	defer sourceBuffer.Release()

	destinationBuffer := device.CreateNewBufferWithLength(3)
	defer destinationBuffer.Release()

	maskBuffer := device.CreateNewBufferWithLength(3)
	defer maskBuffer.Release()

	copy(sourceBuffer.GetData(), []float32{0.15, 0.34, 0.9})
	copy(maskBuffer.GetData(), []float32{-1, 0, 1.6})

	commandBuffer.ClearMTLBuffer(destinationBuffer)
	commandBuffer.ReLuMTLBufferBwd(destinationBuffer, sourceBuffer, maskBuffer)
	commandBuffer.Wait()

	require.Equal(t, []float32{0, 0, 0.9}, destinationBuffer.GetData())
}

func TestMTLCommandBuffer_Sequence(t *testing.T) {
	device := NewMTLDevice()
	defer device.Release()

	commandQueue := device.CreateCommandQueue()
	defer commandQueue.Release()

	commandBuffer := commandQueue.GetCommandBuffer()
	defer commandBuffer.Release()

	destinationBuffer := device.CreateNewBufferWithLength(9)
	defer destinationBuffer.Release()

	commandBuffer.FillMTLBufferPart(destinationBuffer, 1, 0, 5)
	commandBuffer.FillMTLBufferPart(destinationBuffer, 2, 2, 5)
	commandBuffer.FillMTLBufferPart(destinationBuffer, 3, 4, 5)
	commandBuffer.Wait()

	require.Equal(t, []float32{1, 1, 2, 2, 3, 3, 3, 3, 3}, destinationBuffer.GetData())
}

func TestMTLCommandBuffer_Add(t *testing.T) {
	device := NewMTLDevice()
	defer device.Release()

	commandQueue := device.CreateCommandQueue()
	defer commandQueue.Release()

	commandBuffer := commandQueue.GetCommandBuffer()
	defer commandBuffer.Release()

	dstBuffer := device.CreateBufferWithBytes([]float32{1, 2, 3, 4, 5})
	defer dstBuffer.Release()

	srcBuffer1 := device.CreateBufferWithBytes([]float32{5, 4, 3, 2, 1})
	defer srcBuffer1.Release()

	srcBuffer2 := device.CreateBufferWithBytes([]float32{1, 1, 1, 1, 1})
	defer srcBuffer2.Release()

	commandBuffer.Add(dstBuffer, srcBuffer1, 2, 1, 2)
	commandBuffer.Add(dstBuffer, srcBuffer2, 0, 0, 5)
	commandBuffer.Wait()

	require.Equal(t, []float32{2, 3, 8, 8, 6}, dstBuffer.GetData())
}

func TestMTLCommandBuffer_AddTo(t *testing.T) {
	device := NewMTLDevice()
	defer device.Release()

	commandQueue := device.CreateCommandQueue()
	defer commandQueue.Release()

	commandBuffer := commandQueue.GetCommandBuffer()
	defer commandBuffer.Release()

	dstBuffer := device.CreateBufferWithBytes([]float32{1, 2, 3, 4, 5})
	defer dstBuffer.Release()

	aBuffer := device.CreateBufferWithBytes([]float32{5, 4, 3, 2, 1})
	defer aBuffer.Release()

	bBuffer := device.CreateBufferWithBytes([]float32{1, 1, 1, 1, 1})
	defer bBuffer.Release()

	commandBuffer.AddTo(dstBuffer, aBuffer, bBuffer)
	commandBuffer.Wait()

	require.Equal(t, []float32{6, 5, 4, 3, 2}, dstBuffer.GetData())
}

func TestMTLCommandBuffer_Mul(t *testing.T) {
	device := NewMTLDevice()
	defer device.Release()

	commandQueue := device.CreateCommandQueue()
	defer commandQueue.Release()

	commandBuffer := commandQueue.GetCommandBuffer()
	defer commandBuffer.Release()

	destination := device.CreateBufferWithBytes([]float32{1, 2, 3, 4, 5})
	defer destination.Release()

	multiplier := device.CreateBufferWithBytes([]float32{5, 4, 3, 2, 1})
	defer multiplier.Release()

	commandBuffer.Mul(destination, multiplier)
	commandBuffer.Wait()

	require.Equal(t, []float32{5, 8, 9, 8, 5}, destination.GetData())
}

func TestMTLCommandBuffer_DropoutBuffer(t *testing.T) {
	device := NewMTLDevice()
	defer device.Release()

	commandQueue := device.CreateCommandQueue()
	defer commandQueue.Release()

	commandBuffer := commandQueue.GetCommandBuffer()
	defer commandBuffer.Release()

	destinationBuffer := device.CreateNewBufferWithLength(32)
	defer destinationBuffer.Release()

	sourceBuffer := device.CreateNewBufferWithLength(32)
	defer sourceBuffer.Release()

	maskOutBuffer := device.CreateNewBufferWithLength(32)
	defer maskOutBuffer.Release()

	for i := 0; i < len(destinationBuffer.GetData()); i++ {
		sourceBuffer.GetData()[i] = rand.Float32()
	}

	commandBuffer.DropoutBuffer(destinationBuffer, sourceBuffer, maskOutBuffer, 0.2)
	commandBuffer.Wait()

	for i := 0; i < len(destinationBuffer.GetData()); i++ {
		if v := maskOutBuffer.GetData()[i]; v > 0.2 {
			require.Equal(t, sourceBuffer.GetData()[i], destinationBuffer.GetData()[i])
		} else {
			require.Zero(t, destinationBuffer.GetData()[i])
		}
	}

	fmt.Println(maskOutBuffer.GetData())
}

func TestMTLCommandBuffer_SoftmaxBuffer(t *testing.T) {
	device := NewMTLDevice()
	defer device.Release()

	commandQueue := device.CreateCommandQueue()
	defer commandQueue.Release()

	commandBuffer := commandQueue.GetCommandBuffer()
	defer commandBuffer.Release()

	colsCount := 3
	rowsCount := 3

	offset := 3
	offsetEnd := 3

	totalLength := offset + (colsCount * rowsCount) + offsetEnd

	destinationBuffer := device.CreateNewBufferWithLength(totalLength)
	defer destinationBuffer.Release()

	sourceBuffer := device.CreateNewBufferWithLength(totalLength)
	defer sourceBuffer.Release()

	sumOutBuffer := device.CreateNewBufferWithLength(rowsCount)
	defer sumOutBuffer.Release()

	for i := offset; i < len(destinationBuffer.GetData())-offsetEnd; i++ {
		sourceBuffer.GetData()[i] = rand.Float32()
	}

	commandBuffer.SoftmaxBuffer(destinationBuffer, sourceBuffer, sumOutBuffer, colsCount, rowsCount, offset)
	commandBuffer.Wait()

	fmt.Println(sourceBuffer.GetData())
	fmt.Println(sumOutBuffer.GetData())
	fmt.Println(destinationBuffer.GetData())
}

func TestMTLCommandBuffer_SoftmaxBufferTril(t *testing.T) {
	device := NewMTLDevice()
	defer device.Release()

	commandQueue := device.CreateCommandQueue()
	defer commandQueue.Release()

	commandBuffer := commandQueue.GetCommandBuffer()
	defer commandBuffer.Release()

	colsCount := 3
	rowsCount := 2

	offset := 0
	offsetEnd := 0

	totalLength := offset + (colsCount * rowsCount) + offsetEnd

	destinationBuffer := device.CreateNewBufferWithLength(totalLength)
	defer destinationBuffer.Release()

	sourceBuffer := device.CreateNewBufferWithLength(totalLength)
	defer sourceBuffer.Release()

	//maxOutBuffer := device.CreateNewBufferWithLength(rowsCount)
	//defer maxOutBuffer.Release()

	//sumOutBuffer := device.CreateNewBufferWithLength(rowsCount)
	//defer sumOutBuffer.Release()

	for i := offset; i < len(destinationBuffer.GetData())-offsetEnd; i++ {
		sourceBuffer.GetData()[i] = rand.Float32()
	}

	//for y := 0; y < rowsCount; y++ {
	//	for x := 0; x < y+1; x++ {
	//		sourceBuffer.GetData()[y*colsCount+x] = rand.Float32()
	//	}
	//}

	commandBuffer.SoftmaxBufferTril(
		destinationBuffer,
		sourceBuffer,
		//maxOutBuffer,
		//sumOutBuffer,
		colsCount,
		rowsCount,
		offset,
	)
	commandBuffer.Wait()

	fmt.Println("src", sourceBuffer.GetData())
	//fmt.Println("max", maxOutBuffer.GetData())
	//fmt.Println("sum", sumOutBuffer.GetData())
	fmt.Println("dst", destinationBuffer.GetData())
}

func TestMTLCommandBuffer_SoftmaxBufferTrilBwd(t *testing.T) {
	device := NewMTLDevice()
	defer device.Release()

	commandQueue := device.CreateCommandQueue()
	defer commandQueue.Release()

	commandBuffer := commandQueue.GetCommandBuffer()
	defer commandBuffer.Release()

	colsCount := 3
	rowsCount := 2

	offset := 0
	offsetEnd := 0

	totalLength := offset + (colsCount * rowsCount) + offsetEnd

	destinationBuffer := device.CreateNewBufferWithLength(totalLength)
	defer destinationBuffer.Release()

	sourceBuffer := device.CreateNewBufferWithLength(totalLength)
	defer sourceBuffer.Release()

	softmaxBuffer := device.CreateNewBufferWithLength(totalLength)
	defer softmaxBuffer.Release()

	//softmaxGradBuffer := device.CreateNewBufferWithLength(totalLength)
	//defer softmaxGradBuffer.Release()

	//sumOutBuffer := device.CreateNewBufferWithLength(rowsCount)
	//defer sumOutBuffer.Release()

	rand.Seed(123)

	for i := offset; i < len(destinationBuffer.GetData())-offsetEnd; i++ {
		sourceBuffer.GetData()[i] = rand.Float32()
		softmaxBuffer.GetData()[i] = rand.Float32()
		//destinationBuffer.GetData()[i] = 1
	}

	commandBuffer.SoftmaxBufferTrilBwd(
		destinationBuffer,
		sourceBuffer,
		softmaxBuffer,
		//softmaxGradBuffer,
		//sumOutBuffer,
		colsCount,
		rowsCount,
		offset,
	)
	commandBuffer.Wait()

	fmt.Println("src", sourceBuffer.GetData())
	fmt.Println("sfm", softmaxBuffer.GetData())
	//fmt.Println("smg", softmaxGradBuffer.GetData())
	//fmt.Println("sum", sumOutBuffer.GetData())
	fmt.Println("dst", destinationBuffer.GetData())
}

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
	// Simple check ability to perform matrix multiplication using common mtlBuffer for each matrix.

	// C = A @ B

	device := NewMTLDevice()
	defer device.Release()

	commandQueue := device.CreateCommandQueue()
	defer commandQueue.Release()

	commandBuffer := commandQueue.GetCommandBuffer()
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

	commandBuffer := commandQueue.GetCommandBuffer()
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

	commandBuffer := commandQueue.GetCommandBuffer()
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

	commandBuffer := commandQueue.GetCommandBuffer()
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

	commandBuffer2 := commandQueue.GetCommandBuffer()
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

	commandBuffer := commandQueue.GetCommandBuffer()
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

	commandBuffer2 := commandQueue.GetCommandBuffer()
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

func TestMTLCommandBuffer_UpdateWithAdam(t *testing.T) {
	device := NewMTLDevice()
	defer device.Release()

	commandQueue := device.CreateCommandQueue()
	defer commandQueue.Release()

	commandBuffer := commandQueue.GetCommandBuffer()
	defer commandBuffer.Release()

	dataBuffer := device.CreateNewBufferWithLength(10)
	gradBuffer := device.CreateNewBufferWithLength(10)
	mBuffer := device.CreateNewBufferWithLength(10)
	vBuffer := device.CreateNewBufferWithLength(10)

	commandBuffer.UpdateWithAdam(dataBuffer, gradBuffer, mBuffer, vBuffer, 0.9, 0.98, 0.8, 0.7)
}
