package mps

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestMTLBuffer_CreateMatrix(t *testing.T) {
	device := NewMTLDevice()
	defer device.Release()

	buffer := device.CreateNewBufferWithLength(3 * 3)
	defer buffer.Release()

	matrix := buffer.CreateMatrix(3, 3, 0)
	defer matrix.Release()

	matrixInitialData := []float32{
		1, 1, 1,
		2, 2, 2,
		3, 3, 3,
	}

	copy(buffer.GetData(), matrixInitialData)
	require.Equal(t, matrixInitialData, matrix.GetData())
}

func TestMTLBuffer_CreateMatrix_BufferWithBytes(t *testing.T) {
	device := NewMTLDevice()
	defer device.Release()

	matrixInitialData := []float32{
		1, 1, 1,
		2, 2, 2,
		3, 3, 3,
	}

	buffer := device.CreateBufferWithBytes(matrixInitialData)
	defer buffer.Release()

	matrix := buffer.CreateMatrix(3, 3, 0)
	defer matrix.Release()

	copy(buffer.GetData(), matrixInitialData)
	require.Equal(t, matrixInitialData, matrix.GetData())
}
