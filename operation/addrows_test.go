package operation

import (
	"testing"

	"github.com/atkhx/mps"
	"github.com/stretchr/testify/require"
)

func TestOpAddRows(t *testing.T) {
	device := mps.NewMTLDevice()
	defer device.Release()

	rowWidth := 3

	aValues := []float32{
		1, 2, 3,
		4, 5, 6,

		2, 3, 4,
		5, 6, 7,
	}

	bValues := []float32{1, 3, 5}

	expectedResult := []float32{
		2, 5, 8,
		5, 8, 11,

		3, 6, 9,
		6, 9, 12,
	}

	cGradsValues := []float32{
		2, 4, 6,
		8, 10, 12,

		1, 3, 5,
		7, 9, 11,
	}

	expectedBGrads := []float32{
		18, 26, 34,
	}

	aData := device.CreateBufferWithBytes(aValues)
	aGrads := device.CreateBufferWithLength(len(aValues))

	bData := device.CreateBufferWithBytes(bValues)
	bGrads := device.CreateBufferWithLength(len(bValues))

	cData := device.CreateBufferWithLength(len(expectedResult))
	cGrads := device.CreateBufferWithBytes(cGradsValues)

	b := device.CreateCommandQueue().GetCommandBuffer()
	operation := NewOpAddRows(device, aData, aGrads, bData, bGrads, cData, cGrads, rowWidth)
	operation.Forward(b)
	operation.Backward(b)
	b.Wait()

	require.Equal(t, expectedResult, cData.GetData())
	require.Equal(t, cGradsValues, aGrads.GetData())
	require.Equal(t, expectedBGrads, bGrads.GetData())
}
