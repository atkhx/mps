package operation

import (
	"testing"

	"github.com/atkhx/mps"
	"github.com/stretchr/testify/require"
)

func TestOpMulRows(t *testing.T) {
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
		1, 6, 15,
		4, 15, 30,

		2, 9, 20,
		5, 18, 35,
	}

	cGradsValues := []float32{
		2, 4, 6,
		8, 10, 12,

		1, 3, 5,
		7, 9, 11,
	}

	expectedAGrads := []float32{
		2, 12, 30,
		8, 30, 60,

		1, 9, 25,
		7, 27, 55,
	}

	expectedBGrads := []float32{
		71, 121, 187,
	}

	aData := device.CreateBufferWithBytes(aValues)
	aGrads := device.CreateBufferWithLength(len(aValues))

	bData := device.CreateBufferWithBytes(bValues)
	bGrads := device.CreateBufferWithLength(len(bValues))

	cData := device.CreateBufferWithLength(len(expectedResult))
	cGrads := device.CreateBufferWithBytes(cGradsValues)

	b := device.CreateCommandQueue().GetCommandBuffer()
	operation := NewOpMulRows(device, aData, aGrads, bData, bGrads, cData, cGrads, rowWidth)
	operation.Forward(b)
	operation.Backward(b)
	b.Wait()

	require.Equal(t, expectedResult, cData.GetData())
	require.Equal(t, expectedAGrads, aGrads.GetData())
	require.Equal(t, expectedBGrads, bGrads.GetData())
}
