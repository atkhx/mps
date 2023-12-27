package operation

import (
	"testing"

	"github.com/atkhx/mps"
	"github.com/stretchr/testify/require"
)

func TestOpAddEqual(t *testing.T) {
	device := mps.NewMTLDevice()
	defer device.Release()

	aValues := []float32{1, 2, 3, 4, 5, 6}
	bValues := []float32{2, 3, 4, 5, 6, 7}

	expectedResult := []float32{3, 5, 7, 9, 11, 13}

	cGradsValues := []float32{3, 4, 5, 6, 7, 8}

	aData := device.CreateBufferWithBytes(aValues)
	aGrads := device.CreateBufferWithLength(len(aValues))

	bData := device.CreateBufferWithBytes(bValues)
	bGrads := device.CreateBufferWithLength(len(bValues))

	cData := device.CreateBufferWithLength(len(expectedResult))
	cGrads := device.CreateBufferWithBytes(cGradsValues)

	b := device.CreateCommandQueue().GetCommandBuffer()
	operation := NewOpAddEqual(device, aData, aGrads, bData, bGrads, cData, cGrads)
	operation.Forward(b)
	operation.Backward(b)
	b.Wait()

	require.Equal(t, expectedResult, cData.GetData())
	require.Equal(t, cGradsValues, aGrads.GetData())
	require.Equal(t, cGradsValues, bGrads.GetData())
}
