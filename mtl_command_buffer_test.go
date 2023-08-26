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

	commandBuffer := commandQueue.CreateCommandBuffer()
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

func TestMTLCommandBuffer_ClearMTLBuffer(t *testing.T) {
	device := NewMTLDevice()
	defer device.Release()

	commandQueue := device.CreateCommandQueue()
	defer commandQueue.Release()

	commandBuffer := commandQueue.CreateCommandBuffer()
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

	commandBuffer := commandQueue.CreateCommandBuffer()
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

	commandBuffer := commandQueue.CreateCommandBuffer()
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

	commandBuffer := commandQueue.CreateCommandBuffer()
	defer commandBuffer.Release()

	destinationBuffer := device.CreateNewBufferWithLength(9)
	defer destinationBuffer.Release()

	commandBuffer.FillMTLBufferPart(destinationBuffer, 1, 0, 5)
	commandBuffer.FillMTLBufferPart(destinationBuffer, 2, 2, 5)
	commandBuffer.FillMTLBufferPart(destinationBuffer, 3, 4, 5)
	commandBuffer.Wait()

	require.Equal(t, []float32{1, 1, 2, 2, 3, 3, 3, 3, 3}, destinationBuffer.GetData())
}

func TestMTLCommandBuffer_MulBuffer(t *testing.T) {
	device := NewMTLDevice()
	defer device.Release()

	commandQueue := device.CreateCommandQueue()
	defer commandQueue.Release()

	commandBuffer := commandQueue.CreateCommandBuffer()
	defer commandBuffer.Release()

	destinationBuffer := device.CreateBufferWithBytes([]float32{1, 2, 3, 4, 5})
	defer destinationBuffer.Release()

	multiplierBuffer := device.CreateBufferWithBytes([]float32{5, 4, 3, 2, 1})
	defer multiplierBuffer.Release()

	commandBuffer.MulBuffer(destinationBuffer, multiplierBuffer)
	commandBuffer.Wait()

	require.Equal(t, []float32{5, 8, 9, 8, 5}, destinationBuffer.GetData())
}

func TestMTLCommandBuffer_DropoutBuffer(t *testing.T) {
	device := NewMTLDevice()
	defer device.Release()

	commandQueue := device.CreateCommandQueue()
	defer commandQueue.Release()

	commandBuffer := commandQueue.CreateCommandBuffer()
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
		switch v := maskOutBuffer.GetData()[i]; v {
		case 1.0:
			require.Equal(t, sourceBuffer.GetData()[i], destinationBuffer.GetData()[i])
		case 0.0:
			require.Zero(t, destinationBuffer.GetData()[i])
		default:
			t.Errorf("invlice maskoutBuffer value: %f", v)
		}
	}
}
