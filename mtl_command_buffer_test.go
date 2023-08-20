package mps

import (
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
