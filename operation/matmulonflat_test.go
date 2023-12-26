package operation

import (
	"fmt"
	"testing"

	"github.com/atkhx/mps"
)

func TestNewOpMatrixMultiplyFlat(t *testing.T) {
	device := mps.NewMTLDevice()
	defer device.Release()

	aGradBuffer := device.CreateBufferWithLength(9)
	aDataBuffer := device.CreateBufferWithBytes([]float32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	})

	bGradBuffer := device.CreateBufferWithLength(6)
	bDataBuffer := device.CreateBufferWithBytes([]float32{
		-0.1,
		-0.1,
		-0.1,

		0.1,
		0.1,
		0.1,
	})

	cGradBuffer := device.CreateBufferWithLength(6)
	for i := range cGradBuffer.GetData() {
		cGradBuffer.GetData()[i] = 1
	}

	cDataBuffer := device.CreateBufferWithLength(6)

	op := NewOpMatrixMultiplyFlat(
		device,
		aDataBuffer,
		aGradBuffer,
		bDataBuffer,
		bGradBuffer,
		cDataBuffer,
		cGradBuffer,
		3, 3, 1,
		1, 3, 2,
		1, 3, 2,
		1,
	)

	commandQueue := device.CreateCommandQueue()
	defer commandQueue.Release()

	commandBuffer := commandQueue.GetCommandBuffer()
	defer commandBuffer.Release()

	op.Forward(commandBuffer)
	op.Backward(commandBuffer)
	commandBuffer.Wait()

	fmt.Println(cDataBuffer.GetData())
	fmt.Println(cGradBuffer.GetData())
	fmt.Println(aGradBuffer.GetData())
	fmt.Println(bGradBuffer.GetData())
}
