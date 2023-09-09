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

func TestMTLCommandBuffer_Mul(t *testing.T) {
	device := NewMTLDevice()
	defer device.Release()

	commandQueue := device.CreateCommandQueue()
	defer commandQueue.Release()

	commandBuffer := commandQueue.CreateCommandBuffer()
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

func TestMTLCommandBuffer_SoftmaxBuffer(t *testing.T) {
	device := NewMTLDevice()
	defer device.Release()

	commandQueue := device.CreateCommandQueue()
	defer commandQueue.Release()

	commandBuffer := commandQueue.CreateCommandBuffer()
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

	commandBuffer := commandQueue.CreateCommandBuffer()
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

	commandBuffer := commandQueue.CreateCommandBuffer()
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
