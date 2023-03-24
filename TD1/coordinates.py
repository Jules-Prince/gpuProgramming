from numba import cuda
import numba as nb
import sys


@cuda.jit
def coordinates1D() :
    local_thread_id = cuda.threadIdx.x
    local_block_id = cuda.blockIdx.x
    global_id = cuda.grid(1)
    computedGlobalId = (local_block_id*cuda.blockDim.x)+local_thread_id
    print("Current thread : ",local_thread_id," Current Block : ",local_block_id," PID : ",global_id," Compute global_id : ",computedGlobalId)

@cuda.jit
def coordinates2D() :
    local_thread_idX = cuda.threadIdx.x
    local_thread_idY = cuda.threadIdx.y
    local_block_idX = cuda.blockIdx.x
    local_block_idY = cuda.blockIdx.y
    global_id = cuda.grid(2)
    computedGlobalIdX = (local_block_idX*cuda.blockDim.x)+local_thread_idX
    computedGlobalIdY = (local_block_idY*cuda.blockDim.y)+local_thread_idY
    print("Current thread : (",local_thread_idX,", ",local_thread_idY,")  Current Block : (",local_block_idX,", ",local_block_idY,
          " PID : ",global_id," Compute global_id : ",computedGlobalIdX, end='')


def init1D() :
    threadsPerBlock = 2#(4,1,1)
    blocksPerGrid = 2 #(1,1,1)
    print("Starting",sys._getframe(  ).f_code.co_name)
    print("threadsPerBlock ", threadsPerBlock)
    print("blocksPerGrid", blocksPerGrid)
    print(str(threadsPerBlock*blocksPerGrid)," threads vont être executé")
    coordinates1D[blocksPerGrid,threadsPerBlock]()
    cuda.synchronize()

def init2D() :
    threadsPerBlock = (2,2)#(4,1,1)
    blocksPerGrid = (2,2) #(1,1,1)
    print("Starting",sys._getframe(  ).f_code.co_name)
    print("threadsPerBlock ", threadsPerBlock)
    print("blocksPerGrid", blocksPerGrid)
    print(str(threadsPerBlock[0]*blocksPerGrid[0]+threadsPerBlock[1]*blocksPerGrid[1]),"threads vont être executé")
    coordinates1D[blocksPerGrid,threadsPerBlock]()
    cuda.synchronize()

if __name__ == '__main__':
    init2D()