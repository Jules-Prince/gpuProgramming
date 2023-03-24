from PIL import Image
from numba import cuda
import numba as nb
import numpy as np
import time
import math
import climage

@cuda.jit
def rgb2black(array, result) :
    global_idX,  global_idY = cuda.grid(2)
    if global_idX<array.shape[0] and global_idY<array.shape[1] :
        result[global_idY][global_idX] = np.uint8(array[global_idY][global_idX][0] * 0.30 + 
                                                  array[global_idY][global_idX][1] * 0.59 + 
                                                  array[global_idY][global_idX][2] * 0.11 )
    
def main():
    start_time = time.time()
    #LOAD IMAGE
    corgiImgColor = Image.open("img/corgi.jpg")
    width, height= corgiImgColor.size
    #IMG TO ARRAY
    corgiArrayColor = np.asarray(corgiImgColor)
    
    #INIT THREAD
    threadsPerBlock = (16,16)
    blocksPerGrid   = (math.ceil(width  / threadsPerBlock[0]),
                       math.ceil(height / threadsPerBlock[1]))
    #SEND ARRAY 
    corgiArrayBlack   = np.zeros(corgiArrayColor.shape, dtype=np.uint8)
    d_corgiArrayColor = cuda.to_device(corgiArrayColor)
    d_corgiArrayBlack = cuda.to_device(corgiArrayBlack)
    
    #EXECUTE KERNEL
    rgb2black[ blocksPerGrid,threadsPerBlock](d_corgiArrayColor, d_corgiArrayBlack)
    cuda.synchronize()

    #SAVE THE IMAGE
    corgiArrayBlack = d_corgiArrayBlack.copy_to_host()
    end_time = time.time()
    corgiImgBlack = Image.fromarray(corgiArrayBlack)
    
    corgiImgBlack.save("img/corgiBnW.jpg")

    print(climage.convert("img/corgi.jpg"))
    print(climage.convert("img/corgiBnW.jpg"))

    print("--- %s seconds ---" % (end_time - start_time))


if __name__ == '__main__':
    main()