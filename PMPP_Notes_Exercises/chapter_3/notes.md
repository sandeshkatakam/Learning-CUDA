## PMPP Chapter 3 Notes: Multidimensional grids and Data

## Multi Dimensional Grid Organization

* A `grid` is a three-dimensional (3D) array of blocks, and each block is a 3D array of threads
* Execution params of the kernel function call
    * First execution parameter spcifies the dimensions of the grid in no. of blocks
    * second execution parameter specifies the dimensions of each block in no. of threads
    * Each such parameter has the datatype `dim3` which is an integer vector type of three elements `x`, `y`, `z`. These three elements specify the sizes of three dimensions by setting the size of unused dimension to 1

* **Note**: In chapter 2, we just used one dimension x that's why we called `threadIdx.x`, `blockIdx.x`, `blockDim.x` because in that case we have computations involving 1D data (vectors). If we move futher to matrix multiplications of 3D Tensors then we might need all of dimensions
* For specific one dimensional case, we can modify our previous `vecAddKernel` code to accomodate more general grid and block dimensions

```cpp
dim3 dimGrid(32,1,1);
dim3 dimBlock(128,1,1);

vecAddKernel<<<dimGrid, dimBlock>>>(...);
```

**Note**: 
* Here since we are not use all the dimensions (our data is 1D) we can set the remaning to `y` and `z` to 1
* Also note that `dimBlock` and `dimGrid` are host code variables that are defined. These variables can have any allowed C variable names as long as the datatype is `dim3`

One more example using `ceil()` to specify the block numbers:  
```cpp
dim3 dimGrid(ceil(n/256.0),1,1)'
dim3 dimBlock(256,1,1);
vecAddKernel<<<dimGrid, dimBlock>>>(...);
```
For convenience, CUDA provides a special shortcut for calling a kernel with 1D grids and blocks:(one can use arithmetic expression to specify the config of 1D grids and blocks, it evaluates the expression for `x` and rest `y` and `z` are set to 1)
```cpp
vecAddKernel<<<ceil(n/256.0), 256>>>(...);
```

**Note:** It takes advantage of how C++ constructors and default parameters work. `dim3` parameter has default values set to 1. Now, when a single value is passed where a `dim3` is expected, that value will be passed to the first parameter of the constructor while the other expected parameters take the default values

### Some details about CUDA C built-in variables
* Range of `gridDim` built-in variables:
    * `gridDim.x` : $1$ to $2^{31} -1$
    * `gridDim.y`, `gridDim.z`: $1$ to $2^{16} -1$ (65,535)
* All Threads in a block share the same `blockIdx.x`, `blockIdx.y`, `blockIdx.z` values
* Range of `blockIdx` variables:
    * `blockIdx.x` : $0$ to `gridDim.x` - 1
    * `blockIdx.y` : $0$ to `gridDim.y` - 1
    * `blockIdx.z` : $0$ to `gridDim.z` - 1
* The number of threads in each dimension of block can be accessed by th e configuration parameters `x`, `y` and `z` fields of `blockDim`


**Fact:** The total size of block in Current CUDA systems is limited to `1024 threads`, these threads can be distributed in whatever way we like as long as the number of threads is less than or equal to 1024  
examples: For `blockDim` values of (512,1,1), (8,16,4) are all allowed, but (32,32,2) is not allowed
* A grid and a block do not need to have the same dimensionality (even tho block thread are restricted to 1024 we can spin off many blocks i.e. higher grid dimensions)

## Mapping Threads to Multidimensional Data