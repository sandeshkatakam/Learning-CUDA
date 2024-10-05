## PMPP Chapter 2 Notes: Heterogenous Data Parallel Computing



### Data parallelism

* Independent evaluation of different pieces of data is the basis of "data parallelism"

| Data Parallelism | Task Parallelism | 
| --- | --- |  
| Data Decomposition | Task Decomposition |
| Example: Vector Addition of two vectors where the elements of vectors are summed elementwise by each thread doing the same kind of computation | Example: A simple application need to do a vector addition and a matix-vector multiplication, each of these can be a task |  
| Main Source of Scalability for Parallel Programs | Only useful when we have too many tasks |   
| Exists if the same kind of computation task should be performed on all elements of the data that is divided | Exists if the two tasks can be performed independently (no communication of data is required between two tasks) |  


## CUDA C Program Structure

* It extends the popular ANSI C with minimal new syntax and library functions to let programmers target heterogenous systems (CPU + GPU)
* The Structure of CUDA C program reflects the co existence of host (CPU) and Device (GPU)
* Each CUDA C code contains both Host (CPU) code and one or more Device (GPU) code
* The Device code is marked with special CUDA Keywords 
* The Device code includes functions, or kernels, whose code is executed in a data-parallel manner

### Workflow of CUDA Program
* Execution starts with host code(CPU Serial Code)
* A kernel function launches a number of threads on a device to execute the kernel
* All Threads launched by a kernel call are collectively called `grid`
* Threads are the basis of parallel computation
* When all the threads of `grid` completes then it moves to the next CPU Serial code (host code)
* The CPU and GPU code doesn't need to happen one after another, they can overlap too
* Many heterogenous computing applications manage overlapped CPU and GPU execution to take advantage of both CPUs and GPUs
* **Note**: In the context of data parallelism the number of threads launched are usually the number of elements in the data
* CUDA Programmers can assume that these threads take very few clock cycles to generate and schedule. This assumption contrasts with tradition CPU threads, which typically take thousands of clock cycles to generate and schedule 

```
CUDA Execution WorkFlow:  
    CPU Serial Code --> Device Parallel Kernel Code (KenelA<<<nBlk, nTid>>>) --> CPU Serial Code --> Device Parallel Kernel Code (KenelB<<<nBlk, nTid>>>)

```

#### Threads:
* A Thread is a simplified view of how a processor executes a sequential program (It's an abstraction for how computers execute computation)
* It consists of the code of the program, the point in the code that is being executed, and the values of its variables
* The Execution of thread is sequential (Not parallelizable any more beyond this abstraction) **Even in CUDA**
* If we want parallel execution, we can do this by launching a number of threads (Multithreading can be done on CPUs too, but number of threads are limited not as much as we can launch on GPUs,mostly because of it's latency oriented design) using thread libraries or special languages
* A CUDA Program initiates parallel execution by calling kernel functions, which causes the underlying runtime mechanisms to launch a grid of threads that process different parts of the data in parallel.

## CASE STUDY: A Vector Addition Kernel

* How a simple CPU Serial code vector addition is implemented (host code)
```cpp
// Compute Vector Sum C_h = A_h + B_h
void vecAdd(float* A_h, float* B_h, float* C_h, int n) {
    for (int i = 0; i < n; ++i){
        c_h[i] = a_h[i] + b_h[i];
    }
}

int main(){
    //Memory allocation for arrays A_h, B_h, C_h
    vecAdd(A,B,C, n);
}
```

**Note**: `_h` represents the data/variables in host(CPU) `_d` represents the data/variables in device (GPUs)  

### Pointers in C:
* The Function arguments A,B,C are pointers. In C A pointer can be used to access variables and data structures
    * `float V;` Declaration floating point variable 
    * `float* P;` Delcaration of Pointer variable P 
    * `P = &V` Assign Address of V to Pointer P, then P points to Vector V and `*P` becomes synonym for V

* An Array in C Program can be accessed through a pointer that points to its 0th element. For example, `P = &(A[0])` makes P point to the 0th element of Array A. In fact, the Array name A is in itself a pointer to its 0th element
* Passing an Array name A as the first argument to function call to `vecAdd` makes the function's first parameter A_h point to the 0th element of A

Now we move the for-loop part of the code to the device to execute parallel since we are executing the same sum computation over all the elements

```cpp

__global__ 
void VecAdd(float* A_h, float* B_h, float* C_h, int n){

    int size = n*sizeof(n);

    float *A_d, float *B_d, float* C_d;

    // Part 1: Allocate device memory for A, B and C
    cudaMalloc((void**)&A_d, size);
    cudaMalloc((void**)&B_d, size);
    cudaMalloc((void**)&C_d, size);

    // Copy A and B to device memory
    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(C_d, C_h, size, cudaMemcpyHostToDevice);

    // Part 2: Call kernel- to launch a grid of threads
    // to perform the actual vector addition
    vecAddKernel<<<ceil(n/256.0), 256.0>>>(A_d, B_d, C_d, n);
    // Another way of Invoking the CUDA Kernel with Grids and Block size Specified
    // Kernel invocation with 256 threads
	// dim3 dimGrid(ceil(n / 256.0),1,1);
	// dim3 dimBlock((256.0),1,1);

    // Part 3: Copy C from the device memory
    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);
    //Free Device Vectors
    
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
```
**IMP**: We refer to the above type of host code as `stub` for calling a kernel, which involves allocating space and copying data and invoking kernel function etc.

**Note:**
* Often the copying of memory back and forth creates a bottleneck and we don't ge the desired performance, and we usually don't copy the datastructures/variables that often from device to host and host to device. We instead keep large and important data structures on the device and simply invoke device functions on them from the host code

### Device Memory Specific CUDA API Functions

* `cudaMalloc()`:
    * Allocates object in the device global memory
    * Two params:
        * **Address of a pointer** to the allocated object
        * This should be casted to `(Void**)` because the function expects a generic pointer
        * The Memory allocation function is a generic function that is not restricted to any particular type of objects. This allows `cudaMalloc()` to write the address of the object to a pointer regardless of datatype
        * **size** of allocated object in terms of bytes (it's consistent with C malloc function in bytes)

* `cudaFree()`:
    * Frees object from device global memory
    * **Pointer** to freed object
* `cudaMemcpy`:
    * Memory Data Transfer
    * Requires four params:
        * Pointer to destination: `A_d` if device or `A_h` if host
        * Pointer to source: `A_d` if device or `A_h` if host
        * Number of bytes copied: `size`
        * Type/Direction of Transfer: `cudaMemcpyDeviceToHost` or `cudaMemcpyHostToDevice`
* **Note**: 
    * CUDA C also has more  advanced library functions for allocating space in the host memory.
    * The fact that `cudaMalloc()` returns a generic object makes the use of dynamically allocated multi-dimensional arrays more complex.
    * `cudaMalloc()` has a different format from the C malloc function. 

### Difference between C Malloc and cudaMalloc function:

#### C malloc function:

* It allocates memory on the host (CPU).
* It takes one parameter: the size of memory to allocate (in bytes).
* It returns a pointer to the allocated memory.
* Usage: void* ptr = malloc(size);


#### CUDA cudaMalloc() function:

* It allocates memory on the device (GPU).
* It takes two parameters:
    * A pointer to a pointer (the address where the allocated memory pointer will be stored).
    * The size of memory to allocate (in bytes).


* It returns an error code to indicate success or failure.
* Usage: cudaError_t error = cudaMalloc((void**)&dev_ptr, size);



**The key differences are:**

* Return value:

    * malloc returns the allocated memory address directly.
    * cudaMalloc() returns an error code and writes the allocated memory address to the provided pointer.


* Error handling:

    * With malloc, you check if the returned pointer is NULL to detect errors.
    * With cudaMalloc(), you check the returned error code for any issues.


* Memory location:

    * malloc allocates on the host (CPU) memory.
    * cudaMalloc() allocates on the device (GPU) memory.


**Note**:  
* The two-parameter format of cudaMalloc() allows it to be consistent with other CUDA API functions, which typically return error codes for uniform error handling across the CUDA ecosystem.

* The addresses in A_d, B_d and C_d point to locations in the device global memory. These addresses should not be dereferenced in the host code. They should be used in calling API functions an dkernel functions.
* Dereferencing a device global memory pointer in host code can cause exceptions or other types of runtime errors.

### Error Checking and Error Handling in CUDA
* CUDA API Functions return flags that indicate whether an erros has occured when they served the request
* In practice we surround the `cudaMalloc()` call with code that test for error condition and print out error messages.
* A simple version of such code:
```cpp
cudaError_t err = cudaMalloc((void**)&A_d, size);
if(error != cudaSuccess)  {
    printf("%s in %s at line %d \n",         cudaGetErrorString(err),
    __FILE__,__LINE__)'
    exit(EXIT_FAILURE);
}
```
* This way if system is out of device memory,the user will be informed about the situation. This can save many hours of debugging time. 
* A C Macro that can be reused for this purpose:
```cpp
#define CHECK_ERROR(call) { \
cudaError_t err = call; \
if (err != cudaSuccess) { \
printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
exit(err); \
} \
}

```

### Kernel Functions and Threading
* Since all the threads execute the same code, CUDA C programming is an instance of the well-known single-program multiple-data (SPMD) style
* When the program's host code calls a kernel, the CUDA runtime system launches a grid of threads that are organized into a two-level heirarchy
* Hierarchy of CUDA Kernel Grid:
    * `grid` : An array of threaded blocks (All blocks of grid are of same size)
    * `blocks`: Contains threads (each block contain upto `1024 threads`)
    * `threads`: abstraction for unit of computation

#### CUDA Bult-in Variables
* The values of these variables are often pre-initialized by the runtime system and are typically read-only in the program(we should refrain from redefining these variables)

* The total no. of threads in each thread  block is specified by the host code when a kernel is called.
* The same kernel can be called with different numbers of threads at different parts of the host code

* **Built-in variables**
    * `blockDim`: The number of threads in a block
        * It's variable struct with three unsigned integer (x,y and z)
        * For 1-D data only x field is used, for 2-D data both x and y field are used, for 3-D data all x,y and z fields are used
        * The choice of the dimensionality for organizing threads usually reflects the dimensionality of data
        * In general it is recommended that the number of threads in each dimensions of a thread block be a multiple of 32 for hardware efficiency reasons
    * `threadIdx`: Gives each thread a unique coordinate within a block. For e.g. 1-D Data uses only `threadIdx.x`
    * `blockIdx`: Gives all threads in a block a common co-ordinate


**IMPORTANT NOTE**: 
* To calculate the global index i for a unique specific thread in a grid we use `i = blockDim.x * blockIdx.x + threadIdx.x` for 1-D organization of threads
* The above basically calculates a unique thread index for every thread in a grid. for example, the 5th thread in the Block 1 of 256 thread count(note it starts with Block 0 indexing) will be calculated `i = 1 * 256 + 5 = 261(global thread index)`

> #### Reason for Multiples of 32 Threads
This recommendation is based on how GPUs, particularly NVIDIA GPUs, are designed and function at the hardware level.  
In NVIDIA GPUs, threads are executed in groups called **warps**. A warp consists of **32 threads** that execute in lockstep (i.e., they execute the same instruction at the same time).  
GPUs use a **Single Instruction, Multiple Thread (SIMT) architecture**. This means that all threads in a warp execute the same instruction, but on different data.  
When the number of threads in a block is a multiple of 32, it aligns perfectly with the warp size. This leads to several efficiency benefits:
* Full warp utilization: All threads in each warp are active, maximizing parallel execution.
* Reduced divergence: Threads within a warp are less likely to diverge in their execution paths.
* Memory coalescing: Memory accesses are more likely to be coalesced, improving memory bandwidth utilization.
* Scheduling efficiency: The GPU's thread scheduler can work more efficiently when dealing with full warps.
Resource allocation:
GPU resources (like shared memory and registers) are often allocated on a per-warp basis. Using multiples of 32 threads ensures optimal resource utilization.
Avoiding partial warps:
If the number of threads is not a multiple of 32, the last warp will be partially filled. This partial warp still consumes the resources of a full warp but doesn't fully utilize the GPU's processing capability.

#### CUDA C Extension to C language (Keywords)
* `__global__` :
    * keyword indicates that the function being declared is a CUDA C kernel function
    * Such a kernel function is executed on the device and can be called from the host.
    * In CUDA systems, that supports **dynamic parallelism** it can also be called from the device
* `__device__`:
    * Keyword indicates  that the function being declared is a CUDA device function
    * Such function executes on a CUDA device and can be called only from a kernel function or another device function
    * The device function is excuted by the device thread that calls it and does not result in any new device threads being launched
* `__host__`:
    * Keyword indicates that the function being declares is a CUDA host function
    * Such function ia simply a traditional C function that executes on the host and can be called only from antoer host function
    * By default all functions in CUDA program are host functions if they do not have any of the CUDA keywords in their declaration

**Imp Notes**: 
* Both `__host__` and `__device__` in a function declaration tells the compilation system to generate two versions of the object code for the same function. Many user library functions will likely fall into this category  
* The next notable extension to C is built-in variables `threadIdx`, `blockDim`, `blockIdx` . Different threads will see
different values in their `threadIdx.x`, `blockIdx.x`, `blockDim.x` variables. 
* `i` is the automatic variable. In a CUDA Kernel function,
automatic variables are private to each thread. If the grid is launched with 10,000 threads, there will be 10,000 versions of i.
* There is an `if(i<n)` statement in addVecKernel. This is because not all vector lengths can be expressed as multiples of block size. 
For example, assume vector length is 100, the smallest efficient block dimension is 32, one would need to lauch four thread blocks to process all 100 vector elements.
But four blocks contain 128 threads, inorder to disable the last 28 threads in thread block 3 from doint work not expected by the original program.


* **A Table of CUDA C Keywords for Function declaration:**

| **Qualifier Keyword** | **Callable From** | **Executed on** | **Executed by** |
| --- | --- | --- | --- |  
| `__host__`(default) | Host | Host | Caller host thread |
| `__global__` | Host (or Device) | Device | New grid of device threads |
| `__device__` | Device | Device | Caller Device thread |

### Calling Kernel Functions:

```cpp
int vecAdd(float* A, float* B, float* C, int n) {
    // A_d, B_d, C_d allocations and copies are done here

    ...
    // Launch ceil (n/256) blocks of 256 threads each
    vecAddKernel<<<ceil(n/256.0), 256>>>(A_d, B_d, C_d, n);

}
```
* When host code calls a kernel, it sets the grid and thread blok dimensions via **execution configuration parameters**
* The config params are given between the `<<<` and `>>>` before the traditional C function arguments.
    * First config param gives the number of blocks in the grid
    * Second one specifies the number of threads in each block

* Note that all the thread locks operate on different parts of the vectors, and executed in any arbitrary order. The programmer must not make any assumptions regarding execution order
* A small GPU with a small amount of execution resources may execute only one or two of these blocks in parallel. A larger GPU may excute 64 or 128 blocks in parallel
* This gives CUDA kernels scalability in execution speed with hardware i.e., the same code runs at lower speed on small GPUs and higher speed on larger GPUs

### CUDA Compilation Process
* **Compiler**: NVCC Compiler (NVIDIA C Compiler)
* The NVCC Compiler processes a CUDA C program, using the CUDA keywords to separate the host code and device code.
* The host code is straight ANSI C Code, which is compiled with host's standard C/C++ compilers
* The device code which is marked with CUDA keywords that designate CUDA kernels and their associated helper functions and datastructures, is compile dby NVCC into virutal binary files called **PTX**(equivalent to assembly code in CPU)
* These **PTX** files are further compiled by a runtime component of NVCC into real object files and executed on a CUDA GPU device


