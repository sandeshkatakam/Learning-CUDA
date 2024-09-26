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

```mermaid
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
    cudaMalloc((void**)&A_d, size))=;
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

**Note::**
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