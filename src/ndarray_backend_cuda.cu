#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>

namespace needle {
namespace cuda {

#define BASE_THREAD_NUM 256

#define TILE 4
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);
typedef ssize_t ptrdiff_t;

struct CudaArray {
  CudaArray(const size_t size) {
    cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    this->size = size;
  }
  ~CudaArray() { cudaFree(ptr); }
  size_t ptr_as_int() { return (size_t)ptr; }
  
  scalar_t* ptr;
  size_t size;
};

struct CudaDims {
  dim3 block, grid;
};

CudaDims CudaOneDim(size_t size) {
  /**
   * Utility function to get cuda dimensions for 1D call
   * hbsun：
   *    dim.block： how many threads in a block
   *    dim.grid： how many blocks in a grid
   */
  CudaDims dim;
  size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
  dim.block = dim3(BASE_THREAD_NUM, 1, 1);
  dim.grid = dim3(num_blocks, 1, 1);
  return dim;
}

#define MAX_VEC_SIZE 8
struct CudaVec {
  uint32_t size;
  int32_t data[MAX_VEC_SIZE];
};

CudaVec VecToCuda(const std::vector<int32_t>& x) {
  CudaVec shape;
  if (x.size() > MAX_VEC_SIZE) throw std::runtime_error("Exceeded CUDA supported max dimesions");
  shape.size = x.size();
  for (size_t i = 0; i < x.size(); i++) {
    shape.data[i] = x[i];
  }
  return shape;
}

////////////////////////////////////////////////////////////////////////////////
// Fill call
////////////////////////////////////////////////////////////////////////////////

__global__ void FillKernel(scalar_t* out, scalar_t val, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = val;
}

void Fill(CudaArray* out, scalar_t val) {
  CudaDims dim = CudaOneDim(out->size);
  FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Compact and setitem cals
////////////////////////////////////////////////////////////////////////////////

// Untility function to convert contiguous index i to memory location from strides




__global__ void CompactKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  /**
   * The CUDA kernel for the compact opeation.  This should effectively map a single entry in the 
   * non-compact input a, to the corresponding item (at location gid) in the compact array out.
   * 
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   size: size of out array
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
   *   strides: vector of strides of out array
   *   offset: offset of out array
   */
    ssize_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < size)
    {
        size_t idx = 0;
        size_t i = gid;
        for (int j = shape.size - 1; j >= 0; j--)
        {
            idx += (i % shape.data[j]) * strides.data[j];
            i /= shape.data[j];
        }
        out[gid] = a[idx + offset];
    }
}

void Compact(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory.  Unlike the C++ version, in CUDA this will primarily call the 
   * relevant CUDA kernel.  In this case, we illustrate how you should set this up (i.e., we give 
   * you the code for this fuction, and also the prototype for the CompactKernel() function).  For
   * the functions after this, however, you'll need to define these kernels as you see fit to 
   * execute the underlying function.
   * 
   * Args:
   *   a: non-compact represntation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   */

  // Nothing needs to be added here
  CudaDims dim = CudaOneDim(out->size);
  CompactKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
}


__global__ void EwiseSetitemKernel(scalar_t* a, size_t a_size, scalar_t* out, size_t size, CudaVec shape,
                                   CudaVec strides, size_t offset) {
    ssize_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < a_size)
    {
        size_t idx = 0;
        size_t i = gid;
        for (int j = shape.size - 1; j >= 0; j--)
        {
            idx += (i % shape.data[j]) * strides.data[j];
            i /= shape.data[j];
        }
        out[idx + offset] = a[gid];
    }
}


void EwiseSetitem(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array using CUDA.  Yyou will most likely want to implement a
   * EwiseSetitemKernel() function, similar to those above, that will do the actual work.
   * 
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
   CudaDims dim = CudaOneDim(out->size);
   EwiseSetitemKernel<<<dim.grid, dim.block>>>(a.ptr, a.size, out->ptr, out->size, VecToCuda(shape), VecToCuda(strides), offset);
}


__global__ void ScalarSetitemKernel(size_t size, scalar_t val, scalar_t* out, CudaVec shape,
                                    CudaVec strides, size_t offset) {
    ssize_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < size)
    {
        size_t idx = 0;
        size_t i = gid;
        for (int j = shape.size - 1; j >= 0; j--)
        {
            idx += (i % shape.data[j]) * strides.data[j];
            i /= shape.data[j];
        }
        out[idx + offset] = val;
    }
}


void ScalarSetitem(size_t size, scalar_t val, CudaArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   * 
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the 
   *         product of items in shape, but covenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */
    CudaDims dim = CudaOneDim(size);
    ScalarSetitemKernel<<<dim.grid, dim.block>>>(size, val, out->ptr, VecToCuda(shape),
                                          VecToCuda(strides), offset);
}

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////


//__global__ void EwiseAddKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
//  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
//  if (gid < size) out[gid] = a[gid] + b[gid];
//}
//
//void EwiseAdd(const CudaArray& a, const CudaArray& b, CudaArray* out) {
//  /**
//   * Add together two CUDA array
//   */
//  CudaDims dim = CudaOneDim(out->size);
//  EwiseAddKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
//}
//
//__global__ void ScalarAddKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
//  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
//  if (gid < size) out[gid] = a[gid] + val;
//}
//
//void ScalarAdd(const CudaArray& a, scalar_t val, CudaArray* out) {
//  /**
//   * Add together a CUDA array and a scalar value.
//   */
//  CudaDims dim = CudaOneDim(out->size);
//  ScalarAddKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
//}

/**
 * In the code the follows, use the above template to create analogous elementise
 * and and scalar operators for the following functions.  See the numpy backend for
 * examples of how they should work.
 *   - EwiseMul, ScalarMul
 *   - EwiseDiv, ScalarDiv
 *   - ScalarPower
 *   - EwiseMaximum, ScalarMaximum
 *   - EwiseEq, ScalarEq
 *   - EwiseGe, ScalarGe
 *   - EwiseLog
 *   - EwiseExp
 *   - EwiseTanh
 *
 * If you implement all these naively, there will be a lot of repeated code, so
 * you are welcome (but not required), to use macros or templates to define these
 * functions (however you want to do so, as long as the functions match the proper)
 * signatures above.
 */

template <typename Op>
__global__ void EwiseOpKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = Op::apply(a[gid], b[gid]);
}

template <typename Op>
void EwiseOp(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseOpKernel<Op><<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

template <typename Op>
__global__ void ScalarOpKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = Op::apply(a[gid], val);
}

template <typename Op>
void ScalarOp(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarOpKernel<Op><<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

struct Add {
  static __device__ scalar_t apply(scalar_t a, scalar_t b) { return a + b; }
};

struct Mul {
  static __device__ scalar_t apply(scalar_t a, scalar_t b) { return a * b; }
};

struct Div {
  static __device__ scalar_t apply(scalar_t a, scalar_t b) { return a / b; }
};

struct Power {
  static __device__ scalar_t apply(scalar_t a, scalar_t b) { return pow(a, b); }
};

struct Maximum {
  static __device__ scalar_t apply(scalar_t a, scalar_t b) { return fmax(a, b); }
};

struct Eq {
  static __device__ scalar_t apply(scalar_t a, scalar_t b) { return a == b; }
};

struct Ge {
  static __device__ scalar_t apply(scalar_t a, scalar_t b) { return a >= b; }
};

struct Log {
  static __device__ scalar_t apply(scalar_t a, scalar_t b) { return log(a); }
};

struct Exp {
  static __device__ scalar_t apply(scalar_t a, scalar_t b) { return exp(a); }
};

struct Tanh {
  static __device__ scalar_t apply(scalar_t a, scalar_t b) { return tanh(a); }
};

void EwiseAdd(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  EwiseOp<Add>(a, b, out);
}

void ScalarAdd(const CudaArray& a, scalar_t val, CudaArray* out) {
  ScalarOp<Add>(a, val, out);
}

void EwiseMul(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  EwiseOp<Mul>(a, b, out);
}

void ScalarMul(const CudaArray& a, scalar_t val, CudaArray* out) {
  ScalarOp<Mul>(a, val, out);
}

void EwiseDiv(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  EwiseOp<Div>(a, b, out);
}

void ScalarDiv(const CudaArray& a, scalar_t val, CudaArray* out) {
  ScalarOp<Div>(a, val, out);
}

void ScalarPower(const CudaArray& a, scalar_t val, CudaArray* out) {
  ScalarOp<Power>(a, val, out);
}

void EwiseMaximum(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  EwiseOp<Maximum>(a, b, out);
}

void ScalarMaximum(const CudaArray& a, scalar_t val, CudaArray* out) {
  ScalarOp<Maximum>(a, val, out);
}

void EwiseEq(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  EwiseOp<Eq>(a, b, out);
}

void ScalarEq(const CudaArray& a, scalar_t val, CudaArray* out) {
  ScalarOp<Eq>(a, val, out);
}

void EwiseGe(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  EwiseOp<Ge>(a, b, out);
}

void ScalarGe(const CudaArray& a, scalar_t val, CudaArray* out) {
  ScalarOp<Ge>(a, val, out);
}

void EwiseLog(const CudaArray& a, CudaArray* out) {
  EwiseOp<Log>(a, a, out);
}

void EwiseExp(const CudaArray& a, CudaArray* out) {
  EwiseOp<Exp>(a, a, out);
}

void EwiseTanh(const CudaArray& a, CudaArray* out) {
  EwiseOp<Tanh>(a, a, out);
}



////////////////////////////////////////////////////////////////////////////////
// Matmul
////////////////////////////////////////////////////////////////////////////////

__global__ void MatmulKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, uint32_t M, uint32_t N, uint32_t P);


void Matmul(const CudaArray& a, const CudaArray& b, CudaArray* out, uint32_t M, uint32_t N,
            uint32_t P) {
  /**
   * Multiply two (compact) matrices into an output (also comapct) matrix.  You will want to look
   * at the lecture and notes on GPU-based linear algebra to see how to do this.  Since ultimately
   * mugrade is just evaluating correctness, you _can_ implement a version that simply parallelizes
   * over (i,j) entries in the output array.  However, to really get the full benefit of this
   * problem, we would encourage you to use cooperative fetching, shared memory register tiling, 
   * and other ideas covered in the class notes.  Note that unlike the tiled matmul function in
   * the CPU backend, here you should implement a single function that works across all size
   * matrices, whether or not they are a multiple of a tile size.  As with previous CUDA
   * implementations, this function here will largely just set up the kernel call, and you should
   * implement the logic in a separate MatmulKernel() call.
   * 
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: comapct 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   M: rows of a / out
   *   N: columns of a / rows of b
   *   P: columns of b / out
   */
    size_t size = M * P;
    CudaDims dim = CudaOneDim(size);
    MatmulKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, M, N, P);
}

__global__ void MatmulKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, uint32_t M, uint32_t N,
            uint32_t P) {
    /**
     * Kernel to multiply two (compact) matrices into an output (also compact) matrix.  You will want to look
     * at the lecture and notes on GPU-based linear algebra to see how to do this.  Since ultimately
     * mugrade is just evaluating correctness, you _can_ implement a version that simply parallelizes
     * over (i,j) entries in the output array.  However, to really get the full benefit of this
     * problem, we would encourage you to use cooperative fetching, shared memory register tiling,
     * and other ideas covered in the class notes.  Note that unlike the tiled matmul function in
     * the CPU backend, here you should implement a single function that works across all size
     * matrices, whether or not they are a multiple of a tile size.  As with previous CUDA
     * implementations, this function here will largely just set up the kernel call, and you should
     * implement the logic in a separate MatmulKernel() call.
     *
     *
     * Args:
     *   a: compact 2D array of size m x n
     *   b: compact 2D array of size n x p
     *   out: compact 2D array of size m x p to write the output to
     *   M: rows of a / out
     *   N: columns of a / rows of b
     *   P: columns of b / out
     */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = idx / P;
    int j = idx % P;
//    printf("M: %d, N: %d, P: %d, i: %d, j: %d, idx: %d\n", M, N, P, i, j, idx);
    if (i < M && j < P) {
        scalar_t sum = 0;
        for (int k = 0; k < N; k++) {
            sum += a[i * N + k] * b[k * P + j];
        }
        out[idx] = sum;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Max and sum reductions
////////////////////////////////////////////////////////////////////////////////

 // define a struct for the max reduction
    struct Max {
    static __device__ scalar_t apply(scalar_t a, scalar_t b) { return fmax(a, b); }
    static __device__ scalar_t identity() { return -INFINITY; }
    };

    // define a struct for the sum reduction
    struct Sum {
    static __device__ scalar_t apply(scalar_t a, scalar_t b) { return a + b; }
    static __device__ scalar_t identity() { return 0; }
    };


template <typename Op>
__global__ void ReduceKernel(scalar_t* a, size_t a_size, scalar_t* out, size_t reduce_size) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    scalar_t val = Op::identity();
    size_t base = gid * reduce_size;
    for (size_t i = 0; i < reduce_size; i++) {
        if (base + i < a_size) {
            val = Op::apply(val, a[base + i]);
        }
    }

    if (gid < a_size / reduce_size) {
        out[gid] = val;
    }
}


// Optimized version of the above ReduceKernel which will execute in one thread when set axis=None.
// 1. use shared memory to reduce the number of global memory accesses
// 2. the size of output array is a_size / reduce_size
// 2. use a hierarchical mechanism that first aggregated across all threads within one block, then had a secondary
// function that aggregated across these blocks.
template <typename Op>
__global__ void ReduceKernelOptimized(scalar_t* a, size_t a_size, scalar_t* out, size_t reduce_size) {
    // write to out array with the size of a_size / reduce_size
    // use a hierarchical mechanism that first aggregated across all threads within one block, then had a secondary
    // function that aggregated across these blocks.
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    scalar_t val = Op::identity();
    size_t base = gid * reduce_size;
    for (size_t i = 0; i < reduce_size; i++) {
        if (base + i < a_size) {
            val = Op::apply(val, a[base + i]);
        }
    }
}


template <typename Op>
void Reduce(const CudaArray& a, CudaArray* out, size_t reduce_size)
{
    CudaDims dim = CudaOneDim(a.size);
    ReduceKernel<Op><<<dim.grid, dim.block>>>(a.ptr, a.size, out->ptr, reduce_size);
}

void ReduceMax(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.  Even though it is inefficient,
   * for simplicity you can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
    Reduce<Max>(a, out, reduce_size);
}




void ReduceSum(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking summation over `reduce_size` contiguous blocks.  Again, for simplicity you 
   * can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
    Reduce<Sum>(a, out, reduce_size);
}

}  // namespace cuda
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cuda, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cuda;

  m.attr("__device_name__") = "cuda";
  m.attr("__tile_size__") = TILE;

  py::class_<CudaArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &CudaArray::size)
      .def("ptr", &CudaArray::ptr_as_int);

  // return numpy array, copying from CPU
  m.def("to_numpy", [](const CudaArray& a, std::vector<size_t> shape, std::vector<size_t> strides,
                       size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });

    // copy memory to host
    scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);
    if (host_ptr == 0) throw std::bad_alloc();
    cudaError_t err = cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    // return numpy array
    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer);
  });

  // copy numpy array to GPU
  m.def("from_numpy", [](py::array_t<scalar_t> a, CudaArray* out) {
    cudaError_t err =
        cudaMemcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);

  m.def("matmul", Matmul);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}