/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <thrust/system/cuda/detail/guarded_cuda_runtime_api.h>

#include <thrust/detail/config.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/detail/raw_reference_cast.h>
#include <thrust/detail/seq.h>
#include <thrust/system/cuda/config.h>
#include <thrust/system/cuda/detail/util.h>
#include <thrust/system/detail/bad_alloc.h>
#include <thrust/detail/malloc_and_free.h>

#ifdef THRUST_CACHING_DEVICE_MALLOC
#include <cub/util_allocator.cuh>
#endif

#include <cub/detail/target.cuh>

namespace thrust
{
namespace cuda_cub {

#ifdef THRUST_CACHING_DEVICE_MALLOC
#define __CUB_CACHING_MALLOC
#ifndef __CUDA_ARCH__
inline cub::CachingDeviceAllocator &get_allocator()
{
  static cub::CachingDeviceAllocator g_allocator(true);
  return g_allocator;
}
#endif
#endif


// note that malloc returns a raw pointer to avoid
// depending on the heavyweight thrust/system/cuda/memory.h header
template<typename DerivedPolicy>
__host__ __device__
void *malloc(execution_policy<DerivedPolicy> &, std::size_t n)
{
  void *result = 0;

  // need to repeat a lot of code here because we can't use #if inside of the
  // NV_IF_TARGET macro.
  // The device path is the same either way, but the host allocations differ.
#ifdef __CUB_CACHING_MALLOC
  NV_IF_TARGET(NV_IS_HOST, (
    cub::CachingDeviceAllocator &alloc = get_allocator();
    cudaError_t status = alloc.DeviceAllocate(&result, n);

    if (status != cudaSuccess)
    {
      cudaGetLastError(); // Clear global CUDA error state.
      throw thrust::system::detail::bad_alloc(thrust::cuda_category().message(status).c_str());
    }
  ), ( // NV_IS_DEVICE
    result = thrust::raw_pointer_cast(thrust::malloc(thrust::seq, n));
  ));
#else // not __CUB_CACHING_MALLOC
  NV_IF_TARGET(NV_IS_HOST, (
    cudaError_t status = cudaMalloc(&result, n);

    if (status != cudaSuccess)
    {
      cudaGetLastError(); // Clear global CUDA error state.
      throw thrust::system::detail::bad_alloc(thrust::cuda_category().message(status).c_str());
    }
  ), ( // NV_IS_DEVICE
    result = thrust::raw_pointer_cast(thrust::malloc(thrust::seq, n));
  ));
#endif

  return result;
} // end malloc()


template<typename DerivedPolicy, typename Pointer>
__host__ __device__
void free(execution_policy<DerivedPolicy> &, Pointer ptr)
{
  // need to repeat a lot of code here because we can't use #if inside of the
  // NV_IF_TARGET macro.
  // The device path is the same either way, but the host deallocations differ.
#ifdef __CUB_CACHING_MALLOC
  NV_IF_TARGET(NV_IS_HOST, (
    cub::CachingDeviceAllocator &alloc = get_allocator();
    cudaError_t status = alloc.DeviceFree(thrust::raw_pointer_cast(ptr));
    cuda_cub::throw_on_error(status, "device free failed");
  ), ( // NV_IS_DEVICE
    thrust::free(thrust::seq, ptr);
  ));
#else // not __CUB_CACHING_MALLOC
  NV_IF_TARGET(NV_IS_HOST, (
    cudaError_t status = cudaFree(thrust::raw_pointer_cast(ptr));
    cuda_cub::throw_on_error(status, "device free failed");
  ), ( // NV_IS_DEVICE
    thrust::free(thrust::seq, ptr);
  ));
#endif
} // end free()

}    // namespace cuda_cub
} // end namespace thrust
