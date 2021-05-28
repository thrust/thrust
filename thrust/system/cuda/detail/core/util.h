/******************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/
#pragma once

#include <cuda_occupancy.h>
#include <thrust/detail/config.h>
#include <thrust/system/cuda/config.h>
#include <thrust/type_traits/is_contiguous_iterator.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/system/cuda/detail/util.h>

#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_scan.cuh>

#include <cub/detail/ptx_dispatch.cuh>
#include <cub/detail/target.cuh>

namespace thrust
{

namespace cuda_cub {
namespace core {

    // retrieve temp storage size from a PtxPlan.
    // -------------------------------------------------------------------------
    // metafunction introspects a PtxPlan, and if it finds TempStorage type
    // it will return its size.

    __THRUST_DEFINE_HAS_NESTED_TYPE(has_temp_storage, TempStorage)

    template <class PtxPlan, class U>
    struct temp_storage_size_impl;

    template <class PtxPlan>
    struct temp_storage_size_impl<PtxPlan, thrust::detail::false_type>
    {
      static constexpr std::size_t value = 0;
    };

    template <class PtxPlan>
    struct temp_storage_size_impl<PtxPlan, thrust::detail::true_type>
    {
      static constexpr std::size_t value =
        sizeof(typename PtxPlan::TempStorage);
    };

    template <class PtxPlan>
    struct temp_storage_size
        : temp_storage_size_impl<PtxPlan,
                                 typename has_temp_storage<PtxPlan>::type>
    {
    };

    // AgentPlan structure and helpers
    // --------------------------------

    // AgentPlan is a runtime version of the constexpr Agent::PtxPlan. It is
    // used to pass generic Agent parameters around without instantiating
    // templates for each tuned PtxPlan.
    //
    // For any Agent with an `Agent::Tunings` member, use:
    // AgentPlan agent_plan = AgentPlanFromTunings<Agent>::get();
    // to obtain an AgentPlan that best matches the current device.
    struct AgentPlan
    {
      int block_threads;
      int items_per_thread;
      int items_per_tile;
      int shared_memory_size;
      int grid_size;

      CUB_RUNTIME_FUNCTION
      AgentPlan() {}

      CUB_RUNTIME_FUNCTION
      AgentPlan(int block_threads_,
                int items_per_thread_,
                int shared_memory_size_,
                int grid_size_ = 0)
          : block_threads(block_threads_),
            items_per_thread(items_per_thread_),
            items_per_tile(items_per_thread * block_threads),
            shared_memory_size(shared_memory_size_),
            grid_size(grid_size_)
      {
      }

      CUB_RUNTIME_FUNCTION
      AgentPlan(AgentPlan const& plan)
          : block_threads(plan.block_threads),
            items_per_thread(plan.items_per_thread),
            items_per_tile(plan.items_per_tile),
            shared_memory_size(plan.shared_memory_size),
            grid_size(plan.grid_size) {}

      template <class PtxPlan>
      CUB_RUNTIME_FUNCTION
      AgentPlan(PtxPlan,
                typename thrust::detail::disable_if_convertible<
                    PtxPlan,
                    AgentPlan>::type* = NULL)
          : block_threads(PtxPlan::BLOCK_THREADS),
            items_per_thread(PtxPlan::ITEMS_PER_THREAD),
            items_per_tile(PtxPlan::ITEMS_PER_TILE),
            shared_memory_size(temp_storage_size<PtxPlan>::value),
            grid_size(0)
      {
      }
    };    // struct AgentPlan

    // Create an AgentPlan for the current target from Agent::PtxPlan<Tuning>
    // and the Agent::Tunings list. The `AgentPlanType` may be overridden but
    // defaults to `thrust::cuda_cub::core::AgentPlan`.
    //
    // Usage:
    // AgentPlan agent_plan = AgentPlanFromTunings<Agent>::get();
    template <typename Agent,
              typename AgentPlanType = AgentPlan>
    struct AgentPlanFromTunings
    {
      CUB_RUNTIME_FUNCTION
      static AgentPlanType get()
      {
        using tunings_t = typename Agent::Tunings;
        constexpr auto exec_space = cub::detail::runtime_exec_space;
        using dispatcher_t = cub::detail::ptx_dispatch<tunings_t, exec_space>;

        AgentPlanFromTunings functor{};
        dispatcher_t::exec(functor);
        return functor.plan;
      }

      // Used by ptx_dispatch:
      template <typename Tuning>
      CUB_RUNTIME_FUNCTION
      void operator()(cub::detail::type_wrapper<Tuning>)
      {
        using PtxPlan = typename Agent::template PtxPlan<Tuning>;
        this->plan    = AgentPlanType{PtxPlan{}};
      }

    private:
      AgentPlanType plan{};
    };

  /////////////////////////
  /////////////////////////
  /////////////////////////

  CUB_RUNTIME_FUNCTION
  inline int get_sm_count()
  {
    int dev_id;
    cuda_cub::throw_on_error(cudaGetDevice(&dev_id),
                             "get_sm_count :"
                             "failed to cudaGetDevice");

    cudaError_t status;
    int         i32value;
    status = cudaDeviceGetAttribute(&i32value,
                                    cudaDevAttrMultiProcessorCount,
                                    dev_id);
    cuda_cub::throw_on_error(status,
                             "get_sm_count:"
                             "failed to sm_count");
    return i32value;
  }

  CUB_RUNTIME_FUNCTION
  inline std::size_t get_max_shared_memory_per_block()
  {
    int dev_id;
    cuda_cub::throw_on_error(cudaGetDevice(&dev_id),
                             "get_max_shared_memory_per_block :"
                             "failed to cudaGetDevice");

    cudaError_t status;
    int         i32value;
    status = cudaDeviceGetAttribute(&i32value,
                                    cudaDevAttrMaxSharedMemoryPerBlock,
                                    dev_id);
    cuda_cub::throw_on_error(status,
                             "get_max_shared_memory_per_block :"
                             "failed to get max shared memory per block");

    return static_cast<std::size_t>(i32value);
  }

  CUB_RUNTIME_FUNCTION
  inline std::size_t vshmem_size(std::size_t shmem_per_block,
                                 std::size_t num_blocks)
  {
    std::size_t max_shmem_per_block = core::get_max_shared_memory_per_block();
    if (shmem_per_block > max_shmem_per_block)
      return shmem_per_block*num_blocks;
    else
      return 0;
  }

  // LoadIterator
  // ------------
  // if trivial iterator is passed, wrap loads into LDG
  //
  template <class PtxPlan, class It>
  struct LoadIterator
  {
    typedef typename iterator_traits<It>::value_type      value_type;
    typedef typename iterator_traits<It>::difference_type size_type;

    typedef typename thrust::detail::conditional<
        is_contiguous_iterator<It>::value,
        cub::CacheModifiedInputIterator<PtxPlan::LOAD_MODIFIER,
                                        value_type,
                                        size_type>,
        It>::type type;
  };    // struct Iterator

  template <class PtxPlan, class It>
  typename LoadIterator<PtxPlan, It>::type __device__ __forceinline__
  make_load_iterator_impl(It it, thrust::detail::true_type /* is_trivial */)
  {
    return raw_pointer_cast(&*it);
  }

  template <class PtxPlan, class It>
  typename LoadIterator<PtxPlan, It>::type __device__ __forceinline__
  make_load_iterator_impl(It it, thrust::detail::false_type /* is_trivial */)
  {
    return it;
  }

  template <class PtxPlan, class It>
  typename LoadIterator<PtxPlan, It>::type __device__ __forceinline__
  make_load_iterator(PtxPlan const&, It it)
  {
    return make_load_iterator_impl<PtxPlan>(
        it, typename is_contiguous_iterator<It>::type());
  }

  // BlockLoad
  // -----------
  // a helper metaprogram that returns type of a block loader
  template <class PtxPlan,
            class It,
            class T    = typename iterator_traits<It>::value_type>
  struct BlockLoad
  {
    typedef cub::BlockLoad<T,
                           PtxPlan::BLOCK_THREADS,
                           PtxPlan::ITEMS_PER_THREAD,
                           PtxPlan::LOAD_ALGORITHM,
                           1,
                           1>
        type;
  };

  // BlockStore
  // -----------
  // a helper metaprogram that returns type of a block loader
  template <class PtxPlan,
            class It,
            class T = typename iterator_traits<It>::value_type>
  struct BlockStore
  {
    typedef cub::BlockStore<T,
                            PtxPlan::BLOCK_THREADS,
                            PtxPlan::ITEMS_PER_THREAD,
                            PtxPlan::STORE_ALGORITHM,
                            1,
                            1>
        type;
  };

  // cuda_optional
  // --------------
  // used for function that return cudaError_t along with the result
  //
  template <class T>
  class cuda_optional
  {
    cudaError_t status_{};
    T           value_{};

  public:
    __host__ __device__
    cuda_optional() : status_(cudaSuccess) {}

    __host__ __device__
    cuda_optional(T v, cudaError_t status = cudaSuccess) : status_(status), value_(v) {}

    bool __host__ __device__
    isValid() const { return cudaSuccess == status_; }

    cudaError_t __host__ __device__
    status() const { return status_; }

    __host__ __device__ T const &
    value() const { return value_; }

    __host__ __device__ operator T const &() const { return value_; }
  };

  CUB_RUNTIME_FUNCTION
  inline cuda_optional<int> get_ptx_version()
  {
    int ptx_version = 0;
    cudaError_t status = cub::PtxVersion(ptx_version);
    return cuda_optional<int>(ptx_version, status);
  }

  __device__
  inline void sync_threadblock()
  {
    cub::CTA_SYNC();
  }

#define CUDA_CUB_RET_IF_FAIL(e) \
  {                             \
    auto const error = (e);     \
    if (cub::Debug(error, __FILE__, __LINE__)) return error; \
  }

  // uninitialized_array
  // --------------
  // allocates uninitialized data on stack
  template<class T, size_t N>
  struct uninitialized_array
  {
    typedef T value_type;
    typedef T ref[N];
    enum {SIZE = N};
    private:
      char data_[N * sizeof(T)];

    public:
      __host__ __device__ T* data() { return data_; }
      __host__ __device__ const T* data() const { return data_; }
      __host__ __device__ T& operator[](unsigned int idx) { return ((T*)data_)[idx]; }
      __host__ __device__ T const& operator[](unsigned int idx) const { return ((T*)data_)[idx]; }
      __host__ __device__ T& operator[](int idx) { return ((T*)data_)[idx]; }
      __host__ __device__ T const& operator[](int idx) const { return ((T*)data_)[idx]; }
      __host__ __device__ unsigned int size() const { return N; }
      __host__ __device__ operator ref&() { return *reinterpret_cast<ref*>(data_); }
      __host__ __device__ ref& get_ref() { return (ref&)*this; }
  };

  __host__ __device__ __forceinline__
  std::size_t align_to(std::size_t n, std::size_t align)
  {
    return ((n+align-1)/align) * align;
  }

  // FIXME Just call cub directly
  template <int ALLOCATIONS>
  CUB_RUNTIME_FUNCTION
  cudaError_t alias_storage(void*   storage_ptr,
                            std::size_t& storage_size,
                            void*   (&allocations)[ALLOCATIONS],
                            std::size_t  (&allocation_sizes)[ALLOCATIONS])
  {
    return cub::AliasTemporaries(storage_ptr,
                                 storage_size,
                                 allocations,
                                 allocation_sizes);
  }

} // namespace core
} // namespace cuda_cub
} // end namespace thrust
