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


#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC
#include <thrust/system/cuda/config.h>

#include <thrust/detail/cstdint.h>
#include <thrust/detail/temporary_array.h>
#include <thrust/system/cuda/detail/util.h>
#include <thrust/system/cuda/detail/core/agent_launcher.h>
#include <thrust/system/cuda/detail/par_to_seq.h>
#include <thrust/system/cuda/detail/dispatch.h>
#include <thrust/functional.h>
#include <thrust/distance.h>
#include <thrust/detail/mpl/math.h>
#include <thrust/detail/minmax.h>

#include <cub/block/block_adjacent_difference.cuh>
#include <cub/detail/ptx_dispatch.cuh>
#include <cub/device/device_select.cuh>
#include <cub/util_math.cuh>

namespace thrust
{

template <typename DerivedPolicy, typename InputIterator, typename OutputIterator, typename BinaryFunction>
__host__ __device__ OutputIterator
adjacent_difference(
    const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
    InputIterator                                               first,
    InputIterator                                               last,
    OutputIterator                                              result,
    BinaryFunction                                              binary_op);

namespace cuda_cub {

namespace __adjacent_difference {

  namespace mpl = thrust::detail::mpl::math;

  template <int                      _BLOCK_THREADS,
            int                      _ITEMS_PER_THREAD = 1,
            cub::BlockLoadAlgorithm  _LOAD_ALGORITHM   = cub::BLOCK_LOAD_DIRECT,
            cub::CacheLoadModifier   _LOAD_MODIFIER    = cub::LOAD_DEFAULT,
            cub::BlockStoreAlgorithm _STORE_ALGORITHM  = cub::BLOCK_STORE_DIRECT>
  struct PtxPolicy
  {
    static constexpr int BLOCK_THREADS    = _BLOCK_THREADS;
    static constexpr int ITEMS_PER_THREAD = _ITEMS_PER_THREAD;
    static constexpr int ITEMS_PER_TILE   = BLOCK_THREADS * ITEMS_PER_THREAD;

    static constexpr cub::BlockLoadAlgorithm  LOAD_ALGORITHM  = _LOAD_ALGORITHM;
    static constexpr cub::CacheLoadModifier   LOAD_MODIFIER   = _LOAD_MODIFIER;
    static constexpr cub::BlockStoreAlgorithm STORE_ALGORITHM = _STORE_ALGORITHM;
  };

  template<int INPUT_SIZE, int NOMINAL_4B_ITEMS_PER_THREAD>
  struct items_per_thread
  {
    static constexpr int value =
      (INPUT_SIZE <= 8)
        ? NOMINAL_4B_ITEMS_PER_THREAD
        : mpl::min<int,
                   NOMINAL_4B_ITEMS_PER_THREAD,
                   mpl::max<int,
                            1,
                            ((NOMINAL_4B_ITEMS_PER_THREAD * 8) + INPUT_SIZE - 1) /
                              INPUT_SIZE>::value>::value;
  };

  template <class T>
  struct Tuning350 : cub::detail::ptx_base<350>
  {
    static constexpr int INPUT_SIZE = static_cast<int>(sizeof(T));
    static constexpr int NOMINAL_4B_ITEMS_PER_THREAD = 7;
    static constexpr int ITEMS_PER_THREAD =
      items_per_thread<INPUT_SIZE, NOMINAL_4B_ITEMS_PER_THREAD>::value;

    using Policy = PtxPolicy<128,
                             ITEMS_PER_THREAD,
                             cub::BLOCK_LOAD_WARP_TRANSPOSE,
                             cub::LOAD_LDG,
                             cub::BLOCK_STORE_WARP_TRANSPOSE>;
  };

  template <class InputIt,
            class OutputIt,
            class Size,
            class BinaryOp>
  struct AdjacentDifferenceAgent
  {
    typedef typename iterator_traits<InputIt>::value_type input_type;

    // XXX output type must be result of BinaryOp(input_type,input_type);
    typedef input_type output_type;

    // List tunings in reverse order:
    using Tunings = cub::detail::type_list<Tuning350<input_type>>;

    template<class Tuning>
    struct PtxPlan : Tuning::Policy
    {
      typedef typename core::LoadIterator<PtxPlan, InputIt>::type LoadIt;
      typedef typename core::BlockLoad<PtxPlan, LoadIt>::type     BlockLoad;

      typedef typename core::BlockStore<PtxPlan, OutputIt, input_type>::type
          BlockStore;

      typedef cub::BlockAdjacentDifference<input_type,
                                           PtxPlan::BLOCK_THREADS,
                                           1,
                                           1,
                                           Tuning::ptx_arch>
          BlockAdjacentDifference;

      union TempStorage
      {
        typename BlockAdjacentDifference::TempStorage discontinuity;
        typename BlockLoad::TempStorage                load;
        typename BlockStore::TempStorage               store;
      }; // union TempStorage
    }; // struct PtxPlan

    template <typename ActivePtxPlan>
    struct impl
    {
      using BlockAdjacentDifference =
        typename ActivePtxPlan::BlockAdjacentDifference;
      using BlockLoad   = typename ActivePtxPlan::BlockLoad;
      using BlockStore  = typename ActivePtxPlan::BlockStore;
      using LoadIt      = typename ActivePtxPlan::LoadIt;
      using TempStorage = typename ActivePtxPlan::TempStorage;

      static constexpr int ITEMS_PER_THREAD = ActivePtxPlan::ITEMS_PER_THREAD;
      static constexpr int ITEMS_PER_TILE   = ActivePtxPlan::ITEMS_PER_TILE;

      //---------------------------------------------------------------------
      // Per-thread fields
      //---------------------------------------------------------------------

      TempStorage &temp_storage;
      LoadIt       load_it;                // iterator to the first element
      input_type * first_tile_previous;    // iterator to the first element of previous tile value
      OutputIt     output_it;
      BinaryOp     binary_op;

      template <bool IS_LAST_TILE, bool IS_FIRST_TILE>
      void THRUST_DEVICE_FUNCTION
      consume_tile_impl(int  num_remaining,
                        int  tile_idx,
                        Size tile_base)
      {
        input_type  input[ITEMS_PER_THREAD];
        input_type  input_prev[ITEMS_PER_THREAD];
        output_type output[ITEMS_PER_THREAD];

        if (IS_LAST_TILE)
        {
          // Fill last elements with the first element
          // because collectives are not suffix guarded
          BlockLoad(temp_storage.load)
              .Load(load_it + tile_base,
                    input,
                    num_remaining,
                    *(load_it + tile_base));
        }
        else
        {
          BlockLoad(temp_storage.load).Load(load_it + tile_base, input);
        }


        core::sync_threadblock();

        if (IS_FIRST_TILE)
        {
          BlockAdjacentDifference(temp_storage.discontinuity)
              .FlagHeads(output, input, input_prev, binary_op);
          if (threadIdx.x == 0)
            output[0] = input[0];
        }
        else
        {
          input_type tile_prev_input = first_tile_previous[tile_idx];
          BlockAdjacentDifference(temp_storage.discontinuity)
              .FlagHeads(output, input, input_prev, binary_op, tile_prev_input);
        }

        core::sync_threadblock();

        if (IS_LAST_TILE)
        {
          BlockStore(temp_storage.store)
              .Store(output_it + tile_base, output, num_remaining);
        }
        else
        {
          BlockStore(temp_storage.store).Store(output_it + tile_base, output);
        }
      }


      template <bool IS_LAST_TILE>
      void THRUST_DEVICE_FUNCTION
      consume_tile(int  num_remaining,
                   int  tile_idx,
                   Size tile_base)
      {
        if (tile_idx == 0)
        {
          consume_tile_impl<IS_LAST_TILE, true>(num_remaining,
                                                tile_idx,
                                                tile_base);
        }
        else
        {
          consume_tile_impl<IS_LAST_TILE, false>(num_remaining,
                                                 tile_idx,
                                                 tile_base);
        }
      }

      void THRUST_DEVICE_FUNCTION
      consume_range(Size num_items)
      {
        int  tile_idx      = blockIdx.x;
        Size tile_base     = static_cast<Size>(tile_idx) * ITEMS_PER_TILE;
        Size num_remaining = num_items - tile_base;

        if (num_remaining > ITEMS_PER_TILE)    // not a last tile
        {
          consume_tile<false>(num_remaining, tile_idx, tile_base);
        }
        else if (num_remaining > 0)
        {
          consume_tile<true>(num_remaining, tile_idx, tile_base);
        }
      }

      //---------------------------------------------------------------------
      // Constructor
      //---------------------------------------------------------------------

      THRUST_DEVICE_FUNCTION
      impl(TempStorage &temp_storage_,
           InputIt      input_it_,
           input_type * first_tile_previous_,
           OutputIt     result_,
           BinaryOp     binary_op_,
           Size         num_items)
          : temp_storage(temp_storage_),
            load_it(core::make_load_iterator(ActivePtxPlan{}, input_it_)),
            first_tile_previous(first_tile_previous_),
            output_it(result_),
            binary_op(binary_op_)
      {
        consume_range(num_items);
      }
    };    // struct impl

    //---------------------------------------------------------------------
    // Agent entry point
    //---------------------------------------------------------------------

    template <typename ActivePtxPlan>
    THRUST_AGENT_ENTRY(InputIt     first,
                       input_type *first_element,
                       OutputIt    result,
                       BinaryOp    binary_op,
                       Size        num_items,
                       char *      shmem)
    {
      using temp_storage_t = typename ActivePtxPlan::TempStorage;
      auto &storage = *reinterpret_cast<temp_storage_t *>(shmem);

      using impl_t = impl<ActivePtxPlan>;
      impl_t{storage, first, first_element, result, binary_op, num_items};
    }
  }; // struct AdjacentDifferenceAgent

  template <class InputIt,
            class OutputIt,
            class Size>
  struct InitAgent
  {
    struct PtxPlan : PtxPolicy<128> {};

    //---------------------------------------------------------------------
    // Agent entry point
    //---------------------------------------------------------------------

    template <typename /*ActivePtxPlan*/>
    THRUST_AGENT_ENTRY(InputIt  first,
                       OutputIt result,
                       Size     num_tiles,
                       int      items_per_tile,
                       char *   /*shmem*/)
    {
      int tile_idx  = blockIdx.x * blockDim.x + threadIdx.x;
      Size tile_base = static_cast<Size>(tile_idx) * items_per_tile;
      if (tile_base > 0 && tile_idx < num_tiles)
        result[tile_idx] = first[tile_base - 1];
    }
  }; // struct InitAgent

  template <class InputIt,
            class OutputIt,
            class BinaryOp,
            class Size>
  cudaError_t THRUST_RUNTIME_FUNCTION
  doit_step(void *       d_temp_storage,
            size_t &     temp_storage_bytes,
            InputIt      first,
            OutputIt     result,
            BinaryOp     binary_op,
            Size         num_items,
            cudaStream_t stream,
            bool         debug_sync)
  {
    cudaError_t status = cudaSuccess;

    if (!d_temp_storage)
    { // Initialize this for early return.
      temp_storage_bytes = 0;
    }

    if (num_items == 0)
    {
      return status;
    }

    // Declare type aliases for agents, etc:
    using adj_diff_agent_t =
      AdjacentDifferenceAgent<InputIt, OutputIt, Size, BinaryOp>;

    using input_t = typename iterator_traits<InputIt>::value_type;
    using init_agent_t = InitAgent<InputIt, input_t *, Size>;

    // Create PtxPlans and AgentPlans:
    const auto init_ptx_plan = typename init_agent_t::PtxPlan{};
    const thrust::cuda_cub::core::AgentPlan init_agent_plan{init_ptx_plan};

    const auto adj_diff_agent_plan =
      core::AgentPlanFromTunings<adj_diff_agent_t>::get();

    // Work out shmem requirements:
    const Size tile_size = adj_diff_agent_plan.items_per_tile;
    const Size num_tiles = cub::DivideAndRoundUp(num_items, tile_size);

    const std::size_t tmp1 = num_tiles * sizeof(input_t);
    const std::size_t vshmem_size =
      core::vshmem_size(adj_diff_agent_plan.shared_memory_size, num_tiles);

    std::size_t allocation_sizes[2] = {tmp1, vshmem_size};
    void *allocations[2]            = {nullptr, nullptr};

    status = core::alias_storage(d_temp_storage,
                                 temp_storage_bytes,
                                 allocations,
                                 allocation_sizes);
    CUDA_CUB_RET_IF_FAIL(status);

    if (d_temp_storage == nullptr)
    {
      return status;
    }

    input_t *first_tile_previous = reinterpret_cast<input_t *>(allocations[0]);
    char *vshmem_ptr = vshmem_size > 0
                         ? reinterpret_cast<char *>(allocations[1])
                         : nullptr;

    // Launch init kernel:
    using init_agent_launcher_t = core::AgentLauncher<init_agent_t>;
    init_agent_launcher_t ia{init_agent_plan,
                             num_tiles,
                             stream,
                             "adjacent_difference::init_agent",
                             debug_sync};
    ia.launch_ptx_plan(init_ptx_plan,
                       first,
                       first_tile_previous,
                       num_tiles,
                       tile_size);
    CUDA_CUB_RET_IF_FAIL(cudaPeekAtLastError());

    // Launch adjacent difference kernel:
    using adj_diff_agent_launcher_t = core::AgentLauncher<adj_diff_agent_t>;
    adj_diff_agent_launcher_t da{adj_diff_agent_plan,
                                 num_items,
                                 stream,
                                 vshmem_ptr,
                                 "adjacent_difference::difference_agent",
                                 debug_sync};
    da.launch_ptx_dispatch(typename adj_diff_agent_t::Tunings{},
                           first,
                           first_tile_previous,
                           result,
                           binary_op,
                           num_items);
    CUDA_CUB_RET_IF_FAIL(cudaPeekAtLastError());

    return status;
  }

  template <typename Derived,
            typename InputIt,
            typename OutputIt,
            typename BinaryOp>
  OutputIt THRUST_RUNTIME_FUNCTION
  adjacent_difference(execution_policy<Derived>& policy,
                      InputIt                    first,
                      InputIt                    last,
                      OutputIt                   result,
                      BinaryOp                   binary_op)
  {
    typedef typename iterator_traits<InputIt>::difference_type size_type;

    size_type    num_items    = thrust::distance(first, last);
    size_t       storage_size = 0;
    cudaStream_t stream       = cuda_cub::stream(policy);
    bool         debug_sync   = THRUST_DEBUG_SYNC_FLAG;

    cudaError_t status;
    THRUST_INDEX_TYPE_DISPATCH(status, doit_step, num_items,
        (NULL, storage_size, first, result, binary_op,
           num_items_fixed, stream, debug_sync));
    cuda_cub::throw_on_error(status, "adjacent_difference failed on 1st step");

    // Allocate temporary storage.
    thrust::detail::temporary_array<thrust::detail::uint8_t, Derived>
      tmp(policy, storage_size);
    void *ptr = static_cast<void*>(tmp.data().get());

    THRUST_INDEX_TYPE_DISPATCH(status, doit_step, num_items,
        (ptr, storage_size, first, result, binary_op,
           num_items_fixed, stream, debug_sync));
    cuda_cub::throw_on_error(status, "adjacent_difference failed on 2nd step");

    status = cuda_cub::synchronize(policy);
    cuda_cub::throw_on_error(status, "adjacent_difference failed to synchronize");

    return result + num_items;
  }

}    // namespace __adjacent_difference

//-------------------------
// Thrust API entry points
//-------------------------

__thrust_exec_check_disable__
template <class Derived,
          class InputIt,
          class OutputIt,
          class BinaryOp>
OutputIt __host__ __device__
adjacent_difference(execution_policy<Derived> &policy,
                    InputIt                    first,
                    InputIt                    last,
                    OutputIt                   result,
                    BinaryOp                   binary_op)
{
  OutputIt ret = result;
  if (__THRUST_HAS_CUDART__)
  {
    ret = __adjacent_difference::adjacent_difference(policy,
        first,
        last,
        result,
        binary_op);
  }
  else
  {
#if !__THRUST_HAS_CUDART__
    ret = thrust::adjacent_difference(cvt_to_seq(derived_cast(policy)),
                                      first,
                                      last,
                                      result,
                                      binary_op);
#endif
  }

  return ret;
}

template <class Derived,
          class InputIt,
          class OutputIt>
OutputIt __host__ __device__
adjacent_difference(execution_policy<Derived> &policy,
                    InputIt                    first,
                    InputIt                    last,
                    OutputIt                   result)
{
  typedef typename iterator_traits<InputIt>::value_type input_type;
  return cuda_cub::adjacent_difference(policy,
                                       first,
                                       last,
                                       result,
                                       minus<input_type>());
}


} // namespace cuda_cub
} // end namespace thrust

//
#include <thrust/memory.h>
#include <thrust/adjacent_difference.h>
#endif

