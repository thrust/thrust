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

#include <thrust/system/cuda/detail/util.h>
#include <thrust/detail/type_traits/result_of_adaptable_function.h>
#include <thrust/system/cuda/detail/par_to_seq.h>
#include <thrust/system/cuda/detail/core/agent_launcher.h>
#include <thrust/system/cuda/detail/par_to_seq.h>

#include <cub/detail/cdp_dispatch.cuh>
#include <cub/detail/ptx_dispatch.cuh>

namespace thrust
{

namespace cuda_cub {

namespace __parallel_for {

  template <int _BLOCK_THREADS, int _ITEMS_PER_THREAD = 1>
  struct PtxPolicy
  {
    static constexpr int BLOCK_THREADS    = _BLOCK_THREADS;
    static constexpr int ITEMS_PER_THREAD = _ITEMS_PER_THREAD;
    static constexpr int ITEMS_PER_TILE   = BLOCK_THREADS * ITEMS_PER_THREAD;
  };    // struct PtxPolicy

  struct Tuning350 : cub::detail::ptx_base<350>
  {
    using Policy = PtxPolicy<256, 2>;
  };

  template <class F, class Size>
  struct ParallelForAgent
  {
    // List in reverse order:
    using Tunings = cub::detail::type_list<Tuning350>;

    // Required for the AgentLauncher machinery; Tunings provide parameters,
    // the PtxPlan defines subalgorithms, temp_storage, etc.
    template <typename Tuning>
    struct PtxPlan : Tuning::Policy
    {};

    template <typename ActivePtxPlan>
    struct impl
    {
      static constexpr int BLOCK_THREADS    = ActivePtxPlan::BLOCK_THREADS;
      static constexpr int ITEMS_PER_THREAD = ActivePtxPlan::ITEMS_PER_THREAD;

      template <bool IS_FULL_TILE>
      THRUST_DEVICE_FUNCTION
      static void consume_tile(F f, Size tile_base, int items_in_tile)
      {
#pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
          Size idx = BLOCK_THREADS * ITEM + threadIdx.x;
          if (IS_FULL_TILE || idx < items_in_tile)
            f(tile_base + idx);
        }
      }
    }; // end impl

    template <typename ActivePtxPlan>
    THRUST_AGENT_ENTRY(F     f,
                       Size  num_items,
                       char * /*shmem*/ )
    {
      constexpr int ITEMS_PER_TILE   = ActivePtxPlan::ITEMS_PER_TILE;

      Size tile_base     = static_cast<Size>(blockIdx.x) * ITEMS_PER_TILE;
      Size num_remaining = num_items - tile_base;
      Size items_in_tile = static_cast<Size>(
          num_remaining < ITEMS_PER_TILE ? num_remaining : ITEMS_PER_TILE);

      if (items_in_tile == ITEMS_PER_TILE)
      {
        // full tile
        impl<ActivePtxPlan>::consume_tile<true>(f, tile_base, ITEMS_PER_TILE);
      }
      else
      {
        // partial tile
        impl<ActivePtxPlan>::consume_tile<false>(f, tile_base, items_in_tile);
      }
    }
  };    // struct ParallelForAgent

  template <typename F, typename Size>
  CUB_RUNTIME_FUNCTION
  cudaError_t parallel_for(Size num_items, F f, cudaStream_t stream)
  {
    if (num_items == 0)
    {
      return cudaSuccess;
    }

    constexpr bool debug_sync = THRUST_DEBUG_SYNC_FLAG;

    // Create AgentPlan
    using parallel_for_agent_t = ParallelForAgent<F, Size>;
    const auto parallel_for_agent_plan =
      core::AgentPlanFromTunings<parallel_for_agent_t>::get();

    // Create and launch agent:
    using parallel_for_agent_launcher_t = core::AgentLauncher<parallel_for_agent_t>;
    parallel_for_agent_launcher_t pfa(parallel_for_agent_plan,
                                      num_items,
                                      stream,
                                      "parallel_for::agent",
                                      debug_sync);

    using parallel_for_tunings_t = typename parallel_for_agent_t::Tunings;
    pfa.launch_ptx_dispatch(parallel_for_tunings_t{},
                            // Args to Agent::entry:
                            f,
                            num_items);
    CUDA_CUB_RET_IF_FAIL(cudaPeekAtLastError());

    return cudaSuccess;
  }
}    // __parallel_for

__thrust_exec_check_disable__
template <class Derived,
          class F,
          class Size>
void __host__ __device__
parallel_for(execution_policy<Derived> &policy,
             F                          f,
             Size                       count)
{
  if (count == 0)
  {
    return;
  }

  auto run_par = [&]() {
    cudaStream_t stream = cuda_cub::stream(policy);
    cudaError_t  status = __parallel_for::parallel_for(count, f, stream);
    cuda_cub::throw_on_error(status, "parallel_for failed");
  };

  auto run_seq = [&]() {
    // this lambda is only used when CDP is disabled:
#ifndef CUB_RUNTIME_ENABLED
    for (Size idx = 0; idx != count; ++idx)
    {
      f(idx);
    }
#endif
  };

  cub::detail::cdp_dispatch(run_par, run_seq);
}

}    // namespace cuda_cub

} // end namespace thrust
#endif
