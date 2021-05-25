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
#include <cub/device/device_select.cuh>
#include <thrust/system/cuda/detail/core/agent_launcher.h>
#include <thrust/system/cuda/detail/core/util.h>
#include <thrust/system/cuda/detail/par_to_seq.h>
#include <thrust/detail/function.h>
#include <thrust/distance.h>
#include <thrust/detail/alignment.h>

#include <cub/detail/cdp_dispatch.cuh>
#include <cub/detail/ptx_dispatch.cuh>
#include <cub/util_math.cuh>

namespace thrust
{
// XXX declare generic copy_if interface
// to avoid circulular dependency from thrust/copy.h
template <typename DerivedPolicy, typename InputIterator, typename OutputIterator, typename Predicate>
__host__ __device__
    OutputIterator
    copy_if(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
            InputIterator                                               first,
            InputIterator                                               last,
            OutputIterator                                              result,
            Predicate                                                   pred);

template <typename DerivedPolicy, typename InputIterator1, typename InputIterator2, typename OutputIterator, typename Predicate>
__host__ __device__
    OutputIterator
    copy_if(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
            InputIterator1                                              first,
            InputIterator1                                              last,
            InputIterator2                                              stencil,
            OutputIterator                                              result,
            Predicate                                                   pred);

namespace cuda_cub {

namespace __copy_if {

  template <int                     _BLOCK_THREADS,
            int                     _ITEMS_PER_THREAD = 1,
            cub::BlockLoadAlgorithm _LOAD_ALGORITHM   = cub::BLOCK_LOAD_DIRECT,
            cub::CacheLoadModifier  _LOAD_MODIFIER    = cub::LOAD_LDG,
            cub::BlockScanAlgorithm _SCAN_ALGORITHM   = cub::BLOCK_SCAN_WARP_SCANS>
  struct PtxPolicy
  {
    static constexpr int BLOCK_THREADS      = _BLOCK_THREADS;
    static constexpr int ITEMS_PER_THREAD   = _ITEMS_PER_THREAD;
    static constexpr int ITEMS_PER_TILE     = BLOCK_THREADS * ITEMS_PER_THREAD;
    static constexpr cub::BlockLoadAlgorithm LOAD_ALGORITHM = _LOAD_ALGORITHM;
    static constexpr cub::CacheLoadModifier  LOAD_MODIFIER  = _LOAD_MODIFIER;
    static constexpr cub::BlockScanAlgorithm SCAN_ALGORITHM = _SCAN_ALGORITHM;
  };    // struct PtxPolicy

  template <class T>
  struct Tuning520 : cub::detail::ptx_base<520>
  {
    static constexpr int NOMINAL_4B_ITEMS_PER_THREAD = 9;

    static constexpr int INPUT_SIZE = static_cast<int>(sizeof(T));
    static constexpr int ITEMS_PER_THREAD =
      CUB_MIN(NOMINAL_4B_ITEMS_PER_THREAD,
              CUB_MAX(1, (NOMINAL_4B_ITEMS_PER_THREAD * 4 / sizeof(T))));

    using Policy = PtxPolicy<128,
                             ITEMS_PER_THREAD,
                             cub::BLOCK_LOAD_WARP_TRANSPOSE,
                             cub::LOAD_LDG,
                             cub::BLOCK_SCAN_WARP_SCANS>;
  }; // Tuning520

  template<class T>
  struct Tuning350 : cub::detail::ptx_base<350>
  {
    static constexpr int NOMINAL_4B_ITEMS_PER_THREAD = 10;

    static constexpr int INPUT_SIZE = static_cast<int>(sizeof(T));
    static constexpr int ITEMS_PER_THREAD =
      CUB_MIN(NOMINAL_4B_ITEMS_PER_THREAD,
              CUB_MAX(1, (NOMINAL_4B_ITEMS_PER_THREAD * 4 / sizeof(T))));

    using Policy = PtxPolicy<128,
                             ITEMS_PER_THREAD,
                             cub::BLOCK_LOAD_WARP_TRANSPOSE,
                             cub::LOAD_LDG,
                             cub::BLOCK_SCAN_WARP_SCANS>;
  }; // Tuning350

  struct no_stencil_tag_    {};
  typedef no_stencil_tag_* no_stencil_tag;

  template <class ItemsIt,
            class StencilIt,
            class OutputIt,
            class Predicate,
            class Size,
            class NumSelectedOutputIt>
  struct CopyIfAgent
  {
    typedef typename iterator_traits<ItemsIt>::value_type   item_type;
    typedef typename iterator_traits<StencilIt>::value_type stencil_type;

    typedef cub::ScanTileState<Size> ScanTileState;

    // List tunings in reverse order:
    using Tunings = cub::detail::type_list<Tuning520<item_type>,
                                           Tuning350<item_type>>;

    template <typename Tuning>
    struct PtxPlan : Tuning::Policy
    {
      typedef typename core::LoadIterator<PtxPlan, ItemsIt>::type   ItemsLoadIt;
      typedef typename core::LoadIterator<PtxPlan, StencilIt>::type StencilLoadIt;

      typedef typename core::BlockLoad<PtxPlan, ItemsLoadIt>::type   BlockLoadItems;
      typedef typename core::BlockLoad<PtxPlan, StencilLoadIt>::type BlockLoadStencil;

      typedef cub::TilePrefixCallbackOp<Size,
                                        cub::Sum,
                                        ScanTileState,
                                        Tuning::ptx_arch>
          TilePrefixCallback;

      typedef cub::BlockScan<Size,
                             PtxPlan::BLOCK_THREADS,
                             PtxPlan::SCAN_ALGORITHM,
                             1,
                             1,
                             Tuning::ptx_arch>
          BlockScan;


      union TempStorage
      {
        struct ScanStorage
        {
          typename BlockScan::TempStorage          scan;
          typename TilePrefixCallback::TempStorage prefix;
        } scan_storage;

        typename BlockLoadItems::TempStorage   load_items;
        typename BlockLoadStencil::TempStorage load_stencil;

        core::uninitialized_array<item_type, PtxPlan::ITEMS_PER_TILE> raw_exchange;
      };    // union TempStorage
    };    // struct PtxPlan

    template <typename ActivePtxPlan>
    struct impl
    {
      using ItemsLoadIt        = typename ActivePtxPlan::ItemsLoadIt;
      using StencilLoadIt      = typename ActivePtxPlan::StencilLoadIt;
      using BlockLoadItems     = typename ActivePtxPlan::BlockLoadItems;
      using BlockLoadStencil   = typename ActivePtxPlan::BlockLoadStencil;
      using TilePrefixCallback = typename ActivePtxPlan::TilePrefixCallback;
      using BlockScan          = typename ActivePtxPlan::BlockScan;
      using TempStorage        = typename ActivePtxPlan::TempStorage;

      static constexpr int BLOCK_THREADS    = ActivePtxPlan::BLOCK_THREADS;
      static constexpr int ITEMS_PER_THREAD = ActivePtxPlan::ITEMS_PER_THREAD;
      static constexpr int ITEMS_PER_TILE   = ActivePtxPlan::ITEMS_PER_TILE;

      static constexpr bool USE_STENCIL =
        !thrust::detail::is_same<StencilIt, no_stencil_tag>::value;

      //---------------------------------------------------------------------
      // Per-thread fields
      //---------------------------------------------------------------------

      TempStorage &  storage;
      ScanTileState &tile_state;
      ItemsLoadIt    items_load_it;
      StencilLoadIt  stencil_load_it;
      OutputIt       output_it;
      Predicate      predicate;
      Size           num_items;

      //------------------------------------------
      // scatter results to memory
      //------------------------------------------

      THRUST_DEVICE_FUNCTION void
      scatter(item_type (&items)[ITEMS_PER_THREAD],
              Size (&selection_flags)[ITEMS_PER_THREAD],
              Size (&selection_indices)[ITEMS_PER_THREAD],
              int  num_tile_selections,
              Size num_selections_prefix)
      {
        using core::sync_threadblock;

#pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
          int local_scatter_offset = selection_indices[ITEM] -
                                     num_selections_prefix;
          if (selection_flags[ITEM])
          {
            new (&storage.raw_exchange[local_scatter_offset]) item_type(items[ITEM]);
          }
        }

        sync_threadblock();

        for (int item = threadIdx.x;
             item < num_tile_selections;
             item += BLOCK_THREADS)
        {
          output_it[num_selections_prefix + item] = storage.raw_exchange[item];
        }
      }    // func scatter

      //------------------------------------------
      // specialize predicate on different types
      //------------------------------------------

      template <int T>
      struct __tag {};

      enum ItemStencil
      {
        ITEM,
        STENCIL
      };

      template <bool TAG, class T>
      struct wrap_value
      {
        T const &              x;
        THRUST_DEVICE_FUNCTION wrap_value(T const &x) : x(x) {}

        THRUST_DEVICE_FUNCTION T const &operator()() const { return x; };
      };    // struct wrap_type

      //------- item

      THRUST_DEVICE_FUNCTION bool
      predicate_wrapper(wrap_value<ITEM, item_type> const &x,
                        __tag<false /* USE_STENCIL */>)
      {
        return predicate(x());
      }

      THRUST_DEVICE_FUNCTION bool
      predicate_wrapper(wrap_value<ITEM, item_type> const &,
                        __tag<true>)
      {
        return false;
      }

      //-------- stencil

      template <class T>
      THRUST_DEVICE_FUNCTION bool
      predicate_wrapper(wrap_value<STENCIL, T> const &x,
                        __tag<true>)
      {
        return predicate(x());
      }

      THRUST_DEVICE_FUNCTION bool
      predicate_wrapper(wrap_value<STENCIL, no_stencil_tag_> const &,
                        __tag<true>)
      {
        return false;
      }


      THRUST_DEVICE_FUNCTION bool
      predicate_wrapper(wrap_value<STENCIL, stencil_type> const &,
                        __tag<false>)
      {
        return false;
      }

      template <bool IS_LAST_TILE, ItemStencil TYPE, class T>
      THRUST_DEVICE_FUNCTION void
      compute_selection_flags(int num_tile_items,
                              T (&values)[ITEMS_PER_THREAD],
                              Size (&selection_flags)[ITEMS_PER_THREAD])
      {
#pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
          // Out-of-bounds items are selection_flags
          selection_flags[ITEM] = 1;

          if (!IS_LAST_TILE ||
              (Size(threadIdx.x * ITEMS_PER_THREAD) + ITEM < num_tile_items))
          {
            selection_flags[ITEM] =
                predicate_wrapper(wrap_value<TYPE, T>(values[ITEM]),
                                  __tag<USE_STENCIL>());
          }
        }
      }

      //------------------------------------------
      // consume tiles
      //------------------------------------------

      template <bool IS_LAST_TILE, bool IS_FIRST_TILE>
      Size THRUST_DEVICE_FUNCTION
      consume_tile_impl(int  num_tile_items,
                        int  tile_idx,
                        Size tile_base)
      {
        item_type items_loc[ITEMS_PER_THREAD];
        Size      selection_flags[ITEMS_PER_THREAD];
        Size      selection_idx[ITEMS_PER_THREAD];

        if (IS_LAST_TILE) {
          BlockLoadItems(storage.load_items)
              .Load(items_load_it + tile_base,
                    items_loc,
                    num_tile_items);
        }
        else
        {
          BlockLoadItems(storage.load_items)
              .Load(items_load_it + tile_base,
                    items_loc);
        }

        core::sync_threadblock();

        if (USE_STENCIL)
        {
          stencil_type stencil_loc[ITEMS_PER_THREAD];

          if (IS_LAST_TILE)
          {
            BlockLoadStencil(storage.load_stencil)
                .Load(stencil_load_it + tile_base,
                      stencil_loc,
                      num_tile_items);
          }
          else
          {
            BlockLoadStencil(storage.load_stencil)
                .Load(stencil_load_it + tile_base,
                      stencil_loc);
          }

          compute_selection_flags<IS_LAST_TILE, STENCIL>(num_tile_items,
                                                         stencil_loc,
                                                         selection_flags);
        }
        else /* Use predicate on items rather then stencil */
        {
          compute_selection_flags<IS_LAST_TILE, ITEM>(num_tile_items,
                                                      items_loc,
                                                      selection_flags);
        }

        core::sync_threadblock();

        Size num_tile_selections   = 0;
        Size num_selections        = 0;
        Size num_selections_prefix = 0;
        if (IS_FIRST_TILE)
        {
          BlockScan(storage.scan_storage.scan)
              .ExclusiveSum(selection_flags,
                            selection_idx,
                            num_tile_selections);

          if (threadIdx.x == 0)
          {
            // Update tile status if this is not the last tile
            if (!IS_LAST_TILE)
              tile_state.SetInclusive(0, num_tile_selections);
          }

          // Do not count any out-of-bounds selections
          if (IS_LAST_TILE)
          {
            int num_discount = ITEMS_PER_TILE - num_tile_items;
            num_tile_selections -= num_discount;
          }
          num_selections = num_tile_selections;
        }
        else
        {
          TilePrefixCallback prefix_cb(tile_state,
                                       storage.scan_storage.prefix,
                                       cub::Sum(),
                                       tile_idx);
          BlockScan(storage.scan_storage.scan)
              .ExclusiveSum(selection_flags,
                            selection_idx,
                            prefix_cb);

          num_selections        = prefix_cb.GetInclusivePrefix();
          num_tile_selections   = prefix_cb.GetBlockAggregate();
          num_selections_prefix = prefix_cb.GetExclusivePrefix();

          if (IS_LAST_TILE)
          {
            int num_discount = ITEMS_PER_TILE - num_tile_items;
            num_tile_selections -= num_discount;
            num_selections -= num_discount;
          }
        }

        core::sync_threadblock();

        scatter(items_loc,
                selection_flags,
                selection_idx,
                num_tile_selections,
                num_selections_prefix);


        return num_selections;
      }    // func consume_tile_impl

      template <bool         IS_LAST_TILE>
      THRUST_DEVICE_FUNCTION Size
      consume_tile(int  num_tile_items,
                   int  tile_idx,
                   Size tile_base)
      {
        if (tile_idx == 0)
        {
          return consume_tile_impl<IS_LAST_TILE, true>(num_tile_items,
                                                       tile_idx,
                                                       tile_base);
        }
        else
        {
          return consume_tile_impl<IS_LAST_TILE, false>(num_tile_items,
                                                        tile_idx,
                                                        tile_base);
        }
      }    // func consume_tile

      //---------------------------------------------------------------------
      // Constructor
      //---------------------------------------------------------------------

      THRUST_DEVICE_FUNCTION impl(TempStorage &       storage_,
                                  ScanTileState &     tile_state_,
                                  ItemsIt             items_it,
                                  StencilIt           stencil_it,
                                  OutputIt            output_it_,
                                  Predicate           predicate_,
                                  Size                num_items_,
                                  int                 num_tiles,
                                  NumSelectedOutputIt num_selected_out)
          : storage(storage_),
            tile_state(tile_state_),
            items_load_it(core::make_load_iterator(ActivePtxPlan{}, items_it)),
            stencil_load_it(core::make_load_iterator(ActivePtxPlan{}, stencil_it)),
            output_it(output_it_),
            predicate(predicate_),
            num_items(num_items_)
      {
        int  tile_idx  = blockIdx.x;
        Size tile_base = tile_idx * ITEMS_PER_TILE;

        if (tile_idx < num_tiles - 1)
        {
          consume_tile<false>(ITEMS_PER_TILE,
                              tile_idx,
                              tile_base);
        }
        else
        {
          int  num_remaining  = static_cast<int>(num_items - tile_base);
          Size num_selections = consume_tile<true>(num_remaining,
                                                   tile_idx,
                                                   tile_base);
          if (threadIdx.x == 0)
          {
            *num_selected_out = num_selections;
          }
        }
      }    // ctor impl
    };

    //---------------------------------------------------------------------
    // Agent entry point
    //---------------------------------------------------------------------

    template <typename ActivePtxPlan>
    THRUST_AGENT_ENTRY(ItemsIt             items_it,
                       StencilIt           stencil_it,
                       OutputIt            output_it,
                       Predicate           predicate,
                       Size                num_items,
                       NumSelectedOutputIt num_selected_out,
                       ScanTileState       tile_state,
                       int                 num_tiles,
                       char *              shmem)
    {
      using temp_storage_t = typename ActivePtxPlan::TempStorage;
      auto &storage = *reinterpret_cast<temp_storage_t *>(shmem);

      impl<ActivePtxPlan>(storage,
                          tile_state,
                          items_it,
                          stencil_it,
                          output_it,
                          predicate,
                          num_items,
                          num_tiles,
                          num_selected_out);
    }
  };    // struct CopyIfAgent

  template <class ScanTileState,
            class NumSelectedIt,
            class Size>
  struct InitAgent
  {
    using PtxPlan = PtxPolicy<128>;

    //---------------------------------------------------------------------
    // Agent entry point
    //---------------------------------------------------------------------

    template <typename /*ActivePtxPlan*/>
    THRUST_AGENT_ENTRY(ScanTileState tile_state,
                       Size          num_tiles,
                       NumSelectedIt num_selected_out,
                       char *        /*shmem*/)
    {
      tile_state.InitializeStatus(num_tiles);
      if (blockIdx.x == 0 && threadIdx.x == 0)
        *num_selected_out = 0;
    }
  };    // struct InitAgent

  template <class ItemsIt,
            class StencilIt,
            class OutputIt,
            class Predicate,
            class Size,
            class NumSelectedOutIt>
  CUB_RUNTIME_FUNCTION
  static cudaError_t doit_step(void *           d_temp_storage,
                               size_t &         temp_storage_bytes,
                               ItemsIt          items,
                               StencilIt        stencil,
                               OutputIt         output_it,
                               Predicate        predicate,
                               NumSelectedOutIt num_selected_out,
                               Size             num_items,
                               cudaStream_t     stream,
                               bool             debug_sync)
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
    using copy_if_agent_t =
      CopyIfAgent<ItemsIt, StencilIt, OutputIt, Predicate, Size, NumSelectedOutIt>;
    using scan_tile_state_t = typename copy_if_agent_t::ScanTileState;
    using init_agent_t = InitAgent<scan_tile_state_t, NumSelectedOutIt, Size>;

    // Create PtxPlans and AgentPlans:
    const auto init_ptx_plan = typename init_agent_t::PtxPlan{};
    const auto init_agent_plan = core::AgentPlan{init_ptx_plan};

    const auto copy_if_agent_plan =
      core::AgentPlanFromTunings<copy_if_agent_t>::get();

    // Figure out temp_storage_size:
    const int tile_size = copy_if_agent_plan.items_per_tile;
    const int num_tiles =
      static_cast<int>(cub::DivideAndRoundUp(num_items, tile_size));

    const std::size_t vshmem_size =
      core::vshmem_size(copy_if_agent_plan.shared_memory_size, num_tiles);

    std::size_t allocation_sizes[2] = {0, vshmem_size};
    status = scan_tile_state_t::AllocationSize(num_tiles, allocation_sizes[0]);
    CUDA_CUB_RET_IF_FAIL(status);

    void *allocations[2] = {nullptr, nullptr};
    status = cub::AliasTemporaries(d_temp_storage,
                                   temp_storage_bytes,
                                   allocations,
                                   allocation_sizes);
    CUDA_CUB_RET_IF_FAIL(status);

    if (d_temp_storage == nullptr)
    { // user is just requesting temp_storage_size.
      return status;
    }

    scan_tile_state_t tile_status;
    status = tile_status.Init(num_tiles, allocations[0], allocation_sizes[0]);
    CUDA_CUB_RET_IF_FAIL(status);

    char *const vshmem_ptr = vshmem_size > 0
                               ? reinterpret_cast<char *>(allocations[1])
                               : nullptr;

    using init_agent_launcher_t = core::AgentLauncher<init_agent_t>;
    init_agent_launcher_t ia{init_agent_plan,
                             num_tiles,
                             stream,
                             "copy_if::init_agent",
                             debug_sync};
    ia.launch_ptx_plan(init_ptx_plan,
                       // Args to Agent::entry:
                       tile_status,
                       num_tiles,
                       num_selected_out);
    CUDA_CUB_RET_IF_FAIL(cudaPeekAtLastError());

    using copy_if_agent_launcher_t = core::AgentLauncher<copy_if_agent_t>;
    copy_if_agent_launcher_t pa{copy_if_agent_plan,
                                num_items,
                                stream,
                                vshmem_ptr,
                                "copy_if::partition_agent",
                                debug_sync};
    pa.launch_ptx_dispatch(typename copy_if_agent_t::Tunings{},
                           // Args to Agent::entry:
                           items,
                           stencil,
                           output_it,
                           predicate,
                           num_items,
                           num_selected_out,
                           tile_status,
                           num_tiles);
    CUDA_CUB_RET_IF_FAIL(cudaPeekAtLastError());

    return status;
  }

  template <typename Derived,
            typename InputIt,
            typename StencilIt,
            typename OutputIt,
            typename Predicate>
  CUB_RUNTIME_FUNCTION
  OutputIt copy_if(execution_policy<Derived>& policy,
                   InputIt                    first,
                   InputIt                    last,
                   StencilIt                  stencil,
                   OutputIt                   output,
                   Predicate                  predicate)
  {
    typedef int size_type;

    size_type    num_items          = static_cast<size_type>(thrust::distance(first, last));
    size_t       temp_storage_bytes = 0;
    cudaStream_t stream             = cuda_cub::stream(policy);
    bool         debug_sync         = THRUST_DEBUG_SYNC_FLAG;

    if (num_items == 0)
      return output;

    cudaError_t status;
    status = doit_step(NULL,
                       temp_storage_bytes,
                       first,
                       stencil,
                       output,
                       predicate,
                       reinterpret_cast<size_type*>(NULL),
                       num_items,
                       stream,
                       debug_sync);
    cuda_cub::throw_on_error(status, "copy_if failed on 1st step");

    size_t allocation_sizes[2] = {sizeof(size_type), temp_storage_bytes};
    void * allocations[2]      = {NULL, NULL};

    size_t storage_size = 0;

    status = core::alias_storage(NULL,
                                 storage_size,
                                 allocations,
                                 allocation_sizes);
    cuda_cub::throw_on_error(status, "copy_if failed on 1st alias_storage");

    // Allocate temporary storage.
    thrust::detail::temporary_array<thrust::detail::uint8_t, Derived>
      tmp(policy, storage_size);
    void *ptr = static_cast<void*>(tmp.data().get());

    status = core::alias_storage(ptr,
                                 storage_size,
                                 allocations,
                                 allocation_sizes);
    cuda_cub::throw_on_error(status, "copy_if failed on 2nd alias_storage");

    size_type* d_num_selected_out
      = thrust::detail::aligned_reinterpret_cast<size_type*>(allocations[0]);

    status = doit_step(allocations[1],
                       temp_storage_bytes,
                       first,
                       stencil,
                       output,
                       predicate,
                       d_num_selected_out,
                       num_items,
                       stream,
                       debug_sync);
    cuda_cub::throw_on_error(status, "copy_if failed on 2nd step");

    status = cuda_cub::synchronize(policy);
    cuda_cub::throw_on_error(status, "copy_if failed to synchronize");

    size_type num_selected = get_value(policy, d_num_selected_out);

    return output + num_selected;
  }

}    // namespace __copy_if

//-------------------------
// Thrust API entry points
//-------------------------

__thrust_exec_check_disable__
template <class Derived,
          class InputIterator,
          class OutputIterator,
          class Predicate>
OutputIterator __host__ __device__
copy_if(execution_policy<Derived> &policy,
        InputIterator              first,
        InputIterator              last,
        OutputIterator             result,
        Predicate                  pred)
{
  auto run_par = [&]() -> OutputIterator {
    return __copy_if::copy_if(policy,
                              first,
                              last,
                              __copy_if::no_stencil_tag(),
                              result,
                              pred);
  };

  auto run_seq = [&]() -> OutputIterator {
#ifdef CUB_RUNTIME_ENABLED
    // no-op, this lambda is only used when CDP is disabled.
    return result;
#else
    return thrust::copy_if(cvt_to_seq(derived_cast(policy)),
                           first,
                           last,
                           result,
                           pred);
#endif
  };

  return cub::detail::cdp_dispatch(run_par, run_seq);
} // func copy_if

__thrust_exec_check_disable__
template <class Derived,
          class InputIterator,
          class StencilIterator,
          class OutputIterator,
          class Predicate>
OutputIterator __host__ __device__
copy_if(execution_policy<Derived> &policy,
        InputIterator              first,
        InputIterator              last,
        StencilIterator            stencil,
        OutputIterator             result,
        Predicate                  pred)
{
  auto run_par = [&]() -> OutputIterator {
    return __copy_if::copy_if(policy, first, last, stencil, result, pred);
  };

  auto run_seq = [&]() -> OutputIterator {
#ifdef CUB_RUNTIME_ENABLED
    // no-op, this lambda is only used when CDP is disabled.
    return result;
#else
    return thrust::copy_if(cvt_to_seq(derived_cast(policy)),
                           first,
                           last,
                           stencil,
                           result,
                           pred);
#endif
  };

  return cub::detail::cdp_dispatch(run_par, run_seq);
}    // func copy_if

}    // namespace cuda_cub
} // end namespace thrust

#include <thrust/copy.h>
#endif
