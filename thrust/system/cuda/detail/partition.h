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
#include <thrust/system/cuda/detail/reverse.h>
#include <thrust/system/cuda/detail/find.h>
#include <thrust/system/cuda/detail/uninitialized_copy.h>
#include <thrust/system/cuda/detail/core/agent_launcher.h>
#include <thrust/system/cuda/detail/par_to_seq.h>
#include <thrust/partition.h>
#include <thrust/pair.h>
#include <thrust/distance.h>

#include <cub/agent/single_pass_scan_operators.cuh> // cub::ScanTileState
#include <cub/block/block_scan.cuh>
#include <cub/detail/cdp_dispatch.cuh>
#include <cub/detail/ptx_dispatch.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>

namespace thrust
{
namespace cuda_cub {

namespace __partition {

  template <int                     _BLOCK_THREADS,
            int                     _ITEMS_PER_THREAD = 1,
            cub::BlockLoadAlgorithm _LOAD_ALGORITHM   = cub::BLOCK_LOAD_DIRECT,
            cub::CacheLoadModifier  _LOAD_MODIFIER    = cub::LOAD_LDG,
            cub::BlockScanAlgorithm _SCAN_ALGORITHM   = cub::BLOCK_SCAN_WARP_SCANS>
  struct PtxPolicy
  {
    static constexpr int BLOCK_THREADS    = _BLOCK_THREADS;
    static constexpr int ITEMS_PER_THREAD = _ITEMS_PER_THREAD;
    static constexpr int ITEMS_PER_TILE   = BLOCK_THREADS * ITEMS_PER_THREAD;

    static constexpr cub::BlockLoadAlgorithm LOAD_ALGORITHM = _LOAD_ALGORITHM;
    static constexpr cub::CacheLoadModifier  LOAD_MODIFIER  = _LOAD_MODIFIER;
    static constexpr cub::BlockScanAlgorithm SCAN_ALGORITHM = _SCAN_ALGORITHM;
  };    // struct PtxPolicy

  template <typename T>
  struct Tuning350 : cub::detail::ptx_base<350>
  {
    static constexpr int INPUT_SIZE = static_cast<int>(sizeof(T));

    static constexpr int NOMINAL_4B_ITEMS_PER_THREAD = 10;
    static constexpr int ITEMS_PER_THREAD =
      CUB_MIN(NOMINAL_4B_ITEMS_PER_THREAD,
              CUB_MAX(1, (NOMINAL_4B_ITEMS_PER_THREAD * 4 / sizeof(T))));

    using Policy = PtxPolicy<128,
                             ITEMS_PER_THREAD,
                             cub::BLOCK_LOAD_WARP_TRANSPOSE,
                             cub::LOAD_LDG,
                             cub::BLOCK_SCAN_WARP_SCANS>;
  }; // Tuning350

  template<int T>
  struct __tag{};

  struct no_stencil_tag_    {};
  struct single_output_tag_
  {
    template<class T>
    THRUST_DEVICE_FUNCTION T const& operator=(T const& t) const { return t; }
  };

  typedef no_stencil_tag_* no_stencil_tag;
  typedef single_output_tag_* single_output_tag;;

  template <class ItemsIt,
            class StencilIt,
            class SelectedOutIt,
            class RejectedOutIt,
            class Predicate,
            class Size,
            class NumSelectedOutIt>
  struct PartitionAgent
  {
    using item_type    = typename iterator_traits<ItemsIt>::value_type;
    using stencil_type = typename iterator_traits<StencilIt>::value_type;

    using ScanTileState = cub::ScanTileState<Size>;

    // List tunings in reverse order:
    using Tunings = cub::detail::type_list<Tuning350<item_type>>;

    template <typename Tuning>
    struct PtxPlan : Tuning::Policy
    {
      using ItemsLoadIt = typename core::LoadIterator<PtxPlan, ItemsIt>::type;
      using StencilLoadIt =
        typename core::LoadIterator<PtxPlan, StencilIt>::type;
      using BlockLoadItems =
        typename core::BlockLoad<PtxPlan, ItemsLoadIt>::type;
      using BlockLoadStencil =
        typename core::BlockLoad<PtxPlan, StencilLoadIt>::type;

      using TilePrefixCallback = cub::
        TilePrefixCallbackOp<Size, cub::Sum, ScanTileState, Tuning::ptx_arch>;
      using BlockScan = cub::BlockScan<Size,
                                       PtxPlan::BLOCK_THREADS,
                                       PtxPlan::SCAN_ALGORITHM,
                                       1,
                                       1,
                                       Tuning::ptx_arch>;

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

      static constexpr bool SINGLE_OUTPUT =
        thrust::detail::is_same<RejectedOutIt, single_output_tag>::value;
      static constexpr bool USE_STENCIL =
        !thrust::detail::is_same<StencilIt, no_stencil_tag>::value;
      static constexpr int BLOCK_THREADS    = ActivePtxPlan::BLOCK_THREADS;
      static constexpr int ITEMS_PER_THREAD = ActivePtxPlan::ITEMS_PER_THREAD;
      static constexpr int ITEMS_PER_TILE   = ActivePtxPlan::ITEMS_PER_TILE;

      //---------------------------------------------------------------------
      // Per-thread fields
      //---------------------------------------------------------------------

      TempStorage &  temp_storage;
      ScanTileState &tile_state;
      ItemsLoadIt    items_glob;
      StencilLoadIt  stencil_glob;
      SelectedOutIt  selected_out_glob;
      RejectedOutIt  rejected_out_glob;
      Predicate      predicate;
      Size           num_items;

      //---------------------------------------------------------------------
      // Utilities
      //---------------------------------------------------------------------

      template <bool IS_LAST_TILE>
      THRUST_DEVICE_FUNCTION void
      scatter(item_type (&items)[ITEMS_PER_THREAD],
              Size (&selection_flags)[ITEMS_PER_THREAD],
              Size (&selection_indices)[ITEMS_PER_THREAD],
              int  num_tile_items,
              int  num_tile_selections,
              Size num_selections_prefix,
              Size num_rejected_prefix,
              Size /*num_selections*/)
      {
        int tile_num_rejections = num_tile_items - num_tile_selections;

        // Scatter items to shared memory (rejections first)
#pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
          int item_idx             = (threadIdx.x * ITEMS_PER_THREAD) + ITEM;
          int local_selection_idx  = selection_indices[ITEM] - num_selections_prefix;
          int local_rejection_idx  = item_idx - local_selection_idx;
          int local_scatter_offset = (selection_flags[ITEM])
                                         ? tile_num_rejections + local_selection_idx
                                         : local_rejection_idx;

          temp_storage.raw_exchange[local_scatter_offset] = items[ITEM];
        }

        core::sync_threadblock();

        // Gather items from shared memory and scatter to global
#pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
          int  item_idx       = (ITEM * BLOCK_THREADS) + threadIdx.x;
          int  rejection_idx  = item_idx;
          int  selection_idx  = item_idx - tile_num_rejections;
          Size scatter_offset = (item_idx < tile_num_rejections)
                                    ? num_items -
                                          num_rejected_prefix - rejection_idx - 1
                                    : num_selections_prefix + selection_idx;

          item_type item = temp_storage.raw_exchange[item_idx];

          if (!IS_LAST_TILE || (item_idx < num_tile_items))
          {
            if (SINGLE_OUTPUT || item_idx >= tile_num_rejections)
            {
              selected_out_glob[scatter_offset] = item;
            }
            else    // if !SINGLE_OUTPUT, scatter rejected items separately
            {
              rejected_out_glob[num_items - scatter_offset - 1] = item;
            }
          }
        }
      }    // func scatter

      //------------------------------------------
      // specialize predicate on different types
      //------------------------------------------

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

      //---------------------------------------------------------------------
      // Tile processing
      //---------------------------------------------------------------------

      template <bool IS_LAST_TILE, bool IS_FIRST_TILE>
      Size THRUST_DEVICE_FUNCTION
      consume_tile_impl(int  num_tile_items,
                        int  tile_idx,
                        Size tile_base)
      {
        item_type items_loc[ITEMS_PER_THREAD];
        Size      selection_flags[ITEMS_PER_THREAD];
        Size      selection_idx[ITEMS_PER_THREAD];

        if (IS_LAST_TILE)
        {
          BlockLoadItems(temp_storage.load_items)
              .Load(items_glob + tile_base, items_loc, num_tile_items);
        }
        else
        {
          BlockLoadItems(temp_storage.load_items)
              .Load(items_glob + tile_base, items_loc);
        }

        core::sync_threadblock();

        if (USE_STENCIL)
        {
          stencil_type stencil_loc[ITEMS_PER_THREAD];

          if (IS_LAST_TILE)
          {
            BlockLoadStencil(temp_storage.load_stencil)
                .Load(stencil_glob + tile_base, stencil_loc, num_tile_items);
          }
          else
          {
            BlockLoadStencil(temp_storage.load_stencil)
                .Load(stencil_glob + tile_base, stencil_loc);
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
        Size num_rejected_prefix   = 0;
        if (IS_FIRST_TILE)
        {
          BlockScan(temp_storage.scan_storage.scan)
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
                                       temp_storage.scan_storage.prefix,
                                       cub::Sum(),
                                       tile_idx);
          BlockScan(temp_storage.scan_storage.scan)
              .ExclusiveSum(selection_flags,
                            selection_idx,
                            prefix_cb);

          num_selections        = prefix_cb.GetInclusivePrefix();
          num_tile_selections   = prefix_cb.GetBlockAggregate();
          num_selections_prefix = prefix_cb.GetExclusivePrefix();
          num_rejected_prefix   = tile_base - num_selections_prefix;

          if (IS_LAST_TILE)
          {
            int num_discount = ITEMS_PER_TILE - num_tile_items;
            num_tile_selections -= num_discount;
            num_selections -= num_discount;
          }
        }

        core::sync_threadblock();

        scatter<IS_LAST_TILE>(items_loc,
                              selection_flags,
                              selection_idx,
                              num_tile_items,
                              num_tile_selections,
                              num_selections_prefix,
                              num_rejected_prefix,
                              num_selections);


        return num_selections;
      }


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
      }

      //---------------------------------------------------------------------
      // Constructor
      //---------------------------------------------------------------------

      THRUST_DEVICE_FUNCTION
      impl(TempStorage &    temp_storage_,
           ScanTileState &  tile_state_,
           ItemsLoadIt      items_glob_,
           StencilLoadIt    stencil_glob_,
           SelectedOutIt    selected_out_glob_,
           RejectedOutIt    rejected_out_glob_,
           Predicate        predicate_,
           Size             num_items_,
           int              num_tiles,
           NumSelectedOutIt num_selected_out)
          : temp_storage(temp_storage_),
            tile_state(tile_state_),
            items_glob(items_glob_),
            stencil_glob(stencil_glob_),
            selected_out_glob(selected_out_glob_),
            rejected_out_glob(rejected_out_glob_),
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
      }    //
    };     //struct impl

    //---------------------------------------------------------------------
    // Agent entry point
    //---------------------------------------------------------------------

    template <typename ActivePtxPlan>
    THRUST_AGENT_ENTRY(ItemsIt          items,
                       StencilIt        stencil,
                       SelectedOutIt    selected_out,
                       RejectedOutIt    rejected_out,
                       Predicate        predicate,
                       Size             num_items,
                       NumSelectedOutIt num_selected_out,
                       ScanTileState    tile_state,
                       int              num_tiles,
                       char *           shmem)
    {
      using temp_storage_t = typename ActivePtxPlan::TempStorage;
      auto &storage        = *reinterpret_cast<temp_storage_t *>(shmem);

      impl<ActivePtxPlan>{storage,
                          tile_state,
                          core::make_load_iterator(ActivePtxPlan(), items),
                          core::make_load_iterator(ActivePtxPlan(), stencil),
                          selected_out,
                          rejected_out,
                          predicate,
                          num_items,
                          num_tiles,
                          num_selected_out};
    }
  };       // struct PartitionAgent

  template <class ScanTileState,
            class NumSelectedIt,
            class Size>
  struct InitAgent
  {
    struct PtxPlan : PtxPolicy<128> {};

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

  }; // struct InitAgent

  template <class ItemsIt,
            class StencilIt,
            class SelectedOutIt,
            class RejectedOutIt,
            class Predicate,
            class Size,
            class NumSelectedOutIt>
  static cudaError_t CUB_RUNTIME_FUNCTION
  doit_step(void *           d_temp_storage,
            size_t &         temp_storage_bytes,
            ItemsIt          items,
            StencilIt        stencil,
            SelectedOutIt    selected_out,
            RejectedOutIt    rejected_out,
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
    using partition_agent_t = PartitionAgent<ItemsIt,
                                             StencilIt,
                                             SelectedOutIt,
                                             RejectedOutIt,
                                             Predicate,
                                             Size,
                                             NumSelectedOutIt>;
    using scan_tile_state_t = typename partition_agent_t::ScanTileState;
    using init_agent_t = InitAgent<scan_tile_state_t, NumSelectedOutIt, Size>;

    // Create PtxPlans and AgentPlans:
    const auto init_ptx_plan = typename init_agent_t::PtxPlan{};
    const auto init_agent_plan = core::AgentPlan{init_ptx_plan};

    const auto partition_agent_plan =
      core::AgentPlanFromTunings<partition_agent_t>::get();

    // Figure out temp_storage_size:
    const int tile_size = partition_agent_plan.items_per_tile;
    const std::size_t num_tiles = cub::DivideAndRoundUp(num_items, tile_size);

    const std::size_t vshmem_storage =
      core::vshmem_size(partition_agent_plan.shared_memory_size, num_tiles);

    std::size_t allocation_sizes[2] = {0, vshmem_storage};
    status = scan_tile_state_t::AllocationSize(static_cast<int>(num_tiles),
                                               allocation_sizes[0]);
    CUDA_CUB_RET_IF_FAIL(status);

    void *allocations[2] = {nullptr, nullptr};
    status = cub::AliasTemporaries(d_temp_storage,
                                   temp_storage_bytes,
                                   allocations,
                                   allocation_sizes);
    CUDA_CUB_RET_IF_FAIL(status);

    if (d_temp_storage == nullptr)
    {
      return status;
    }

    scan_tile_state_t tile_status;
    status = tile_status.Init(static_cast<int>(num_tiles),
                              allocations[0],
                              allocation_sizes[0]);
    CUDA_CUB_RET_IF_FAIL(status);

    using init_agent_launcher = core::AgentLauncher<init_agent_t>;
    init_agent_launcher ia{init_agent_plan,
                           num_tiles,
                           stream,
                           "partition::init_agent",
                           debug_sync};
    ia.launch_ptx_plan(init_ptx_plan, tile_status, num_tiles, num_selected_out);
    CUDA_CUB_RET_IF_FAIL(cudaPeekAtLastError());

    char *vshmem_ptr = vshmem_storage > 0
                         ? reinterpret_cast<char *>(allocations[1])
                         : nullptr;

    using partition_agent_launcher = core::AgentLauncher<partition_agent_t>;
    partition_agent_launcher pa{partition_agent_plan,
                                num_items,
                                stream,
                                vshmem_ptr,
                                "partition::partition_agent",
                                debug_sync};
    pa.launch_ptx_dispatch(typename partition_agent_t::Tunings{},
                           items,
                           stencil,
                           selected_out,
                           rejected_out,
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
            typename SelectedOutIt,
            typename RejectedOutIt,
            typename Predicate>
  CUB_RUNTIME_FUNCTION
  pair<SelectedOutIt, RejectedOutIt>
  partition(execution_policy<Derived>& policy,
            InputIt                    first,
            InputIt                    last,
            StencilIt                  stencil,
            SelectedOutIt              selected_result,
            RejectedOutIt              rejected_result,
            Predicate                  predicate)
  {
    typedef typename iterator_traits<InputIt>::difference_type size_type;

    size_type    num_items          = static_cast<size_type>(thrust::distance(first, last));
    size_t       temp_storage_bytes = 0;
    cudaStream_t stream             = cuda_cub::stream(policy);
    bool         debug_sync         = THRUST_DEBUG_SYNC_FLAG;

    cudaError_t status;
    status = doit_step(NULL,
                       temp_storage_bytes,
                       first,
                       stencil,
                       selected_result,
                       rejected_result,
                       predicate,
                       reinterpret_cast<size_type*>(NULL),
                       num_items,
                       stream,
                       debug_sync);
    cuda_cub::throw_on_error(status, "partition failed on 1st step");

    size_t allocation_sizes[2] = {sizeof(size_type), temp_storage_bytes};
    void * allocations[2]      = {NULL, NULL};

    size_t storage_size = 0;

    status = core::alias_storage(NULL,
                                 storage_size,
                                 allocations,
                                 allocation_sizes);
    cuda_cub::throw_on_error(status, "partition failed on 1st alias_storage");

    // Allocate temporary storage.
    thrust::detail::temporary_array<thrust::detail::uint8_t, Derived>
      tmp(policy, storage_size);
    void *ptr = static_cast<void*>(tmp.data().get());

    status = core::alias_storage(ptr,
                                 storage_size,
                                 allocations,
                                 allocation_sizes);
    cuda_cub::throw_on_error(status, "partition failed on 2nd alias_storage");

    size_type* d_num_selected_out
      = thrust::detail::aligned_reinterpret_cast<size_type*>(allocations[0]);

    status = doit_step(allocations[1],
                       temp_storage_bytes,
                       first,
                       stencil,
                       selected_result,
                       rejected_result,
                       predicate,
                       d_num_selected_out,
                       num_items,
                       stream,
                       debug_sync);
    cuda_cub::throw_on_error(status, "partition failed on 2nd step");

    status = cuda_cub::synchronize(policy);
    cuda_cub::throw_on_error(status, "partition failed to synchronize");

    size_type num_selected = 0;
    if (num_items > 0)
    {
      num_selected = get_value(policy, d_num_selected_out);
    }

    return thrust::make_pair(selected_result + num_selected,
                             rejected_result + num_items - num_selected);
  }

  template <typename Derived,
            typename Iterator,
            typename StencilIt,
            typename Predicate>
  CUB_RUNTIME_FUNCTION
  Iterator partition_inplace(execution_policy<Derived>& policy,
                             Iterator                   first,
                             Iterator                   last,
                             StencilIt                  stencil,
                             Predicate                  predicate)
  {
    typedef typename iterator_traits<Iterator>::difference_type size_type;
    typedef typename iterator_traits<Iterator>::value_type      value_type;

    size_type num_items = thrust::distance(first, last);

    // Allocate temporary storage.
    thrust::detail::temporary_array<value_type, Derived> tmp(policy, num_items);

    cuda_cub::uninitialized_copy(policy, first, last, tmp.begin());

    pair<Iterator, single_output_tag> result =
        partition(policy,
                  tmp.data().get(),
                  tmp.data().get() + num_items,
                  stencil,
                  first,
                  single_output_tag(),
                  predicate);

    size_type num_selected = result.first - first;

    return first + num_selected;
  }
}    // namespace __partition

///// copy

//-------------------------
// Thrust API entry points
//-------------------------

__thrust_exec_check_disable__
template <class Derived,
          class InputIt,
          class StencilIt,
          class SelectedOutIt,
          class RejectedOutIt,
          class Predicate>
pair<SelectedOutIt, RejectedOutIt> __host__ __device__
partition_copy(execution_policy<Derived> &policy,
               InputIt                    first,
               InputIt                    last,
               StencilIt                  stencil,
               SelectedOutIt              selected_result,
               RejectedOutIt              rejected_result,
               Predicate                  predicate)
{
  using result_t = pair<SelectedOutIt, RejectedOutIt>;

  auto run_par = [&]() -> result_t {
    return __partition::partition(policy,
                                  first,
                                  last,
                                  stencil,
                                  selected_result,
                                  rejected_result,
                                  predicate);
  };

  auto run_seq = [&]() -> result_t {
#ifdef CUB_RUNTIME_ENABLED
    // no-op, this lambda is only used when CDP is disabled.
    return thrust::make_pair(selected_result, rejected_result);
#else
    return thrust::partition_copy(cvt_to_seq(derived_cast(policy)),
                                  first,
                                  last,
                                  stencil,
                                  selected_result,
                                  rejected_result,
                                  predicate);
#endif
  };

  return cub::detail::cdp_dispatch(run_par, run_seq);
}

__thrust_exec_check_disable__
template <class Derived,
          class InputIt,
          class SelectedOutIt,
          class RejectedOutIt,
          class Predicate>
pair<SelectedOutIt, RejectedOutIt> __host__ __device__
partition_copy(execution_policy<Derived> &policy,
               InputIt                    first,
               InputIt                    last,
               SelectedOutIt              selected_result,
               RejectedOutIt              rejected_result,
               Predicate                  predicate)
{
  using result_t = pair<SelectedOutIt, RejectedOutIt>;

  auto run_par = [&]() -> result_t {
    return __partition::partition(policy,
                                  first,
                                  last,
                                  __partition::no_stencil_tag(),
                                  selected_result,
                                  rejected_result,
                                  predicate);
  };

  auto run_seq = [&]() -> result_t {
#ifdef CUB_RUNTIME_ENABLED
    // no-op, this lambda is only used when CDP is disabled.
    return thrust::make_pair(selected_result, rejected_result);
#else
    return thrust::partition_copy(cvt_to_seq(derived_cast(policy)),
                                  first,
                                  last,
                                  selected_result,
                                  rejected_result,
                                  predicate);
#endif
  };

  return cub::detail::cdp_dispatch(run_par, run_seq);
}

__thrust_exec_check_disable__
template <class Derived,
          class InputIt,
          class SelectedOutIt,
          class RejectedOutIt,
          class Predicate>
pair<SelectedOutIt, RejectedOutIt> __host__ __device__
stable_partition_copy(execution_policy<Derived> &policy,
                      InputIt                    first,
                      InputIt                    last,
                      SelectedOutIt              selected_result,
                      RejectedOutIt              rejected_result,
                      Predicate                  predicate)
{
  using result_t = pair<SelectedOutIt, RejectedOutIt>;

  auto run_par = [&]() -> result_t {
    return __partition::partition(policy,
                                  first,
                                  last,
                                  __partition::no_stencil_tag(),
                                  selected_result,
                                  rejected_result,
                                  predicate);
  };

  auto run_seq = [&]() -> result_t {
#ifdef CUB_RUNTIME_ENABLED
    // no-op, this lambda is only used when CDP is disabled.
    return thrust::make_pair(selected_result, rejected_result);
#else
    return thrust::stable_partition_copy(cvt_to_seq(derived_cast(policy)),
                                         first,
                                         last,
                                         selected_result,
                                         rejected_result,
                                         predicate);
#endif
  };

  return cub::detail::cdp_dispatch(run_par, run_seq);
}

__thrust_exec_check_disable__
template <class Derived,
          class InputIt,
          class StencilIt,
          class SelectedOutIt,
          class RejectedOutIt,
          class Predicate>
pair<SelectedOutIt, RejectedOutIt> __host__ __device__
stable_partition_copy(execution_policy<Derived> &policy,
                      InputIt                    first,
                      InputIt                    last,
                      StencilIt                  stencil,
                      SelectedOutIt              selected_result,
                      RejectedOutIt              rejected_result,
                      Predicate                  predicate)
{
  using result_t = pair<SelectedOutIt, RejectedOutIt>;

  auto run_par = [&]() -> result_t {
    return __partition::partition(policy,
                                  first,
                                  last,
                                  stencil,
                                  selected_result,
                                  rejected_result,
                                  predicate);
  };

  auto run_seq = [&]() -> result_t {
#ifdef CUB_RUNTIME_ENABLED
    // no-op, this lambda is only used when CDP is disabled.
    return thrust::make_pair(selected_result, rejected_result);
#else
    return thrust::stable_partition_copy(cvt_to_seq(derived_cast(policy)),
                                         first,
                                         last,
                                         stencil,
                                         selected_result,
                                         rejected_result,
                                         predicate);
#endif
  };

  return cub::detail::cdp_dispatch(run_par, run_seq);
}

/// inplace

__thrust_exec_check_disable__
template <class Derived,
          class Iterator,
          class StencilIt,
          class Predicate>
Iterator __host__ __device__
partition(execution_policy<Derived> &policy,
          Iterator                   first,
          Iterator                   last,
          StencilIt                  stencil,
          Predicate                  predicate)
{
  using result_t = Iterator;

  auto run_par = [&]() -> result_t {
    return __partition::partition_inplace(policy,
                                          first,
                                          last,
                                          stencil,
                                          predicate);
  };

  auto run_seq = [&]() -> result_t {
#ifdef CUB_RUNTIME_ENABLED
    // no-op, this lambda is only used when CDP is disabled.
    return first;
#else
    return thrust::partition(cvt_to_seq(derived_cast(policy)),
                             first,
                             last,
                             stencil,
                             predicate);
#endif
  };

  return cub::detail::cdp_dispatch(run_par, run_seq);
}

__thrust_exec_check_disable__
template <class Derived,
          class Iterator,
          class Predicate>
Iterator __host__ __device__
partition(execution_policy<Derived> &policy,
          Iterator                   first,
          Iterator                   last,
          Predicate                  predicate)
{
  using result_t = Iterator;

  auto run_par = [&]() -> result_t {
    return __partition::partition_inplace(policy,
                                          first,
                                          last,
                                          __partition::no_stencil_tag(),
                                          predicate);
  };

  auto run_seq = [&]() -> result_t {
#ifdef CUB_RUNTIME_ENABLED
    // no-op, this lambda is only used when CDP is disabled.
    return first;
#else
    return thrust::partition(cvt_to_seq(derived_cast(policy)),
                             first,
                             last,
                             predicate);
#endif
  };

  return cub::detail::cdp_dispatch(run_par, run_seq);
}

__thrust_exec_check_disable__
template <class Derived,
          class Iterator,
          class StencilIt,
          class Predicate>
Iterator __host__ __device__
stable_partition(execution_policy<Derived> &policy,
                 Iterator                   first,
                 Iterator                   last,
                 StencilIt                  stencil,
                 Predicate                  predicate)
{
  using result_t = Iterator;

  auto run_par = [&]() -> result_t {
    auto result = __partition::partition_inplace(policy,
                                                 first,
                                                 last,
                                                 stencil,
                                                 predicate);

    // partition returns rejected values in reverse order
    // so reverse the rejected elements to make it stable
    cuda_cub::reverse(policy, result, last);

    return result;
  };

  auto run_seq = [&]() -> result_t {
#ifdef CUB_RUNTIME_ENABLED
    // no-op, this lambda is only used when CDP is disabled.
    return first;
#else
    return thrust::stable_partition(cvt_to_seq(derived_cast(policy)),
                                    first,
                                    last,
                                    stencil,
                                    predicate);
#endif
  };

  return cub::detail::cdp_dispatch(run_par, run_seq);
}

__thrust_exec_check_disable__
template <class Derived,
          class Iterator,
          class Predicate>
Iterator __host__ __device__
stable_partition(execution_policy<Derived> &policy,
                 Iterator                   first,
                 Iterator                   last,
                 Predicate                  predicate)
{
  using result_t = Iterator;

  auto run_par = [&]() -> result_t {
    auto result = __partition::partition_inplace(policy,
                                                 first,
                                                 last,
                                                 __partition::no_stencil_tag(),
                                                 predicate);

    // partition returns rejected values in reverse order
    // so reverse the rejected elements to make it stable
    cuda_cub::reverse(policy, result, last);
    return result;
  };

  auto run_seq = [&]() -> result_t {
#ifdef CUB_RUNTIME_ENABLED
    // no-op, this lambda is only used when CDP is disabled.
    return first;
#else
    return thrust::stable_partition(cvt_to_seq(derived_cast(policy)),
                                    first,
                                    last,
                                    predicate);
#endif
  };

  return cub::detail::cdp_dispatch(run_par, run_seq);
}

template <class Derived,
          class ItemsIt,
          class Predicate>
bool __host__ __device__
is_partitioned(execution_policy<Derived> &policy,
               ItemsIt                    first,
               ItemsIt                    last,
               Predicate                  predicate)
{
  ItemsIt boundary = cuda_cub::find_if_not(policy, first, last, predicate);
  ItemsIt end      = cuda_cub::find_if(policy,boundary,last,predicate);
  return end == last;
}


}    // namespace cuda_cub
} // end namespace thrust
#endif
