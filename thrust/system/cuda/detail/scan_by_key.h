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
#include <thrust/detail/cstdint.h>
#include <thrust/detail/temporary_array.h>
#include <thrust/system/cuda/detail/util.h>

#include <thrust/system/cuda/execution_policy.h>
#include <thrust/system/cuda/detail/par_to_seq.h>
#include <thrust/system/cuda/detail/core/agent_launcher.h>
#include <thrust/detail/mpl/math.h>
#include <thrust/detail/minmax.h>
#include <thrust/distance.h>

#include <cub/detail/ptx_dispatch.cuh>
#include <cub/util_math.cuh>

namespace thrust
{
namespace cuda_cub {

namespace __scan_by_key {
  namespace mpl = thrust::detail::mpl::math;

  template <int                      _BLOCK_THREADS,
            int                      _ITEMS_PER_THREAD = 1,
            cub::BlockLoadAlgorithm  _LOAD_ALGORITHM   = cub::BLOCK_LOAD_DIRECT,
            cub::CacheLoadModifier   _LOAD_MODIFIER    = cub::LOAD_DEFAULT,
            cub::BlockScanAlgorithm  _SCAN_ALGORITHM   = cub::BLOCK_SCAN_WARP_SCANS,
            cub::BlockStoreAlgorithm _STORE_ALGORITHM  = cub::BLOCK_STORE_DIRECT>
  struct PtxPolicy
  {
    static constexpr int BLOCK_THREADS    = _BLOCK_THREADS;
    static constexpr int ITEMS_PER_THREAD = _ITEMS_PER_THREAD;
    static constexpr int ITEMS_PER_TILE   = BLOCK_THREADS * ITEMS_PER_THREAD;

    static constexpr cub::BlockLoadAlgorithm  LOAD_ALGORITHM  = _LOAD_ALGORITHM;
    static constexpr cub::CacheLoadModifier   LOAD_MODIFIER   = _LOAD_MODIFIER;
    static constexpr cub::BlockScanAlgorithm  SCAN_ALGORITHM  = _SCAN_ALGORITHM;
    static constexpr cub::BlockStoreAlgorithm STORE_ALGORITHM = _STORE_ALGORITHM;
  };    // struct PtxPolicy

  template <typename Key, typename Value, int Nominal4BItemsPerThread>
  struct TuningHelper
  {
    static constexpr int MAX_INPUT_BYTES =
      static_cast<int>(mpl::max<size_t, sizeof(Key), sizeof(Value)>::value);
    static constexpr int COMBINED_INPUT_BYTES =
      static_cast<int>(sizeof(Key) + sizeof(Value));

    static constexpr int ITEMS_PER_THREAD =
      (MAX_INPUT_BYTES <= 8)
        ? Nominal4BItemsPerThread
        : mpl::min<int,
                   Nominal4BItemsPerThread,
                   mpl::max<int,
                            1,
                            ((Nominal4BItemsPerThread * 8) +
                             COMBINED_INPUT_BYTES - 1) /
                              COMBINED_INPUT_BYTES>::value>::value;
  }; // TuningBase

  template <typename Key, typename Value>
  struct Tuning350 : cub::detail::ptx_base<350>
  {
    using Helper = TuningHelper<Key, Value, 6>;
    using Policy = PtxPolicy<128,
                             Helper::ITEMS_PER_THREAD,
                             cub::BLOCK_LOAD_WARP_TRANSPOSE,
                             cub::LOAD_LDG,
                             cub::BLOCK_SCAN_WARP_SCANS,
                             cub::BLOCK_STORE_WARP_TRANSPOSE>;
  }; // Tuning350

  template <typename Key, typename Value>
  struct Tuning520 : cub::detail::ptx_base<520>
  {
    using Helper = TuningHelper<Key, Value, 9>;
    using Policy = PtxPolicy<256,
                             Helper::ITEMS_PER_THREAD,
                             cub::BLOCK_LOAD_WARP_TRANSPOSE,
                             cub::LOAD_LDG,
                             cub::BLOCK_SCAN_WARP_SCANS,
                             cub::BLOCK_STORE_WARP_TRANSPOSE>;
  }; // Tuning520

  template <class KeysInputIt,
            class ValuesInputIt,
            class ValuesOutputIt,
            class EqualityOp,
            class ScanOp,
            class Size,
            class ValueType,
            class Inclusive>
  struct ScanByKeyAgent
  {
    using key_type   = typename iterator_traits<KeysInputIt>::value_type;
    using value_type = ValueType;
    using size_type  = Size;

    using size_value_pair_t = cub::KeyValuePair<size_type, value_type>;
    using key_value_pair_t  = cub::KeyValuePair<key_type, value_type>;

    using ScanTileState = cub::ReduceByKeyScanTileState<value_type, size_type>;
    using ReduceBySegmentOp = cub::ReduceBySegmentOp<ScanOp>;

    // List tunings in reverse order:
    using Tunings = cub::detail::type_list<Tuning520<key_type, value_type>,
                                           Tuning350<key_type, value_type>>;

    template <typename Tuning>
    struct PtxPlan : Tuning::Policy
    {
      using KeysLoadIt =
        typename core::LoadIterator<PtxPlan, KeysInputIt>::type;
      using ValuesLoadIt =
        typename core::LoadIterator<PtxPlan, ValuesInputIt>::type;

      using BlockLoadKeys =
        typename core::BlockLoad<PtxPlan, KeysLoadIt, key_type>::type;
      using BlockLoadValues =
        typename core::BlockLoad<PtxPlan, ValuesLoadIt, value_type>::type;

      using BlockStoreValues =
        typename core::BlockStore<PtxPlan, ValuesOutputIt, value_type>::type;

      using BlockDiscontinuityKeys =
        cub::BlockDiscontinuity<key_type,
                                PtxPlan::BLOCK_THREADS,
                                1,
                                1,
                                Tuning::ptx_arch>;

      using TilePrefixCallback = cub::TilePrefixCallbackOp<size_value_pair_t,
                                                           ReduceBySegmentOp,
                                                           ScanTileState,
                                                           Tuning::ptx_arch>;

      using BlockScan = cub::BlockScan<size_value_pair_t,
                                       PtxPlan::BLOCK_THREADS,
                                       PtxPlan::SCAN_ALGORITHM,
                                       1,
                                       1,
                                       Tuning::ptx_arch>;

      union TempStorage
      {
        struct ScanStorage
        {
          typename BlockScan::TempStorage              scan;
          typename TilePrefixCallback::TempStorage     prefix;
          typename BlockDiscontinuityKeys::TempStorage discontinuity;
        } scan_storage;

        typename BlockLoadKeys::TempStorage   load_keys;
        typename BlockLoadValues::TempStorage load_values;

        typename BlockStoreValues::TempStorage store_values;
      };    // union TempStorage
    };      // struct PtxPlan

    template <typename ActivePtxPlan>
    struct impl
    {
      using KeysLoadIt   = typename ActivePtxPlan::KeysLoadIt;
      using ValuesLoadIt = typename ActivePtxPlan::ValuesLoadIt;

      using BlockLoadKeys    = typename ActivePtxPlan::BlockLoadKeys;
      using BlockLoadValues  = typename ActivePtxPlan::BlockLoadValues;
      using BlockStoreValues = typename ActivePtxPlan::BlockStoreValues;

      using BlockDiscontinuityKeys =
        typename ActivePtxPlan::BlockDiscontinuityKeys;
      using TilePrefixCallback = typename ActivePtxPlan::TilePrefixCallback;
      using BlockScan          = typename ActivePtxPlan::BlockScan;
      using TempStorage        = typename ActivePtxPlan::TempStorage;

      static constexpr int BLOCK_THREADS    = ActivePtxPlan::BLOCK_THREADS;
      static constexpr int ITEMS_PER_THREAD = ActivePtxPlan::ITEMS_PER_THREAD;
      static constexpr int ITEMS_PER_TILE   = ActivePtxPlan::ITEMS_PER_TILE;

      //---------------------------------------------------------------------
      // Per thread data
      //---------------------------------------------------------------------

      TempStorage &  storage;
      ScanTileState &tile_state;

      KeysLoadIt     keys_load_it;
      ValuesLoadIt   values_load_it;
      ValuesOutputIt values_output_it;

      cub::InequalityWrapper<EqualityOp> inequality_op;
      ReduceBySegmentOp                  scan_op;


      //---------------------------------------------------------------------
      // Block scan utility methods (first tile)
      //---------------------------------------------------------------------

      // Exclusive scan specialization
      //
      THRUST_DEVICE_FUNCTION void
      scan_tile(size_value_pair_t (&scan_items)[ITEMS_PER_THREAD],
                size_value_pair_t &tile_aggregate,
                thrust::detail::false_type /* is_inclusive */)
      {
        BlockScan(storage.scan_storage.scan)
            .ExclusiveScan(scan_items, scan_items, scan_op, tile_aggregate);
      }

      // Inclusive scan specialization
      //
      THRUST_DEVICE_FUNCTION void
      scan_tile(size_value_pair_t (&scan_items)[ITEMS_PER_THREAD],
                size_value_pair_t &tile_aggregate,
                thrust::detail::true_type /* is_inclusive */)
      {
        BlockScan(storage.scan_storage.scan)
            .InclusiveScan(scan_items, scan_items, scan_op, tile_aggregate);
      }

      //---------------------------------------------------------------------
      // Block scan utility methods (subsequent tiles)
      //---------------------------------------------------------------------

      // Exclusive scan specialization (with prefix from predecessors)
      //
      THRUST_DEVICE_FUNCTION void
      scan_tile(size_value_pair_t (&scan_items)[ITEMS_PER_THREAD],
                size_value_pair_t & tile_aggregate,
                TilePrefixCallback &prefix_op,
                thrust::detail::false_type /* is_incclusive */)
      {
        BlockScan(storage.scan_storage.scan)
            .ExclusiveScan(scan_items, scan_items, scan_op, prefix_op);
        tile_aggregate = prefix_op.GetBlockAggregate();
      }

      // Inclusive scan specialization (with prefix from predecessors)
      //
      THRUST_DEVICE_FUNCTION void
      scan_tile(size_value_pair_t (&scan_items)[ITEMS_PER_THREAD],
                size_value_pair_t & tile_aggregate,
                TilePrefixCallback &prefix_op,
                thrust::detail::true_type /* is_inclusive */)
      {
        BlockScan(storage.scan_storage.scan)
            .InclusiveScan(scan_items, scan_items, scan_op, prefix_op);
        tile_aggregate = prefix_op.GetBlockAggregate();
      }

      //---------------------------------------------------------------------
      // Zip utility methods
      //---------------------------------------------------------------------

      template <bool IS_LAST_TILE>
      THRUST_DEVICE_FUNCTION void
      zip_values_and_flags(size_type num_remaining,
                           value_type (&values)[ITEMS_PER_THREAD],
                           size_type (&segment_flags)[ITEMS_PER_THREAD],
                           size_value_pair_t (&scan_items)[ITEMS_PER_THREAD])
      {
        // Zip values and segment_flags
#pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
          // Set segment_flags for first out-of-bounds item, zero for others
          if (IS_LAST_TILE &&
              Size(threadIdx.x * ITEMS_PER_THREAD) + ITEM == num_remaining)
            segment_flags[ITEM] = 1;

          scan_items[ITEM].value = values[ITEM];
          scan_items[ITEM].key   = segment_flags[ITEM];
        }
      }

      THRUST_DEVICE_FUNCTION void unzip_values(
          value_type (&values)[ITEMS_PER_THREAD],
          size_value_pair_t (&scan_items)[ITEMS_PER_THREAD])
      {
        // Zip values and segment_flags
#pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
          values[ITEM] = scan_items[ITEM].value;
        }
      }

      //---------------------------------------------------------------------
      // Cooperatively scan a device-wide sequence of tiles with other CTAs
      //---------------------------------------------------------------------

      // Process a tile of input (dynamic chained scan)
      //
      template <bool IS_LAST_TILE, class AddInitToScan>
      THRUST_DEVICE_FUNCTION void
      consume_tile(Size          /*num_items*/,
                   Size          num_remaining,
                   int           tile_idx,
                   Size          tile_base,
                   AddInitToScan add_init_to_scan)
      {
        using core::sync_threadblock;

        // Load items
        key_type          keys[ITEMS_PER_THREAD];
        value_type        values[ITEMS_PER_THREAD];
        size_type         segment_flags[ITEMS_PER_THREAD];
        size_value_pair_t scan_items[ITEMS_PER_THREAD];

        if (IS_LAST_TILE)
        {
          // Fill last element with the first element
          // because collectives are not suffix guarded
          BlockLoadKeys(storage.load_keys)
              .Load(keys_load_it + tile_base,
                    keys,
                    num_remaining,
                    *(keys_load_it + tile_base));
        }
        else
        {
          BlockLoadKeys(storage.load_keys)
              .Load(keys_load_it + tile_base, keys);
        }

        sync_threadblock();

        if (IS_LAST_TILE)
        {
          // Fill last element with the first element
          // because collectives are not suffix guarded
          BlockLoadValues(storage.load_values)
              .Load(values_load_it + tile_base,
                    values,
                    num_remaining,
                    *(values_load_it + tile_base));
        }
        else
        {
          BlockLoadValues(storage.load_values)
              .Load(values_load_it + tile_base, values);
        }

        sync_threadblock();

        // first tile
        if (tile_idx == 0)
        {
          BlockDiscontinuityKeys(storage.scan_storage.discontinuity)
            .FlagHeads(segment_flags, keys, inequality_op);

          // Zip values and segment_flags
          zip_values_and_flags<IS_LAST_TILE>(num_remaining,
                                             values,
                                             segment_flags,
                                             scan_items);

          // Exclusive scan of values and segment_flags
          size_value_pair_t tile_aggregate;
          scan_tile(scan_items, tile_aggregate, Inclusive());

          if (threadIdx.x == 0)
          {
            if (!IS_LAST_TILE)
              tile_state.SetInclusive(0, tile_aggregate);

            scan_items[0].key = 0;
          }
        }
        else
        {
          key_type tile_pred_key = (threadIdx.x == 0)
                                       ? keys_load_it[tile_base - 1]
                                       : key_type();
          BlockDiscontinuityKeys(storage.scan_storage.discontinuity)
              .FlagHeads(segment_flags,
                         keys,
                         inequality_op,
                         tile_pred_key);

          // Zip values and segment_flags
          zip_values_and_flags<IS_LAST_TILE>(num_remaining,
                                             values,
                                             segment_flags,
                                             scan_items);

          size_value_pair_t  tile_aggregate;
          TilePrefixCallback prefix_op(tile_state, storage.scan_storage.prefix, scan_op, tile_idx);
          scan_tile(scan_items, tile_aggregate, prefix_op, Inclusive());
        }

        sync_threadblock();

        unzip_values(values, scan_items);

        add_init_to_scan(values, segment_flags);

        // Store items
        if (IS_LAST_TILE)
        {
          BlockStoreValues(storage.store_values)
            .Store(values_output_it + tile_base, values, num_remaining);
        }
        else
        {
          BlockStoreValues(storage.store_values)
            .Store(values_output_it + tile_base, values);
        }
      }

      //---------------------------------------------------------------------
      // Constructor
      //---------------------------------------------------------------------

      // Dequeue and scan tiles of items as part of a dynamic chained scan
      // with Init functor
      template <class AddInitToScan>
      THRUST_DEVICE_FUNCTION
      impl(TempStorage &  storage_,
           ScanTileState &tile_state_,
           KeysInputIt    keys_input_it,
           ValuesInputIt  values_input_it,
           ValuesOutputIt values_output_it_,
           EqualityOp     equality_op_,
           ScanOp         scan_op_,
           Size           num_items,
           AddInitToScan  add_init_to_scan)
          : storage(storage_),
            tile_state(tile_state_),
            keys_load_it(core::make_load_iterator(ActivePtxPlan(), keys_input_it)),
            values_load_it(core::make_load_iterator(ActivePtxPlan(), values_input_it)),
            values_output_it(values_output_it_),
            inequality_op(equality_op_),
            scan_op(scan_op_)
      {
        int  tile_idx      = blockIdx.x;
        Size tile_base     = ITEMS_PER_TILE * tile_idx;
        Size num_remaining = num_items - tile_base;

        if (num_remaining > ITEMS_PER_TILE)
        {
          // Not the last tile (full)
          consume_tile<false>(num_items,
                              num_remaining,
                              tile_idx,
                              tile_base,
                              add_init_to_scan);
        }
        else if (num_remaining > 0)
        {
          // The last tile (possibly partially-full)
          consume_tile<true>(num_items,
                             num_remaining,
                             tile_idx,
                             tile_base,
                             add_init_to_scan);
        }
      }
    };    // struct impl

    //---------------------------------------------------------------------
    // Agent entry point
    //---------------------------------------------------------------------

    template <typename ActivePtxPlan, typename AddInitToScan>
    THRUST_AGENT_ENTRY(KeysInputIt    keys_input_it,
                       ValuesInputIt  values_input_it,
                       ValuesOutputIt values_output_it,
                       EqualityOp     equality_op,
                       ScanOp         scan_op,
                       ScanTileState  tile_state,
                       Size           num_items,
                       AddInitToScan  add_init_to_scan,
                       char *         shmem)
    {
      using temp_storage_t = typename ActivePtxPlan::TempStorage;
      auto &storage        = *reinterpret_cast<temp_storage_t *>(shmem);

      impl<ActivePtxPlan>{storage,
                          tile_state,
                          keys_input_it,
                          values_input_it,
                          values_output_it,
                          equality_op,
                          scan_op,
                          num_items,
                          add_init_to_scan};
    }

  };    // struct ScanByKeyAgent

  template <class ScanTileState,
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
                       char * /*shmem*/)
    {
      tile_state.InitializeStatus(num_tiles);
    }
  }; // struct InitAgent

  template<class T>
  struct DoNothing
  {
    typedef T     type;
    template <int ITEMS_PER_THREAD, class Size>
    THRUST_DEVICE_FUNCTION void
    operator()(T (&/*items*/)[ITEMS_PER_THREAD],
               Size (&/*flags*/)[ITEMS_PER_THREAD])
    {
    }
  };    // struct DoNothing

  template<class ValueType, class ScanOp>
  struct AddInitToScan
  {
    using type = ValueType;

    type init;
    ScanOp scan_op;

    THRUST_RUNTIME_FUNCTION
    AddInitToScan(type init_, ScanOp scan_op_)
        : init(init_), scan_op(scan_op_) {}

    template <int ITEMS_PER_THREAD, class Size>
    THRUST_DEVICE_FUNCTION void
    operator()(type (&items)[ITEMS_PER_THREAD],
               Size (&flags)[ITEMS_PER_THREAD])
    {
#pragma unroll
      for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
      {
        items[ITEM] = flags[ITEM] ? init : scan_op(init, items[ITEM]);
      }
    }
  };    // struct AddInitToScan

  template <class Inclusive,
            class KeysInputIt,
            class ValuesInputIt,
            class ValuesOutputIt,
            class EqualityOp,
            class ScanOp,
            class Size,
            class AddInitToScan>
  THRUST_RUNTIME_FUNCTION cudaError_t
  doit_step(void *         d_temp_storage,
            size_t &       temp_storage_bytes,
            KeysInputIt    keys_input_it,
            ValuesInputIt  values_input_it,
            Size           num_items,
            ValuesOutputIt values_output_it,
            EqualityOp     equality_op,
            ScanOp         scan_op,
            AddInitToScan  add_init_to_scan,
            cudaStream_t   stream,
            bool           debug_sync)
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

    using InitValueType = typename AddInitToScan::type;

    // Declare type aliases for agents, etc:
    using sbk_agent_t = ScanByKeyAgent<KeysInputIt,
                                       ValuesInputIt,
                                       ValuesOutputIt,
                                       EqualityOp,
                                       ScanOp,
                                       Size,
                                       InitValueType,
                                       Inclusive>;

    using scan_tile_state_t = typename sbk_agent_t::ScanTileState;
    using init_agent_t      = InitAgent<scan_tile_state_t, Size>;

    // Create PtxPlans and AgentPlans:
    const auto init_ptx_plan   = typename init_agent_t::PtxPlan{};
    const auto init_agent_plan = core::AgentPlan{init_ptx_plan};
    const auto sbk_agent_plan  = core::AgentPlanFromTunings<sbk_agent_t>::get();

    const int tile_size = sbk_agent_plan.items_per_tile;
    const std::size_t num_tiles = cub::DivideAndRoundUp(num_items, tile_size);

    const std::size_t vshmem_size =
      core::vshmem_size(sbk_agent_plan.shared_memory_size, num_tiles);

    std::size_t allocation_sizes[2] = {0, vshmem_size};
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

    scan_tile_state_t tile_state;
    status = tile_state.Init(static_cast<int>(num_tiles),
                             allocations[0],
                             allocation_sizes[0]);
    CUDA_CUB_RET_IF_FAIL(status);

    char *vshmem_ptr = vshmem_size > 0 ? static_cast<char *>(allocations[1])
                                       : nullptr;

    using init_agent_launcher_t = core::AgentLauncher<init_agent_t>;
    init_agent_launcher_t ia{init_agent_plan,
                             num_tiles,
                             stream,
                             "scan_by_key::init_agent",
                             debug_sync};
    ia.launch_ptx_plan(init_ptx_plan, tile_state, num_tiles);
    CUDA_CUB_RET_IF_FAIL(cudaPeekAtLastError());

    using sbk_agent_launcher_t = core::AgentLauncher<sbk_agent_t>;
    sbk_agent_launcher_t sbka{sbk_agent_plan,
                              num_items,
                              stream,
                              vshmem_ptr,
                              "scan_by_key::scan_agent",
                              debug_sync};
    sbka.launch_ptx_dispatch(typename sbk_agent_t::Tunings{},
                             keys_input_it,
                             values_input_it,
                             values_output_it,
                             equality_op,
                             scan_op,
                             tile_state,
                             num_items,
                             add_init_to_scan);
    CUDA_CUB_RET_IF_FAIL(cudaPeekAtLastError());

    return status;
  }    // func doit_pass

  template <typename Inclusive,
            typename Derived,
            typename KeysInputIt,
            typename ValuesInputIt,
            typename ValuesOutputIt,
            typename EqualityOp,
            typename ScanOp,
            typename AddInitToScan>
  THRUST_RUNTIME_FUNCTION
  ValuesOutputIt scan_by_key(execution_policy<Derived>& policy,
                             KeysInputIt                keys_first,
                             KeysInputIt                keys_last,
                             ValuesInputIt              values_first,
                             ValuesOutputIt             values_result,
                             EqualityOp                 equality_op,
                             ScanOp                     scan_op,
                             AddInitToScan              add_init_to_scan)
  {
    int          num_items    = static_cast<int>(thrust::distance(keys_first, keys_last));
    size_t       storage_size = 0;
    cudaStream_t stream       = cuda_cub::stream(policy);
    bool         debug_sync   = THRUST_DEBUG_SYNC_FLAG;

    if (num_items == 0)
      return values_result;

    cudaError_t status;
    status = doit_step<Inclusive>(NULL,
                                  storage_size,
                                  keys_first,
                                  values_first,
                                  num_items,
                                  values_result,
                                  equality_op,
                                  scan_op,
                                  add_init_to_scan,
                                  stream,
                                  debug_sync);
    cuda_cub::throw_on_error(status, "scan_by_key: failed on 1st step");

    // Allocate temporary storage.
    thrust::detail::temporary_array<thrust::detail::uint8_t, Derived>
      tmp(policy, storage_size);
    void *ptr = static_cast<void*>(tmp.data().get());

    status = doit_step<Inclusive>(ptr,
                                  storage_size,
                                  keys_first,
                                  values_first,
                                  num_items,
                                  values_result,
                                  equality_op,
                                  scan_op,
                                  add_init_to_scan,
                                  stream,
                                  debug_sync);
    cuda_cub::throw_on_error(status, "scan_by_key: failed on 2nd step");

    status = cuda_cub::synchronize(policy);
    cuda_cub::throw_on_error(status, "scan_by_key: failed to synchronize");

    return values_result + num_items;
  }    // func doit
}    // namspace scan_by_key

//-------------------------
// Thrust API entry points
//-------------------------

//---------------------------
//   Inclusive scan
//---------------------------

__thrust_exec_check_disable__
template <class Derived,
          class KeyInputIt,
          class ValInputIt,
          class ValOutputIt,
          class BinaryPred,
          class ScanOp>
ValOutputIt __host__ __device__
inclusive_scan_by_key(execution_policy<Derived> &policy,
                      KeyInputIt                 key_first,
                      KeyInputIt                 key_last,
                      ValInputIt                 value_first,
                      ValOutputIt                value_result,
                      BinaryPred                 binary_pred,
                      ScanOp                     scan_op)
{
  ValOutputIt ret = value_result;
  if (__THRUST_HAS_CUDART__)
  {
    typedef typename iterator_traits<ValInputIt>::value_type T;
    ret = __scan_by_key::scan_by_key<thrust::detail::true_type>(policy,
                                                        key_first,
                                                        key_last,
                                                        value_first,
                                                        value_result,
                                                        binary_pred,
                                                        scan_op,
                                                        __scan_by_key::DoNothing<T>());
  }
  else
  {
#if !__THRUST_HAS_CUDART__
    ret = thrust::inclusive_scan_by_key(cvt_to_seq(derived_cast(policy)),
                                        key_first,
                                        key_last,
                                        value_first,
                                        value_result,
                                        binary_pred,
                                        scan_op);
#endif
  }
  return ret;
}

template <class Derived,
          class KeyInputIt,
          class ValInputIt,
          class ValOutputIt,
          class BinaryPred>
ValOutputIt __host__ __device__
inclusive_scan_by_key(execution_policy<Derived> &policy,
                      KeyInputIt                 key_first,
                      KeyInputIt                 key_last,
                      ValInputIt                 value_first,
                      ValOutputIt                value_result,
                      BinaryPred                 binary_pred)
{
  typedef typename thrust::iterator_traits<ValInputIt>::value_type value_type;
  return cuda_cub::inclusive_scan_by_key(policy,
                                         key_first,
                                         key_last,
                                         value_first,
                                         value_result,
                                         binary_pred,
                                         thrust::plus<>());
}

template <class Derived,
          class KeyInputIt,
          class ValInputIt,
          class ValOutputIt>
ValOutputIt __host__ __device__
inclusive_scan_by_key(execution_policy<Derived> &policy,
                      KeyInputIt                 key_first,
                      KeyInputIt                 key_last,
                      ValInputIt                 value_first,
                      ValOutputIt                value_result)
{
  typedef typename thrust::iterator_traits<KeyInputIt>::value_type key_type;
  return cuda_cub::inclusive_scan_by_key(policy,
                                         key_first,
                                         key_last,
                                         value_first,
                                         value_result,
                                         thrust::equal_to<>());
}


//---------------------------
//   Exclusive scan
//---------------------------

__thrust_exec_check_disable__
template <class Derived,
          class KeyInputIt,
          class ValInputIt,
          class ValOutputIt,
          class Init,
          class BinaryPred,
          class ScanOp>
ValOutputIt __host__ __device__
exclusive_scan_by_key(execution_policy<Derived> &policy,
                      KeyInputIt                 key_first,
                      KeyInputIt                 key_last,
                      ValInputIt                 value_first,
                      ValOutputIt                value_result,
                      Init                       init,
                      BinaryPred                 binary_pred,
                      ScanOp                     scan_op)
{
  ValOutputIt ret = value_result;
  if (__THRUST_HAS_CUDART__)
  {
    ret = __scan_by_key::scan_by_key<thrust::detail::false_type>(
        policy,
        key_first,
        key_last,
        value_first,
        value_result,
        binary_pred,
        scan_op,
        __scan_by_key::AddInitToScan<Init, ScanOp>(init, scan_op));
  }
  else
  {
#if !__THRUST_HAS_CUDART__
    ret = thrust::exclusive_scan_by_key(cvt_to_seq(derived_cast(policy)),
                                        key_first,
                                        key_last,
                                        value_first,
                                        value_result,
                                        init,
                                        binary_pred,
                                        scan_op);
#endif
  }
  return ret;
}

template <class Derived,
          class KeyInputIt,
          class ValInputIt,
          class ValOutputIt,
          class Init,
          class BinaryPred>
ValOutputIt __host__ __device__
exclusive_scan_by_key(execution_policy<Derived> &policy,
                      KeyInputIt                 key_first,
                      KeyInputIt                 key_last,
                      ValInputIt                 value_first,
                      ValOutputIt                value_result,
                      Init                       init,
                      BinaryPred                 binary_pred)
{
  return cuda_cub::exclusive_scan_by_key(policy,
                                         key_first,
                                         key_last,
                                         value_first,
                                         value_result,
                                         init,
                                         binary_pred,
                                         plus<>());
}

template <class Derived,
          class KeyInputIt,
          class ValInputIt,
          class ValOutputIt,
          class Init>
ValOutputIt __host__ __device__
exclusive_scan_by_key(execution_policy<Derived> &policy,
                      KeyInputIt                 key_first,
                      KeyInputIt                 key_last,
                      ValInputIt                 value_first,
                      ValOutputIt                value_result,
                      Init                       init)
{
  typedef typename iterator_traits<KeyInputIt>::value_type key_type;
  return cuda_cub::exclusive_scan_by_key(policy,
                                         key_first,
                                         key_last,
                                         value_first,
                                         value_result,
                                         init,
                                         equal_to<>());
}


template <class Derived,
          class KeyInputIt,
          class ValInputIt,
          class ValOutputIt>
ValOutputIt __host__ __device__
exclusive_scan_by_key(execution_policy<Derived> &policy,
                      KeyInputIt                 key_first,
                      KeyInputIt                 key_last,
                      ValInputIt                 value_first,
                      ValOutputIt                value_result)
{
  typedef typename iterator_traits<ValInputIt>::value_type value_type;
  return cuda_cub::exclusive_scan_by_key(policy,
                                         key_first,
                                         key_last,
                                         value_first,
                                         value_result,
                                         value_type{});
}


}    // namespace cuda_cub
} // end namespace thrust

#include <thrust/scan.h>

#endif
