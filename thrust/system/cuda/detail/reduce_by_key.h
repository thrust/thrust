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
#include <thrust/detail/type_traits.h>

#include <thrust/detail/cstdint.h>
#include <thrust/detail/temporary_array.h>
#include <thrust/system/cuda/detail/util.h>
#include <thrust/detail/raw_reference_cast.h>
#include <thrust/detail/type_traits/iterator/is_output_iterator.h>
#include <thrust/system/cuda/detail/par_to_seq.h>
#include <thrust/system/cuda/detail/core/agent_launcher.h>
#include <thrust/system/cuda/detail/get_value.h>
#include <thrust/pair.h>
#include <thrust/functional.h>
#include <thrust/detail/mpl/math.h>
#include <thrust/detail/minmax.h>
#include <thrust/distance.h>
#include <thrust/detail/alignment.h>

// DeviceReduce is not actually used, but some implementation details are.
// These algorithms need to be unified.
#include <cub/device/device_reduce.cuh>

#include <cub/detail/ptx_dispatch.cuh>
#include <cub/util_math.cuh>

namespace thrust
{

template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator1,
          typename OutputIterator2,
          typename BinaryPredicate>
__host__ __device__ thrust::pair<OutputIterator1, OutputIterator2>
reduce_by_key(
    const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
    InputIterator1                                              keys_first,
    InputIterator1                                              keys_last,
    InputIterator2                                              values_first,
    OutputIterator1                                             keys_output,
    OutputIterator2                                             values_output,
    BinaryPredicate                                             binary_pred);

namespace cuda_cub {

namespace __reduce_by_key {

  template<bool> struct is_true : thrust::detail::false_type {};
  template<> struct is_true<true> : thrust::detail::true_type {};

  namespace mpl = thrust::detail::mpl::math;

  template <int                     _BLOCK_THREADS,
            int                     _ITEMS_PER_THREAD = 1,
            cub::BlockLoadAlgorithm _LOAD_ALGORITHM   = cub::BLOCK_LOAD_DIRECT,
            cub::CacheLoadModifier  _LOAD_MODIFIER    = cub::LOAD_DEFAULT,
            cub::BlockScanAlgorithm _SCAN_ALGORITHM   = cub::BLOCK_SCAN_WARP_SCANS>
  struct PtxPolicy
  {
    static constexpr int BLOCK_THREADS    = _BLOCK_THREADS;
    static constexpr int ITEMS_PER_THREAD = _ITEMS_PER_THREAD;
    static constexpr int ITEMS_PER_TILE   = BLOCK_THREADS * ITEMS_PER_THREAD;

    static constexpr cub::BlockLoadAlgorithm LOAD_ALGORITHM = _LOAD_ALGORITHM;
    static constexpr cub::CacheLoadModifier LOAD_MODIFIER   = _LOAD_MODIFIER;
    static constexpr cub::BlockScanAlgorithm SCAN_ALGORITHM = _SCAN_ALGORITHM;
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
        : mpl::min<
            int,
            Nominal4BItemsPerThread,
            mpl::max<int,
                     1,
                     ((Nominal4BItemsPerThread * 8) + COMBINED_INPUT_BYTES - 1) /
                       COMBINED_INPUT_BYTES>::value>::value;
  }; // TuningHelper

  template <typename Key, typename Value>
  struct Tuning350 : cub::detail::ptx_base<350>
  {
    using Helper = TuningHelper<Key, Value, 6>;
    using Policy = PtxPolicy<128,
                             Helper::ITEMS_PER_THREAD,
                             cub::BLOCK_LOAD_WARP_TRANSPOSE,
                             cub::LOAD_LDG,
                             cub::BLOCK_SCAN_WARP_SCANS>;
  }; // Tuning350

  template <typename Key, typename Value>
  struct Tuning520 : cub::detail::ptx_base<520>
  {
    using Helper = TuningHelper<Key, Value, 9>;
    using Policy = PtxPolicy<256,
                             Helper::ITEMS_PER_THREAD,
                             cub::BLOCK_LOAD_WARP_TRANSPOSE,
                             cub::LOAD_LDG,
                             cub::BLOCK_SCAN_WARP_SCANS>;
  }; // Tuning520

  template <class KeysInputIt,
            class ValuesInputIt,
            class KeysOutputIt,
            class ValuesOutputIt,
            class EqualityOp,
            class ReductionOp,
            class NumRunsOutputIt,
            class Size>
  struct ReduceByKeyAgent
  {
    using key_type   = typename iterator_traits<KeysInputIt>::value_type;
    using value_type = typename iterator_traits<ValuesInputIt>::value_type;
    using size_type  = Size;

    using size_value_pair_t = cub::KeyValuePair<size_type, value_type>;
    using key_value_pair_t  = cub::KeyValuePair<key_type, value_type>;

    using ScanTileState = cub::ReduceByKeyScanTileState<value_type, size_type>;
    using ReduceBySegmentOp = cub::ReduceBySegmentOp<ReductionOp>;

    // List tunings in reverse order:
    using Tunings = cub::detail::type_list<Tuning520<key_type, value_type>,
                                           Tuning350<key_type, value_type>>;

    template<typename Tuning>
    struct PtxPlan : Tuning::Policy
    {
      using KeysLoadIt =
        typename core::LoadIterator<PtxPlan, KeysInputIt>::type;
      using ValuesLoadIt =
        typename core::LoadIterator<PtxPlan, ValuesInputIt>::type;

      using BlockLoadKeys = typename core::BlockLoad<PtxPlan, KeysLoadIt>::type;
      using BlockLoadValues =
        typename core::BlockLoad<PtxPlan, ValuesLoadIt>::type;

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

        core::uninitialized_array<key_value_pair_t, PtxPlan::ITEMS_PER_TILE + 1>
          raw_exchange;
      };    // union TempStorage
    };  // struct PtxPlan

    template <typename ActivePtxPlan>
    struct impl
    {
      using KeysLoadIt             = typename ActivePtxPlan::KeysLoadIt;
      using ValuesLoadIt           = typename ActivePtxPlan::ValuesLoadIt;
      using BlockLoadKeys          = typename ActivePtxPlan::BlockLoadKeys;
      using BlockLoadValues        = typename ActivePtxPlan::BlockLoadValues;
      using BlockDiscontinuityKeys = typename ActivePtxPlan::BlockDiscontinuityKeys;
      using TilePrefixCallback     = typename ActivePtxPlan::TilePrefixCallback;
      using BlockScan              = typename ActivePtxPlan::BlockScan;
      using TempStorage            = typename ActivePtxPlan::TempStorage;

      static constexpr int BLOCK_THREADS      = ActivePtxPlan::BLOCK_THREADS;
      static constexpr int ITEMS_PER_THREAD   = ActivePtxPlan::ITEMS_PER_THREAD;
      static constexpr int ITEMS_PER_TILE     = ActivePtxPlan::ITEMS_PER_TILE;
      static constexpr bool TWO_PHASE_SCATTER = (ITEMS_PER_THREAD > 1);

      // Whether or not the scan operation has a zero-valued identity value
      // (true if we're performing addition on a primitive type)
      static constexpr bool HAS_IDENTITY_ZERO =
        thrust::detail::is_same<ReductionOp, plus<value_type>>::value &&
        thrust::detail::is_arithmetic<value_type>::value;

      //---------------------------------------------------------------------
      // Per-thread fields
      //---------------------------------------------------------------------

      TempStorage &                      storage;
      KeysLoadIt                         keys_load_it;
      ValuesLoadIt                       values_load_it;
      KeysOutputIt                       keys_output_it;
      ValuesOutputIt                     values_output_it;
      NumRunsOutputIt                    num_runs_output_it;
      cub::InequalityWrapper<EqualityOp> inequality_op;
      ReduceBySegmentOp                  scan_op;

      //---------------------------------------------------------------------
      // Block scan utility methods
      //---------------------------------------------------------------------

      // Scan with identity (first tile)
      //
      THRUST_DEVICE_FUNCTION void
      scan_tile(size_value_pair_t (&scan_items)[ITEMS_PER_THREAD],
                size_value_pair_t &tile_aggregate,
                thrust::detail::true_type /* has_identity */)
      {
        size_value_pair_t identity;
        identity.value = 0;
        identity.key   = 0;
        BlockScan(storage.scan_storage.scan)
            .ExclusiveScan(scan_items, scan_items, identity, scan_op, tile_aggregate);
      }

      // Scan without identity (first tile).
      // Without an identity, the first output item is undefined.
      //
      THRUST_DEVICE_FUNCTION void
      scan_tile(size_value_pair_t (&scan_items)[ITEMS_PER_THREAD],
                size_value_pair_t &tile_aggregate,
                thrust::detail::false_type /* has_identity */)
      {
        BlockScan(storage.scan_storage.scan)
            .ExclusiveScan(scan_items, scan_items, scan_op, tile_aggregate);
      }

      // Scan with identity (subsequent tile)
      //
      THRUST_DEVICE_FUNCTION void
      scan_tile(size_value_pair_t (&scan_items)[ITEMS_PER_THREAD],
                size_value_pair_t & tile_aggregate,
                TilePrefixCallback &prefix_op,
                thrust::detail::true_type /*  has_identity */)
      {
        BlockScan(storage.scan_storage.scan)
            .ExclusiveScan(scan_items,
                           scan_items,
                           scan_op,
                           prefix_op);
        tile_aggregate = prefix_op.GetBlockAggregate();
      }

      // Scan without identity (subsequent tile).
      // Without an identity, the first output item is undefined.
      THRUST_DEVICE_FUNCTION void
      scan_tile(size_value_pair_t (&scan_items)[ITEMS_PER_THREAD],
                size_value_pair_t & tile_aggregate,
                TilePrefixCallback &prefix_op,
                thrust::detail::false_type /* has_identity */)
      {
        BlockScan(storage.scan_storage.scan)
            .ExclusiveScan(scan_items,
                           scan_items,
                           scan_op,
                           prefix_op);
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

      THRUST_DEVICE_FUNCTION void zip_keys_and_values(
          key_type (&keys)[ITEMS_PER_THREAD],
          size_type (&segment_indices)[ITEMS_PER_THREAD],
          size_value_pair_t (&scan_items)[ITEMS_PER_THREAD],
          key_value_pair_t (&scatter_items)[ITEMS_PER_THREAD])
      {
        // Zip values and segment_flags
#pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
          scatter_items[ITEM].key   = keys[ITEM];
          scatter_items[ITEM].value = scan_items[ITEM].value;
          segment_indices[ITEM]     = scan_items[ITEM].key;
        }
      }

      //---------------------------------------------------------------------
      // Scatter utility methods
      //---------------------------------------------------------------------

      // Directly scatter flagged items to output offsets
      // (specialized for IS_SEGMENTED_REDUCTION_FIXUP == false)
      THRUST_DEVICE_FUNCTION void scatter_direct(
          key_value_pair_t (&scatter_items)[ITEMS_PER_THREAD],
          size_type (&segment_flags)[ITEMS_PER_THREAD],
          size_type (&segment_indices)[ITEMS_PER_THREAD])
      {
        // Scatter flagged keys and values
#pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
          if (segment_flags[ITEM])
          {
            keys_output_it[segment_indices[ITEM]] = scatter_items[ITEM].key;
            values_output_it[segment_indices[ITEM]] = scatter_items[ITEM].value;
          }
        }
      }

      // 2-phase scatter flagged items to output offsets
      // (specialized for IS_SEGMENTED_REDUCTION_FIXUP == false
      //
      // The exclusive scan causes each head flag to be paired with
      // the previous value aggregate:
      //   * the scatter offsets must be decremented for value aggregates
      //
      THRUST_DEVICE_FUNCTION void scatter_two_phase(
          key_value_pair_t (&scatter_items)[ITEMS_PER_THREAD],
          size_type (&segment_flags)[ITEMS_PER_THREAD],
          size_type (&segment_indices)[ITEMS_PER_THREAD],
          size_type num_tile_segments,
          size_type num_tile_segments_prefix)
      {
        using core::sync_threadblock;

        sync_threadblock();

        // Compact and scatter keys
#pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
          if (segment_flags[ITEM])
          {
            storage.raw_exchange[segment_indices[ITEM] -
                                 num_tile_segments_prefix] = scatter_items[ITEM];
          }
        }

        sync_threadblock();

        for (int item = threadIdx.x; item < num_tile_segments; item += BLOCK_THREADS)
        {
          size_type        idx  = num_tile_segments_prefix + item;
          key_value_pair_t pair = storage.raw_exchange[item];
          keys_output_it[idx]   = pair.key;
          values_output_it[idx] = pair.value;
        }
      }


      // Scatter flagged items
      //
      THRUST_DEVICE_FUNCTION void scatter(
          key_value_pair_t (&scatter_items)[ITEMS_PER_THREAD],
          size_type (&segment_flags)[ITEMS_PER_THREAD],
          size_type (&segment_indices)[ITEMS_PER_THREAD],
          size_type num_tile_segments,
          size_type num_tile_segments_prefix)
      {
        // Do a one-phase scatter if (a) two-phase is disabled or
        // (b) the average number of selected items per thread is less than one
        if (TWO_PHASE_SCATTER && (num_tile_segments > BLOCK_THREADS))
        {
          scatter_two_phase(scatter_items,
                            segment_flags,
                            segment_indices,
                            num_tile_segments,
                            num_tile_segments_prefix);
        }
        else
        {
          scatter_direct(scatter_items,
                         segment_flags,
                         segment_indices);
        }
      }

      //---------------------------------------------------------------------
      // Finalization utility methods
      //---------------------------------------------------------------------

      // Finalize the carry-out from the last tile
      // (specialized for IS_SEGMENTED_REDUCTION_FIXUP == false)
      THRUST_DEVICE_FUNCTION void
      finalize_last_tile(size_type num_segments,
                         size_type num_remaining,
                         key_type    last_key,
                         value_type  last_value)
      {
        // Last thread will output final count and last item, if necessary
        if (threadIdx.x == BLOCK_THREADS - 1)
        {
          // If the last tile is a whole tile, the inclusive prefix
          // contains accumulated value reduction for the last segment
          if (num_remaining == ITEMS_PER_TILE)
          {
            // Scatter key and value
            keys_output_it[num_segments]   = last_key;
            values_output_it[num_segments] = last_value;
            num_segments++;
          }

          // Output the total number of items selected
          *num_runs_output_it = num_segments;
        }
      }

      //---------------------------------------------------------------------
      // Cooperatively scan a device-wide sequence of tiles with other CTAs
      //---------------------------------------------------------------------

      // Process first tile of input (dynamic chained scan).
      // Returns the running  count of segments
      // and aggregated values (including this tile)
      //
      template <bool IS_LAST_TILE>
      THRUST_DEVICE_FUNCTION void
      consume_first_tile(Size           num_remaining,
                         Size           tile_offset,
                         ScanTileState &tile_state)
      {
        using core::sync_threadblock;

        key_type          keys[ITEMS_PER_THREAD];               // Tile keys
        key_type          pred_keys[ITEMS_PER_THREAD];          // Tile keys shifted up (predecessor)
        value_type        values[ITEMS_PER_THREAD];             // Tile values
        size_type         segment_flags[ITEMS_PER_THREAD];      // Segment head flags
        size_type         segment_indices[ITEMS_PER_THREAD];    // Segment indices
        size_value_pair_t scan_items[ITEMS_PER_THREAD];         // Zipped values and segment flags|indices
        key_value_pair_t  scatter_items[ITEMS_PER_THREAD];      // Zipped key value pairs for scattering

        // Load keys (last tile repeats final element)
        if (IS_LAST_TILE)
        {
          // Fill last elements with the first element
          // because collectives are not suffix guarded
          BlockLoadKeys(storage.load_keys)
              .Load(keys_load_it + tile_offset,
                    keys,
                    num_remaining,
                    *(keys_load_it + tile_offset));
        }
        else
        {
          BlockLoadKeys(storage.load_keys)
              .Load(keys_load_it + tile_offset, keys);
        }

        sync_threadblock();

        // Load values (last tile repeats final element)
        if (IS_LAST_TILE)
        {
          BlockLoadValues(storage.load_values)
              .Load(values_load_it + tile_offset,
                    values,
                    num_remaining,
                    *(values_load_it + tile_offset));
        }
        else
        {
          BlockLoadValues(storage.load_values)
              .Load(values_load_it + tile_offset, values);
        }

        sync_threadblock();

        // Set head segment_flags.
        // First tile sets the first flag for the first item
        BlockDiscontinuityKeys(storage.scan_storage.discontinuity)
            .FlagHeads(segment_flags, keys, pred_keys, inequality_op);

        // Unset the flag for the first item in the first tile
        // so we won't scatter it
        //
        if (threadIdx.x == 0)
          segment_flags[0] = 0;

        // Zip values and segment_flags
        zip_values_and_flags<IS_LAST_TILE>(num_remaining,
                                           values,
                                           segment_flags,
                                           scan_items);

        // Exclusive scan of values and segment_flags
        size_value_pair_t tile_aggregate;
        scan_tile(scan_items, tile_aggregate, is_true<HAS_IDENTITY_ZERO>());

        if (threadIdx.x == 0)
        {
          // Update tile status if this is not the last tile
          if (!IS_LAST_TILE)
            tile_state.SetInclusive(0, tile_aggregate);

          // Initialize the segment index for the first scan item if necessary
          // (the exclusive prefix for the first item is garbage)
          if (!HAS_IDENTITY_ZERO)
            scan_items[0].key = 0;
        }

        // Unzip values and segment indices
        zip_keys_and_values(pred_keys,
                            segment_indices,
                            scan_items,
                            scatter_items);

        // Scatter flagged items
        scatter(scatter_items,
                segment_flags,
                segment_indices,
                tile_aggregate.key,
                0);

        if (IS_LAST_TILE)
        {
          // Finalize the carry-out from the last tile
          finalize_last_tile(tile_aggregate.key,
                             num_remaining,
                             keys[ITEMS_PER_THREAD - 1],
                             tile_aggregate.value);
        }
      }

      // Process subsequent tile of input (dynamic chained scan).
      // Returns the running count of segments
      // and aggregated values (including this tile)

      template <bool IS_LAST_TILE>
      THRUST_DEVICE_FUNCTION void
      consume_subsequent_tile(Size           num_remaining,
                              int            tile_idx,
                              Size           tile_offset,
                              ScanTileState &tile_state)
      {
        using core::sync_threadblock;

        key_type          keys[ITEMS_PER_THREAD];               // Tile keys
        key_type          pred_keys[ITEMS_PER_THREAD];          // Tile keys shifted up (predecessor)
        value_type        values[ITEMS_PER_THREAD];             // Tile values
        size_type         segment_flags[ITEMS_PER_THREAD];      // Segment head flags
        size_type         segment_indices[ITEMS_PER_THREAD];    // Segment indices
        size_value_pair_t scan_items[ITEMS_PER_THREAD];         // Zipped values and segment flags|indices
        key_value_pair_t  scatter_items[ITEMS_PER_THREAD];      // Zipped key value pairs for scattering

        // Load keys (last tile repeats final element)
        if (IS_LAST_TILE)
        {
          BlockLoadKeys(storage.load_keys)
              .Load(keys_load_it + tile_offset,
                    keys,
                    num_remaining,
                    *(keys_load_it + tile_offset));
        }
        else
        {
          BlockLoadKeys(storage.load_keys)
              .Load(keys_load_it + tile_offset, keys);
        }

        key_type tile_pred_key = (threadIdx.x == 0)
                                     ? keys_load_it[tile_offset - 1]
                                     : key_type();

        sync_threadblock();

        // Load values (last tile repeats final element)
        if (IS_LAST_TILE)
        {
          BlockLoadValues(storage.load_values)
              .Load(values_load_it + tile_offset,
                    values,
                    num_remaining,
                    *(values_load_it + tile_offset));
        }
        else
        {
          BlockLoadValues(storage.load_values)
              .Load(values_load_it + tile_offset, values);
        }

        sync_threadblock();

        // Set head segment_flags
        BlockDiscontinuityKeys(storage.scan_storage.discontinuity)
            .FlagHeads(segment_flags,
                       keys,
                       pred_keys,
                       inequality_op,
                       tile_pred_key);

        // Zip values and segment_flags
        zip_values_and_flags<IS_LAST_TILE>(num_remaining,
                                           values,
                                           segment_flags,
                                           scan_items);

        // Exclusive scan of values and segment_flags
        size_value_pair_t  tile_aggregate;
        TilePrefixCallback prefix_op(tile_state, storage.scan_storage.prefix, scan_op, tile_idx);
        scan_tile(scan_items,
                  tile_aggregate,
                  prefix_op,
                  is_true<HAS_IDENTITY_ZERO>());
        size_value_pair_t tile_inclusive_prefix = prefix_op.GetInclusivePrefix();

        // Unzip values and segment indices
        zip_keys_and_values(pred_keys, segment_indices, scan_items, scatter_items);

        // Scatter flagged items
        scatter(scatter_items,
                segment_flags,
                segment_indices,
                tile_aggregate.key,
                prefix_op.GetExclusivePrefix().key);

        if (IS_LAST_TILE)
        {
          // Finalize the carry-out from the last tile
          finalize_last_tile(tile_inclusive_prefix.key,
                             num_remaining,
                             keys[ITEMS_PER_THREAD - 1],
                             tile_inclusive_prefix.value);
        }
      }
      template <bool IS_LAST_TILE>
      THRUST_DEVICE_FUNCTION void
      consume_tile(size_type      num_remaining,
                   int            tile_idx,
                   size_type      tile_offset,
                   ScanTileState &tile_state)
      {
        if (tile_idx == 0)
        {
          consume_first_tile<IS_LAST_TILE>(num_remaining,
                                           tile_offset,
                                           tile_state);
        }
        else
        {
          consume_subsequent_tile<IS_LAST_TILE>(num_remaining,
                                                tile_idx,
                                                tile_offset,
                                                tile_state);
        }
      }

      //---------------------------------------------------------------------
      // Constructor : consume_range
      //---------------------------------------------------------------------

      THRUST_DEVICE_FUNCTION impl(TempStorage &   storage_,
                                  KeysInputIt     keys_input_it_,
                                  ValuesInputIt   values_input_it_,
                                  KeysOutputIt    keys_output_it_,
                                  ValuesOutputIt  values_output_it_,
                                  NumRunsOutputIt num_runs_output_it_,
                                  EqualityOp      equality_op_,
                                  ReductionOp     reduction_op_,
                                  Size            num_items,
                                  int             /*num_tiles*/,
                                  ScanTileState & tile_state)
          : storage(storage_),
            keys_load_it(core::make_load_iterator(ActivePtxPlan(), keys_input_it_)),
            values_load_it(core::make_load_iterator(ActivePtxPlan(), values_input_it_)),
            keys_output_it(keys_output_it_),
            values_output_it(values_output_it_),
            num_runs_output_it(num_runs_output_it_),
            inequality_op(equality_op_),
            scan_op(reduction_op_)
      {
        // Blocks are launched in increasing order,
        // so just assign one tile per block
        //
        int  tile_idx          = blockIdx.x;
        Size tile_offset       = tile_idx * ITEMS_PER_TILE;
        Size num_remaining     = num_items - tile_offset;

        if (num_remaining > ITEMS_PER_TILE)
        {
          // Not the last tile (full)
          consume_tile<false>(num_remaining, tile_idx, tile_offset, tile_state);
        }
        else if (num_remaining > 0)
        {
          // The last tile (possibly partially-full)
          consume_tile<true>(num_remaining, tile_idx, tile_offset, tile_state);
        }
      }
    };    // struct impl

    //---------------------------------------------------------------------
    // Agent entry point
    //---------------------------------------------------------------------

    template <typename ActivePtxPlan>
    THRUST_AGENT_ENTRY(KeysInputIt     keys_input_it,
                       ValuesInputIt   values_input_it,
                       KeysOutputIt    keys_output_it,
                       ValuesOutputIt  values_output_it,
                       NumRunsOutputIt num_runs_output_it,
                       ScanTileState   tile_state,
                       EqualityOp      equality_op,
                       ReductionOp     reduction_op,
                       Size            num_items,
                       int             num_tiles,
                       char *          shmem)
    {
      using temp_storage_t = typename ActivePtxPlan::TempStorage;
      auto &storage        = *reinterpret_cast<temp_storage_t *>(shmem);

      impl<ActivePtxPlan>{storage,
                            keys_input_it,
                            values_input_it,
                            keys_output_it,
                            values_output_it,
                            num_runs_output_it,
                            equality_op,
                            reduction_op,
                            num_items,
                            num_tiles,
                            tile_state};
    }

  };    // struct ReduceByKeyAgent

  template <class ScanTileState,
            class Size,
            class NumSelectedIt>
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

  template <class KeysInputIt,
            class ValuesInputIt,
            class KeysOutputIt,
            class ValuesOutputIt,
            class NumRunsOutputIt,
            class EqualityOp,
            class ReductionOp,
            class Size>
  THRUST_RUNTIME_FUNCTION cudaError_t
  doit_step(void *          d_temp_storage,
            size_t &        temp_storage_bytes,
            KeysInputIt     keys_input_it,
            ValuesInputIt   values_input_it,
            KeysOutputIt    keys_output_it,
            ValuesOutputIt  values_output_it,
            NumRunsOutputIt num_runs_output_it,
            EqualityOp      equality_op,
            ReductionOp     reduction_op,
            Size            num_items,
            cudaStream_t    stream,
            bool            debug_sync)
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
    using rbk_agent_t = ReduceByKeyAgent<KeysInputIt,
                                         ValuesInputIt,
                                         KeysOutputIt,
                                         ValuesOutputIt,
                                         EqualityOp,
                                         ReductionOp,
                                         NumRunsOutputIt,
                                         Size>;

    using scan_tile_state_t = typename rbk_agent_t::ScanTileState;
    using init_agent_t = InitAgent<scan_tile_state_t, Size, NumRunsOutputIt>;

    // Create PtxPlans and AgentPlans:
    const auto init_ptx_plan = typename init_agent_t::PtxPlan{};
    const auto init_agent_plan = core::AgentPlan{init_ptx_plan};

    const auto rbk_agent_plan = core::AgentPlanFromTunings<rbk_agent_t>::get();

    // Number of input tiles
    const int  tile_size = rbk_agent_plan.items_per_tile;
    const Size num_tiles = cub::DivideAndRoundUp(num_items, tile_size);

    const std::size_t vshmem_size =
      core::vshmem_size(rbk_agent_plan.shared_memory_size, num_tiles);

    std::size_t allocation_sizes[2] = {9, vshmem_size};
    status = scan_tile_state_t::AllocationSize(static_cast<int>(num_tiles),
                                               allocation_sizes[0]);
    CUDA_CUB_RET_IF_FAIL(status);

    void *allocations[2] = {nullptr, nullptr};
    status = cub::AliasTemporaries(d_temp_storage,
                                   temp_storage_bytes,
                                   allocations,
                                   allocation_sizes);
    CUDA_CUB_RET_IF_FAIL(status);

    if (d_temp_storage == NULL)
    {
      return status;
    }

    scan_tile_state_t tile_state;
    status = tile_state.Init(static_cast<int>(num_tiles),
                             allocations[0],
                             allocation_sizes[0]);
    CUDA_CUB_RET_IF_FAIL(status);

    using init_agent_launcher_t = core::AgentLauncher<init_agent_t>;
    init_agent_launcher_t ia{init_agent_plan,
                             num_tiles,
                             stream,
                             "reduce_by_key::init_agent",
                             debug_sync};
    ia.launch_ptx_plan(init_ptx_plan, tile_state, num_tiles, num_runs_output_it);
    CUDA_CUB_RET_IF_FAIL(cudaPeekAtLastError());

    char *vshmem_ptr = vshmem_size > 0
                         ? reinterpret_cast<char *>(allocations[1])
                         : nullptr;

    using rbk_agent_launcher_t = core::AgentLauncher<rbk_agent_t>;
    rbk_agent_launcher_t rbka{rbk_agent_plan,
                              num_items,
                              stream,
                              vshmem_ptr,
                              "reduce_by_keys::reduce_by_key_agent",
                              debug_sync};
    rbka.launch_ptx_dispatch(typename rbk_agent_t::Tunings{},
                             keys_input_it,
                             values_input_it,
                             keys_output_it,
                             values_output_it,
                             num_runs_output_it,
                             tile_state,
                             equality_op,
                             reduction_op,
                             num_items,
                             num_tiles);
    CUDA_CUB_RET_IF_FAIL(cudaPeekAtLastError());

    return status;
  }

  template <typename Derived,
            typename KeysInputIt,
            typename ValuesInputIt,
            typename KeysOutputIt,
            typename ValuesOutputIt,
            typename EqualityOp,
            typename ReductionOp>
  THRUST_RUNTIME_FUNCTION
  pair<KeysOutputIt, ValuesOutputIt>
  reduce_by_key(execution_policy<Derived>& policy,
                KeysInputIt                keys_first,
                KeysInputIt                keys_last,
                ValuesInputIt              values_first,
                KeysOutputIt               keys_output,
                ValuesOutputIt             values_output,
                EqualityOp                 equality_op,
                ReductionOp                reduction_op)
  {
    typedef int size_type;

    size_type    num_items          = static_cast<size_type>(thrust::distance(keys_first, keys_last));
    size_t       temp_storage_bytes = 0;
    cudaStream_t stream             = cuda_cub::stream(policy);
    bool         debug_sync         = THRUST_DEBUG_SYNC_FLAG;

    if (num_items == 0)
      return thrust::make_pair(keys_output, values_output);

    cudaError_t status;
    status = doit_step(NULL,
                       temp_storage_bytes,
                       keys_first,
                       values_first,
                       keys_output,
                       values_output,
                       reinterpret_cast<size_type*>(NULL),
                       equality_op,
                       reduction_op,
                       num_items,
                       stream,
                       debug_sync);
    cuda_cub::throw_on_error(status, "reduce_by_key failed on 1st step");

    size_t allocation_sizes[2] = {sizeof(size_type), temp_storage_bytes};
    void * allocations[2]      = {NULL, NULL};

    size_t storage_size = 0;
    status = core::alias_storage(NULL,
                                 storage_size,
                                 allocations,
                                 allocation_sizes);
    cuda_cub::throw_on_error(status, "reduce failed on 1st alias_storage");

    // Allocate temporary storage.
    thrust::detail::temporary_array<thrust::detail::uint8_t, Derived>
      tmp(policy, storage_size);
    void *ptr = static_cast<void*>(tmp.data().get());

    status = core::alias_storage(ptr,
                                 storage_size,
                                 allocations,
                                 allocation_sizes);
    cuda_cub::throw_on_error(status, "reduce failed on 2nd alias_storage");

    size_type* d_num_runs_out
      = thrust::detail::aligned_reinterpret_cast<size_type*>(allocations[0]);

    status = doit_step(allocations[1],
                       temp_storage_bytes,
                       keys_first,
                       values_first,
                       keys_output,
                       values_output,
                       d_num_runs_out,
                       equality_op,
                       reduction_op,
                       num_items,
                       stream,
                       debug_sync);
    cuda_cub::throw_on_error(status, "reduce_by_key failed on 2nd step");

    status = cuda_cub::synchronize(policy);
    cuda_cub::throw_on_error(status, "reduce_by_key: failed to synchronize");

    int num_runs_out = cuda_cub::get_value(policy, d_num_runs_out);

    return thrust::make_pair(
      keys_output + num_runs_out,
      values_output + num_runs_out
    );
  }

}    // namespace __reduce_by_key

//-------------------------
// Thrust API entry points
//-------------------------

__thrust_exec_check_disable__
template <class Derived,
          class KeyInputIt,
          class ValInputIt,
          class KeyOutputIt,
          class ValOutputIt,
          class BinaryPred,
          class BinaryOp>
pair<KeyOutputIt, ValOutputIt> __host__ __device__
reduce_by_key(execution_policy<Derived> &policy,
              KeyInputIt                 keys_first,
              KeyInputIt                 keys_last,
              ValInputIt                 values_first,
              KeyOutputIt                keys_output,
              ValOutputIt                values_output,
              BinaryPred                 binary_pred,
              BinaryOp                   binary_op)
{
  pair<KeyOutputIt, ValOutputIt> ret = thrust::make_pair(keys_output, values_output);
  if (__THRUST_HAS_CUDART__)
  {
    ret = __reduce_by_key::reduce_by_key(policy,
                                         keys_first,
                                         keys_last,
                                         values_first,
                                         keys_output,
                                         values_output,
                                         binary_pred,
                                         binary_op);
  }
  else
  {
#if !__THRUST_HAS_CUDART__
    ret = thrust::reduce_by_key(cvt_to_seq(derived_cast(policy)),
                                keys_first,
                                keys_last,
                                values_first,
                                keys_output,
                                values_output,
                                binary_pred,
                                binary_op);
#endif
  }
  return ret;
}


template <class Derived,
          class KeyInputIt,
          class ValInputIt,
          class KeyOutputIt,
          class ValOutputIt,
          class BinaryPred>
pair<KeyOutputIt, ValOutputIt> __host__ __device__
reduce_by_key(execution_policy<Derived> &policy,
              KeyInputIt                 keys_first,
              KeyInputIt                 keys_last,
              ValInputIt                 values_first,
              KeyOutputIt                keys_output,
              ValOutputIt                values_output,
              BinaryPred                 binary_pred)
{
  typedef typename thrust::detail::eval_if<
    thrust::detail::is_output_iterator<ValOutputIt>::value,
    thrust::iterator_value<ValInputIt>,
    thrust::iterator_value<ValOutputIt>
  >::type value_type;
  return cuda_cub::reduce_by_key(policy,
                              keys_first,
                              keys_last,
                              values_first,
                              keys_output,
                              values_output,
                              binary_pred,
                              plus<value_type>());
}

template <class Derived,
          class KeyInputIt,
          class ValInputIt,
          class KeyOutputIt,
          class ValOutputIt>
pair<KeyOutputIt, ValOutputIt> __host__ __device__
reduce_by_key(execution_policy<Derived> &policy,
              KeyInputIt                 keys_first,
              KeyInputIt                 keys_last,
              ValInputIt                 values_first,
              KeyOutputIt                keys_output,
              ValOutputIt                values_output)
{
  typedef typename thrust::iterator_value<KeyInputIt>::type KeyT;
  return cuda_cub::reduce_by_key(policy,
                              keys_first,
                              keys_last,
                              values_first,
                              keys_output,
                              values_output,
                              equal_to<KeyT>());
}

} // namespace cuda_

} // end namespace thrust

#include <thrust/memory.h>
#include <thrust/reduce.h>

#endif
