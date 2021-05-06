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
#include <thrust/system/cuda/config.h>
#include <thrust/system/cuda/detail/core/agent_launcher.h>
#include <thrust/system/cuda/detail/core/util.h>
#include <thrust/system/cuda/detail/execution_policy.h>
#include <thrust/system/cuda/detail/par_to_seq.h>
#include <thrust/detail/trivial_sequence.h>
#include <thrust/detail/integer_math.h>
#include <thrust/extrema.h>
#include <thrust/sort.h>
#include <thrust/distance.h>
#include <thrust/sequence.h>
#include <thrust/detail/alignment.h>
#include <thrust/type_traits/is_contiguous_iterator.h>

#include <cub/detail/ptx_dispatch.cuh>
#include <cub/device/device_radix_sort.cuh>

namespace thrust
{
namespace cuda_cub {

namespace __merge_sort {

  template <class KeysIt1,
            class KeysIt2,
            class Size,
            class BinaryPred>
  THRUST_DEVICE_FUNCTION Size
  merge_path(KeysIt1    keys1,
             KeysIt2    keys2,
             Size       keys1_count,
             Size       keys2_count,
             Size       diag,
             BinaryPred binary_pred)
  {
    typedef typename iterator_traits<KeysIt1>::value_type key1_type;
    typedef typename iterator_traits<KeysIt2>::value_type key2_type;

    Size keys1_begin = thrust::max<Size>(0, diag - keys2_count);
    Size keys1_end   = thrust::min<Size>(diag, keys1_count);

    while (keys1_begin < keys1_end)
    {
      Size      mid  = (keys1_begin + keys1_end) >> 1;
      key1_type key1 = keys1[mid];
      key2_type key2 = keys2[diag - 1 - mid];
      bool      pred = binary_pred(key2, key1);
      if (pred)
      {
        keys1_end = mid;
      }
      else
      {
        keys1_begin = mid + 1;
      }
    }
    return keys1_begin;
  }

  template <class It, class T2, class CompareOp, int ITEMS_PER_THREAD>
  THRUST_DEVICE_FUNCTION void
  serial_merge(It  keys_shared,
               int keys1_beg,
               int keys2_beg,
               int keys1_count,
               int keys2_count,
               T2 (&output)[ITEMS_PER_THREAD],
               int (&indices)[ITEMS_PER_THREAD],
               CompareOp compare_op)
  {
    int keys1_end = keys1_beg + keys1_count;
    int keys2_end = keys2_beg + keys2_count;

    typedef typename iterator_value<It>::type key_type;

    key_type key1 = keys_shared[keys1_beg];
    key_type key2 = keys_shared[keys2_beg];


#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
    {
      bool p = (keys2_beg < keys2_end) &&
               ((keys1_beg >= keys1_end) ||
                compare_op(key2,key1));

      output[ITEM]  = p ? key2 : key1;
      indices[ITEM] = p ? keys2_beg++ : keys1_beg++;

      if (p)
      {
        key2 = keys_shared[keys2_beg];
      }
      else
      {
        key1 = keys_shared[keys1_beg];
      }
    }
  }

  template <int                      _BLOCK_THREADS,
            int                      _ITEMS_PER_THREAD = 1,
            cub::BlockLoadAlgorithm  _LOAD_ALGORITHM   = cub::BLOCK_LOAD_DIRECT,
            cub::CacheLoadModifier   _LOAD_MODIFIER    = cub::LOAD_LDG,
            cub::BlockStoreAlgorithm _STORE_ALGORITHM  = cub::BLOCK_STORE_DIRECT>
  struct PtxPolicy
  {
    static constexpr int BLOCK_THREADS    = _BLOCK_THREADS;
    static constexpr int ITEMS_PER_THREAD = _ITEMS_PER_THREAD;
    static constexpr int ITEMS_PER_TILE   = BLOCK_THREADS * ITEMS_PER_THREAD;

    static constexpr cub::BlockLoadAlgorithm LOAD_ALGORITHM = _LOAD_ALGORITHM;
    static constexpr cub::CacheLoadModifier LOAD_MODIFIER   = _LOAD_MODIFIER;
    static constexpr cub::BlockStoreAlgorithm STORE_ALGORITHM =
      _STORE_ALGORITHM;
  }; // PtxPolicy

  template <typename T>
  struct Tuning350 : cub::detail::ptx_base<350>
  {
    static constexpr int INPUT_SIZE = static_cast<int>(sizeof(T));

    static constexpr int NOMINAL_4B_ITEMS_PER_THREAD = 11;
    static constexpr int ITEMS_PER_THREAD =
      CUB_MIN(NOMINAL_4B_ITEMS_PER_THREAD,
              CUB_MAX(1, (NOMINAL_4B_ITEMS_PER_THREAD * 4 / INPUT_SIZE)));

    using Policy = PtxPolicy<256,
                             ITEMS_PER_THREAD,
                             cub::BLOCK_LOAD_WARP_TRANSPOSE,
                             cub::LOAD_LDG,
                             cub::BLOCK_STORE_WARP_TRANSPOSE>;
  }; // Tuning350

  template <typename T>
  struct Tuning520 : cub::detail::ptx_base<520>
  {
    static constexpr int INPUT_SIZE = static_cast<int>(sizeof(T));

    static constexpr int NOMINAL_4B_ITEMS_PER_THREAD = 15;
    static constexpr int ITEMS_PER_THREAD =
      CUB_MIN(NOMINAL_4B_ITEMS_PER_THREAD,
              CUB_MAX(1, (NOMINAL_4B_ITEMS_PER_THREAD * 4 / INPUT_SIZE)));

    using Policy = PtxPolicy<512,
                             ITEMS_PER_THREAD,
                             cub::BLOCK_LOAD_WARP_TRANSPOSE,
                             cub::LOAD_LDG,
                             cub::BLOCK_STORE_WARP_TRANSPOSE>;
  }; // Tuning520

  template <typename T>
  struct Tuning600 : cub::detail::ptx_base<600>
  {
    const static int INPUT_SIZE = static_cast<int>(sizeof(T));

    static constexpr int NOMINAL_4B_ITEMS_PER_THREAD = 17;
    static constexpr int ITEMS_PER_THREAD =
      CUB_MIN(NOMINAL_4B_ITEMS_PER_THREAD,
              CUB_MAX(1, (NOMINAL_4B_ITEMS_PER_THREAD * 4 / sizeof(T))));

    using Policy = PtxPolicy<256,
                             ITEMS_PER_THREAD,
                             cub::BLOCK_LOAD_WARP_TRANSPOSE,
                             cub::LOAD_DEFAULT,
                             cub::BLOCK_STORE_WARP_TRANSPOSE>;
  };

  template <class KeysIt,
            class ItemsIt,
            class Size,
            class CompareOp,
            class SORT_ITEMS,
            class STABLE>
  struct BlockSortAgent
  {
    typedef typename iterator_traits<KeysIt>::value_type key_type;
    typedef typename iterator_traits<ItemsIt>::value_type item_type;

    // List tunings in reverse order:
    using Tunings = cub::detail::type_list<Tuning600<key_type>,
                                           Tuning520<key_type>,
                                           Tuning350<key_type>>;

    template <typename Tuning>
    struct PtxPlan : Tuning::Policy
    {
      using KeysLoadIt  = typename core::LoadIterator<PtxPlan, KeysIt>::type;
      using ItemsLoadIt = typename core::LoadIterator<PtxPlan, ItemsIt>::type;

      using BlockLoadKeys = typename core::BlockLoad<PtxPlan, KeysLoadIt>::type;
      using BlockLoadItems =
        typename core::BlockLoad<PtxPlan, ItemsLoadIt>::type;

      using BlockStoreKeysIt = typename core::BlockStore<PtxPlan, KeysIt>::type;
      using BlockStoreItemsIt =
        typename core::BlockStore<PtxPlan, ItemsIt>::type;
      using BlockStoreKeysRaw =
        typename core::BlockStore<PtxPlan, key_type *>::type;
      using BlockStoreItemsRaw =
        typename core::BlockStore<PtxPlan, item_type *>::type;

      union TempStorage
      {
        typename BlockLoadKeys::TempStorage   load_keys;
        typename BlockLoadItems::TempStorage  load_items;
        typename BlockStoreKeysIt::TempStorage  store_keys_it;
        typename BlockStoreItemsIt::TempStorage store_items_it;
        typename BlockStoreKeysRaw::TempStorage  store_keys_raw;
        typename BlockStoreItemsRaw::TempStorage store_items_raw;

        core::uninitialized_array<key_type, PtxPlan::ITEMS_PER_TILE + 1>  keys_shared;
        core::uninitialized_array<item_type, PtxPlan::ITEMS_PER_TILE + 1> items_shared;
      };    // union TempStorage
    };      // struct PtxPlan

    template <typename ActivePtxPlan>
    struct impl
    {
      using KeysLoadIt         = typename ActivePtxPlan::KeysLoadIt;
      using ItemsLoadIt        = typename ActivePtxPlan::ItemsLoadIt;
      using BlockLoadKeys      = typename ActivePtxPlan::BlockLoadKeys;
      using BlockLoadItems     = typename ActivePtxPlan::BlockLoadItems;
      using BlockStoreKeysIt   = typename ActivePtxPlan::BlockStoreKeysIt;
      using BlockStoreItemsIt  = typename ActivePtxPlan::BlockStoreItemsIt;
      using BlockStoreKeysRaw  = typename ActivePtxPlan::BlockStoreKeysRaw;
      using BlockStoreItemsRaw = typename ActivePtxPlan::BlockStoreItemsRaw;
      using TempStorage        = typename ActivePtxPlan::TempStorage;

      static constexpr int ITEMS_PER_THREAD = ActivePtxPlan::ITEMS_PER_THREAD;
      static constexpr int BLOCK_THREADS    = ActivePtxPlan::BLOCK_THREADS;
      static constexpr int ITEMS_PER_TILE   = ActivePtxPlan::ITEMS_PER_TILE;

      //---------------------------------------------------------------------
      // Per thread data
      //---------------------------------------------------------------------

      bool         ping;
      TempStorage& storage;
      KeysLoadIt   keys_in;
      ItemsLoadIt  items_in;
      Size         keys_count;
      KeysIt       keys_out_it;
      ItemsIt      items_out_it;
      key_type*    keys_out_raw;
      item_type*   items_out_raw;
      CompareOp    compare_op;

      //---------------------------------------------------------------------
      // Serial stable sort network
      //---------------------------------------------------------------------

      THRUST_DEVICE_FUNCTION
      void stable_odd_even_sort(key_type (&keys)[ITEMS_PER_THREAD],
                                item_type (&items)[ITEMS_PER_THREAD])
      {
#pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; ++i)
        {
#pragma unroll
          for (int j = 1 & i; j < ITEMS_PER_THREAD - 1; j += 2)
          {
            if (compare_op(keys[j + 1], keys[j]))
            {
              using thrust::swap;
              swap(keys[j], keys[j + 1]);
              if (SORT_ITEMS::value)
              {
                swap(items[j], items[j + 1]);
              }
            }
          }    // inner loop
        }      // outer loop
      }

      //---------------------------------------------------------------------
      // Parallel thread block merge sort
      //---------------------------------------------------------------------

      template <bool IS_LAST_TILE>
      THRUST_DEVICE_FUNCTION void
      block_mergesort(int tid,
                      int count,
                      key_type (&keys_loc)[ITEMS_PER_THREAD],
                      item_type (&items_loc)[ITEMS_PER_THREAD])
      {
        using core::uninitialized_array;
        using core::sync_threadblock;

        // if first element of thread is in input range, stable sort items
        //
        if (!IS_LAST_TILE || ITEMS_PER_THREAD * tid < count)
        {
          stable_odd_even_sort(keys_loc, items_loc);
        }

        // each thread has  sorted keys_loc
        // merge sort keys_loc in shared memory
        //
#pragma unroll
        for (int coop = 2; coop <= BLOCK_THREADS; coop *= 2)
        {
          sync_threadblock();

          // store keys in shmem
          //
#pragma unroll
          for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
          {
            int idx                  = ITEMS_PER_THREAD * threadIdx.x + ITEM;
            storage.keys_shared[idx] = keys_loc[ITEM];
          }

          sync_threadblock();

          int  indices[ITEMS_PER_THREAD];

          int list  = ~(coop - 1) & tid;
          int start = ITEMS_PER_THREAD * list;
          int size  = ITEMS_PER_THREAD * (coop >> 1);

          int diag = min(count,
                         ITEMS_PER_THREAD * ((coop - 1) & tid));

          int keys1_beg = min(count, start);
          int keys1_end = min(count, keys1_beg + size);
          int keys2_beg = keys1_end;
          int keys2_end = min(count, keys2_beg + size);

          int keys1_count = keys1_end - keys1_beg;
          int keys2_count = keys2_end - keys2_beg;

          int partition_diag = merge_path(&storage.keys_shared[keys1_beg],
                                          &storage.keys_shared[keys2_beg],
                                          keys1_count,
                                          keys2_count,
                                          diag,
                                          compare_op);

          int keys1_beg_loc   = keys1_beg + partition_diag;
          int keys1_end_loc   = keys1_end;
          int keys2_beg_loc   = keys2_beg + diag - partition_diag;
          int keys2_end_loc   = keys2_end;
          int keys1_count_loc = keys1_end_loc - keys1_beg_loc;
          int keys2_count_loc = keys2_end_loc - keys2_beg_loc;
          serial_merge(&storage.keys_shared[0],
                       keys1_beg_loc,
                       keys2_beg_loc,
                       keys1_count_loc,
                       keys2_count_loc,
                       keys_loc,
                       indices,
                       compare_op);


          if (SORT_ITEMS::value)
          {
            sync_threadblock();

            // store keys in shmem
            //
#pragma unroll
            for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
            {
              int idx                   = ITEMS_PER_THREAD * threadIdx.x + ITEM;
              storage.items_shared[idx] = items_loc[ITEM];
            }

            sync_threadblock();

            // gather items from shmem
            //
#pragma unroll
            for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
            {
              items_loc[ITEM] = storage.items_shared[indices[ITEM]];
            }
          }
        }
      }    // func block_merge_sort

      //---------------------------------------------------------------------
      // Tile processing
      //---------------------------------------------------------------------

      template <bool IS_LAST_TILE>
      THRUST_DEVICE_FUNCTION void
      consume_tile(int  tid,
                   Size /*tile_idx*/,
                   Size tile_base,
                   int  num_remaining)
      {
        using core::uninitialized_array;
        using core::sync_threadblock;

        item_type items_loc[ITEMS_PER_THREAD];
        if (SORT_ITEMS::value)
        {
          BlockLoadItems(storage.load_items)
              .Load(items_in + tile_base,
                    items_loc,
                    num_remaining,
                    *(items_in + tile_base));

          sync_threadblock();
        }

        key_type keys_loc[ITEMS_PER_THREAD];
        if (IS_LAST_TILE)
        {
          BlockLoadKeys(storage.load_keys)
              .Load(keys_in + tile_base,
                    keys_loc,
                    num_remaining,
                    *(keys_in + tile_base));
        }
        else
        {
          BlockLoadKeys(storage.load_keys)
              .Load(keys_in + tile_base, keys_loc);
        }

        if (IS_LAST_TILE)
        {
          // if last tile, find valid max_key
          // and fill the remainig keys with it
          //
          key_type max_key = keys_loc[0];
#pragma unroll
          for (int ITEM = 1; ITEM < ITEMS_PER_THREAD; ++ITEM)
          {
            if (ITEMS_PER_THREAD * tid + ITEM < num_remaining)
            {
              max_key = compare_op(max_key, keys_loc[ITEM])
                            ? keys_loc[ITEM]
                            : max_key;
            }
            else
            {
              keys_loc[ITEM] = max_key;
            }
          }
        }

        sync_threadblock();

        if (IS_LAST_TILE)
        {
          block_mergesort<IS_LAST_TILE>(tid,
                                        num_remaining,
                                        keys_loc,
                                        items_loc);
        }
        else
        {
          block_mergesort<IS_LAST_TILE>(tid,
                                        ITEMS_PER_TILE,
                                        keys_loc,
                                        items_loc);
        }

        sync_threadblock();

        if (ping)
        {
          if (IS_LAST_TILE)
          {
            BlockStoreKeysIt(storage.store_keys_it)
                .Store(keys_out_it + tile_base, keys_loc, num_remaining);
          }
          else
          {
            BlockStoreKeysIt(storage.store_keys_it)
                .Store(keys_out_it + tile_base, keys_loc);
          }

          if (SORT_ITEMS::value)
          {
            sync_threadblock();

            BlockStoreItemsIt(storage.store_items_it)
                .Store(items_out_it + tile_base, items_loc, num_remaining);
          }
        }
        else
        {
          if (IS_LAST_TILE)
          {
            BlockStoreKeysRaw(storage.store_keys_raw)
                .Store(keys_out_raw + tile_base, keys_loc, num_remaining);
          }
          else
          {
            BlockStoreKeysRaw(storage.store_keys_raw)
                .Store(keys_out_raw + tile_base, keys_loc);
          }

          if (SORT_ITEMS::value)
          {
            sync_threadblock();

            BlockStoreItemsRaw(storage.store_items_raw)
                .Store(items_out_raw + tile_base, items_loc, num_remaining);
          }
        }
      }

      //---------------------------------------------------------------------
      // Constructor
      //---------------------------------------------------------------------

      THRUST_DEVICE_FUNCTION
      impl(bool         ping_,
           TempStorage& storage_,
           KeysLoadIt   keys_in_,
           ItemsLoadIt  items_in_,
           Size         keys_count_,
           KeysIt       keys_out_it_,
           ItemsIt      items_out_it_,
           key_type*    keys_out_raw_,
           item_type*   items_out_raw_,
           CompareOp    compare_op_)
          : ping(ping_),
            storage(storage_),
            keys_in(keys_in_),
            items_in(items_in_),
            keys_count(keys_count_),
            keys_out_it(keys_out_it_),
            items_out_it(items_out_it_),
            keys_out_raw(keys_out_raw_),
            items_out_raw(items_out_raw_),
            compare_op(compare_op_)
      {
        int  tid           = threadIdx.x;
        Size tile_idx      = blockIdx.x;
        Size num_tiles     = gridDim.x;
        Size tile_base     = tile_idx * ITEMS_PER_TILE;

        // ITEMS_PER_TILE is host static constexpr variable, and min(...) takes
        // a const reference. We cannot take the address of a static host var
        // from device code. Use a local instead.
        constexpr auto local_items_per_tile = ITEMS_PER_TILE;
        int items_in_tile = min<int>(keys_count - tile_base,
                                     local_items_per_tile);

        if (tile_idx < num_tiles - 1)
        {
          consume_tile<false>(tid, tile_idx, tile_base, ITEMS_PER_TILE);
        }
        else
        {
          consume_tile<true>(tid, tile_idx, tile_base, items_in_tile);
        }
      }
    };    // struct impl

    //---------------------------------------------------------------------
    // Agent entry point
    //---------------------------------------------------------------------

    template <typename ActivePtxPlan>
    THRUST_AGENT_ENTRY(bool       ping,
                       KeysIt     keys_inout,
                       ItemsIt    items_inout,
                       Size       keys_count,
                       key_type*  keys_out,
                       item_type* items_out,
                       CompareOp  compare_op,
                       char*      shmem)
    {
      using temp_storage_t = typename ActivePtxPlan::TempStorage;
      auto &storage        = *reinterpret_cast<temp_storage_t *>(shmem);

      impl<ActivePtxPlan>{ping,
                          storage,
                          core::make_load_iterator(ActivePtxPlan(), keys_inout),
                          core::make_load_iterator(ActivePtxPlan(),
                                                   items_inout),
                          keys_count,
                          keys_inout,
                          items_inout,
                          keys_out,
                          items_out,
                          compare_op};
    }
  };    // struct BlockSortAgent

  template <class KeysIt,
            class Size,
            class CompareOp>
  struct PartitionAgent
  {
    using key_type = typename iterator_traits<KeysIt>::value_type;
    struct PtxPlan : PtxPolicy<256> {};

    //---------------------------------------------------------------------
    // Agent entry point
    //---------------------------------------------------------------------

    template <typename /*ActivePtxPlan*/>
    THRUST_AGENT_ENTRY(bool      ping,
                       KeysIt    keys_ping,
                       key_type* keys_pong,
                       Size      keys_count,
                       Size      num_partitions,
                       Size*     merge_partitions,
                       CompareOp compare_op,
                       Size      coop,
                       int       items_per_tile,
                       char*     /*shmem*/)
    {
      Size partition_idx = blockDim.x * blockIdx.x + threadIdx.x;
      if (partition_idx < num_partitions)
      {
        Size list  = ~(coop - 1) & partition_idx;
        Size start = items_per_tile * list;
        Size size  = items_per_tile * (coop >> 1);

        Size keys1_beg = min(keys_count, start);
        Size keys1_end = min(keys_count, start + size);
        Size keys2_beg = keys1_end;
        Size keys2_end = min(keys_count, keys2_beg + size);


        Size partition_at = min(keys2_end - keys1_beg,
                                items_per_tile * ((coop - 1) & partition_idx));

        Size partition_diag = ping ? merge_path(keys_ping + keys1_beg,
                                                keys_ping + keys2_beg,
                                                keys1_end - keys1_beg,
                                                keys2_end - keys2_beg,
                                                partition_at,
                                                compare_op)
                                   : merge_path(keys_pong + keys1_beg,
                                                keys_pong + keys2_beg,
                                                keys1_end - keys1_beg,
                                                keys2_end - keys2_beg,
                                                partition_at,
                                                compare_op);


        merge_partitions[partition_idx] = keys1_beg + partition_diag;
      }
    }
  };    // struct PartitionAgent

  template <class KeysIt,
            class ItemsIt,
            class Size,
            class CompareOp,
            class MERGE_ITEMS>
  struct MergeAgent
  {
    using key_type  = typename iterator_traits<KeysIt>::value_type;
    using item_type = typename iterator_traits<ItemsIt>::value_type;

    using KeysOutputPongIt  = KeysIt;
    using ItemsOutputPongIt = ItemsIt;
    using KeysOutputPingIt  = key_type *;
    using ItemsOutputPingIt = item_type *;

    // List tunings in reverse order:
    using Tunings = cub::detail::type_list<Tuning600<key_type>,
                                           Tuning520<key_type>,
                                           Tuning350<key_type>>;

    template <typename Tuning>
    struct PtxPlan : Tuning::Policy
    {
      using KeysLoadPingIt = typename core::LoadIterator<PtxPlan, KeysIt>::type;
      using ItemsLoadPingIt =
        typename core::LoadIterator<PtxPlan, ItemsIt>::type;
      using KeysLoadPongIt =
        typename core::LoadIterator<PtxPlan, key_type *>::type;
      using ItemsLoadPongIt =
        typename core::LoadIterator<PtxPlan, item_type *>::type;

      using BlockLoadKeysPing =
        typename core::BlockLoad<PtxPlan, KeysLoadPingIt>::type;
      using BlockLoadItemsPing =
        typename core::BlockLoad<PtxPlan, ItemsLoadPingIt>::type;
      using BlockLoadKeysPong =
        typename core::BlockLoad<PtxPlan, KeysLoadPongIt>::type;
      using BlockLoadItemsPong =
        typename core::BlockLoad<PtxPlan, ItemsLoadPongIt>::type;

      using BlockStoreKeysPong =
        typename core::BlockStore<PtxPlan, KeysOutputPongIt>::type;
      using BlockStoreItemsPong =
        typename core::BlockStore<PtxPlan, ItemsOutputPongIt>::type;
      using BlockStoreKeysPing =
        typename core::BlockStore<PtxPlan, KeysOutputPingIt>::type;
      using BlockStoreItemsPing =
        typename core::BlockStore<PtxPlan, ItemsOutputPingIt>::type;

      // gather required temporary storage in a union
      //
      union TempStorage
      {
        typename BlockLoadKeysPing::TempStorage  load_keys_ping;
        typename BlockLoadItemsPing::TempStorage load_items_ping;
        typename BlockLoadKeysPong::TempStorage  load_keys_pong;
        typename BlockLoadItemsPong::TempStorage load_items_pong;

        typename BlockStoreKeysPing::TempStorage  store_keys_ping;
        typename BlockStoreItemsPing::TempStorage store_items_ping;
        typename BlockStoreKeysPong::TempStorage  store_keys_pong;
        typename BlockStoreItemsPong::TempStorage store_items_pong;

        core::uninitialized_array<key_type, PtxPlan::ITEMS_PER_TILE + 1>  keys_shared;
        core::uninitialized_array<item_type, PtxPlan::ITEMS_PER_TILE + 1> items_shared;
      };    // union TempStorage
    };    // struct PtxPlan

    template <typename ActivePtxPlan>
    struct impl
    {
      using KeysLoadPingIt  = typename ActivePtxPlan::KeysLoadPingIt;
      using ItemsLoadPingIt = typename ActivePtxPlan::ItemsLoadPingIt;
      using KeysLoadPongIt  = typename ActivePtxPlan::KeysLoadPongIt;
      using ItemsLoadPongIt = typename ActivePtxPlan::ItemsLoadPongIt;

      using BlockLoadKeysPing  = typename ActivePtxPlan::BlockLoadKeysPing;
      using BlockLoadItemsPing = typename ActivePtxPlan::BlockLoadItemsPing;
      using BlockLoadKeysPong  = typename ActivePtxPlan::BlockLoadKeysPong;
      using BlockLoadItemsPong = typename ActivePtxPlan::BlockLoadItemsPong;

      using BlockStoreKeysPing  = typename ActivePtxPlan::BlockStoreKeysPing;
      using BlockStoreItemsPing = typename ActivePtxPlan::BlockStoreItemsPing;
      using BlockStoreKeysPong  = typename ActivePtxPlan::BlockStoreKeysPong;
      using BlockStoreItemsPong = typename ActivePtxPlan::BlockStoreItemsPong;

      using TempStorage = typename ActivePtxPlan::TempStorage;

      static constexpr int ITEMS_PER_THREAD = ActivePtxPlan::ITEMS_PER_THREAD;
      static constexpr int BLOCK_THREADS    = ActivePtxPlan::BLOCK_THREADS;
      static constexpr int ITEMS_PER_TILE   = ActivePtxPlan::ITEMS_PER_TILE;

      //---------------------------------------------------------------------
      // Per thread data
      //---------------------------------------------------------------------

      bool            ping;
      TempStorage&    storage;

      KeysLoadPingIt  keys_in_ping;
      ItemsLoadPingIt items_in_ping;
      KeysLoadPongIt  keys_in_pong;
      ItemsLoadPongIt items_in_pong;

      Size            keys_count;

      KeysOutputPongIt  keys_out_pong;
      ItemsOutputPongIt items_out_pong;
      KeysOutputPingIt  keys_out_ping;
      ItemsOutputPingIt items_out_ping;

      CompareOp       compare_op;
      Size*           merge_partitions;
      Size            coop;

      //---------------------------------------------------------------------
      // Utility functions
      //---------------------------------------------------------------------

      template <bool IS_FULL_TILE, class T, class It1, class It2>
      THRUST_DEVICE_FUNCTION void
      gmem_to_reg(T (&output)[ITEMS_PER_THREAD],
                  It1 input1,
                  It2 input2,
                  int count1,
                  int count2)
      {
        if (IS_FULL_TILE)
        {
#pragma unroll
          for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
          {
            int idx = BLOCK_THREADS * ITEM + threadIdx.x;
            output[ITEM] = (idx < count1) ? input1[idx] : input2[idx - count1];
          }
        }
        else
        {
#pragma unroll
          for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
          {
            int idx = BLOCK_THREADS * ITEM + threadIdx.x;
            if (idx < count1 + count2)
            {
              output[ITEM] = (idx < count1) ? input1[idx] : input2[idx - count1];
            }
          }
        }
      }

      template <class T, class It>
      THRUST_DEVICE_FUNCTION void
      reg_to_shared(It output,
                    T (&input)[ITEMS_PER_THREAD])
      {
#pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
          int idx = BLOCK_THREADS * ITEM + threadIdx.x;
          output[idx] = input[ITEM];
        }
      }

      //---------------------------------------------------------------------
      // Tile processing
      //---------------------------------------------------------------------

      template <bool IS_FULL_TILE>
      THRUST_DEVICE_FUNCTION void
      consume_tile(int  tid,
                   Size tile_idx,
                   Size tile_base,
                   int  count)
      {
        using core::sync_threadblock;
        using core::uninitialized_array;

        Size partition_beg = merge_partitions[tile_idx + 0];
        Size partition_end = merge_partitions[tile_idx + 1];

        Size list = ~(coop - 1) & tile_idx;
        Size start = ITEMS_PER_TILE * list;
        Size size  = ITEMS_PER_TILE * (coop >> 1);

        Size diag   = ITEMS_PER_TILE * tile_idx - start;

        Size keys1_beg = partition_beg;
        Size keys1_end = partition_end;
        Size keys2_beg = min<Size>(keys_count, 2 * start + size + diag - partition_beg);
        Size keys2_end = min<Size>(keys_count, 2 * start + size + diag + ITEMS_PER_TILE - partition_end);

        if (coop - 1 == ((coop - 1) & tile_idx))
        {
          keys1_end = min(keys_count, start + size);
          keys2_end = min(keys_count, start + size * 2);
        }

        // number of keys per tile
        //
        int num_keys1 = static_cast<int>(keys1_end - keys1_beg);
        int num_keys2 = static_cast<int>(keys2_end - keys2_beg);

        // load keys1 & keys2
        key_type keys_loc[ITEMS_PER_THREAD];
        if (ping)
        {
          gmem_to_reg<IS_FULL_TILE>(keys_loc,
                                    keys_in_ping + keys1_beg,
                                    keys_in_ping + keys2_beg,
                                    num_keys1,
                                    num_keys2);
        }
        else
        {
          gmem_to_reg<IS_FULL_TILE>(keys_loc,
                                    keys_in_pong + keys1_beg,
                                    keys_in_pong + keys2_beg,
                                    num_keys1,
                                    num_keys2);
        }
        reg_to_shared(&storage.keys_shared[0], keys_loc);

        // preload items into registers already
        //
        item_type items_loc[ITEMS_PER_THREAD];
        if (MERGE_ITEMS::value)
        {
          if (ping)
          {
            gmem_to_reg<IS_FULL_TILE>(items_loc,
                                      items_in_ping + keys1_beg,
                                      items_in_ping + keys2_beg,
                                      num_keys1,
                                      num_keys2);
          }
          else
          {
            gmem_to_reg<IS_FULL_TILE>(items_loc,
                                      items_in_pong + keys1_beg,
                                      items_in_pong + keys2_beg,
                                      num_keys1,
                                      num_keys2);
          }
        }

        sync_threadblock();

        // use binary search in shared memory
        // to find merge path for each of thread
        // we can use int type here, because the number of
        // items in shared memory is limited
        //
        int diag0_loc = min<Size>(num_keys1 + num_keys2,
                                  ITEMS_PER_THREAD * tid);

        int keys1_beg_loc = merge_path(&storage.keys_shared[0],
                                       &storage.keys_shared[num_keys1],
                                       num_keys1,
                                       num_keys2,
                                       diag0_loc,
                                       compare_op);
        int keys1_end_loc = num_keys1;
        int keys2_beg_loc = diag0_loc - keys1_beg_loc;
        int keys2_end_loc = num_keys2;

        int num_keys1_loc = keys1_end_loc - keys1_beg_loc;
        int num_keys2_loc = keys2_end_loc - keys2_beg_loc;

        // perform serial merge
        //
        int indices[ITEMS_PER_THREAD];

        serial_merge(&storage.keys_shared[0],
                     keys1_beg_loc,
                     keys2_beg_loc + num_keys1,
                     num_keys1_loc,
                     num_keys2_loc,
                     keys_loc,
                     indices,
                     compare_op);

        sync_threadblock();

        // write keys
        //
        if (ping)
        {
          if (IS_FULL_TILE)
          {
            BlockStoreKeysPing(storage.store_keys_ping)
                .Store(keys_out_ping + tile_base, keys_loc);
          }
          else
          {
            BlockStoreKeysPing(storage.store_keys_ping)
                .Store(keys_out_ping + tile_base, keys_loc, num_keys1 + num_keys2);
          }
        }
        else
        {
          if (IS_FULL_TILE)
          {
            BlockStoreKeysPong(storage.store_keys_pong)
                .Store(keys_out_pong + tile_base, keys_loc);
          }
          else
          {
            BlockStoreKeysPong(storage.store_keys_pong)
                .Store(keys_out_pong + tile_base, keys_loc, num_keys1 + num_keys2);
          }
        }

        // if items are provided, merge them
        if (MERGE_ITEMS::value)
        {
          sync_threadblock();

          reg_to_shared(&storage.items_shared[0], items_loc);

          sync_threadblock();

          // gather items from shared mem
          //
#pragma unroll
          for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
          {
            items_loc[ITEM] = storage.items_shared[indices[ITEM]];
          }

          sync_threadblock();

          // write from reg to gmem
          //
          if (ping)
          {
            if (IS_FULL_TILE)
            {
              BlockStoreItemsPing(storage.store_items_ping)
                  .Store(items_out_ping + tile_base, items_loc);
            }
            else
            {
              BlockStoreItemsPing(storage.store_items_ping)
                  .Store(items_out_ping + tile_base, items_loc, count);
            }
          }
          else
          {
            if (IS_FULL_TILE)
            {
              BlockStoreItemsPong(storage.store_items_pong)
                  .Store(items_out_pong + tile_base, items_loc);
            }
            else
            {
              BlockStoreItemsPong(storage.store_items_pong)
                  .Store(items_out_pong + tile_base, items_loc, count);
            }
          }
        }
      }

      //---------------------------------------------------------------------
      // Constructor
      //---------------------------------------------------------------------

      THRUST_DEVICE_FUNCTION
      impl(bool              ping_,
           TempStorage&      storage_,
           KeysLoadPingIt    keys_in_ping_,
           ItemsLoadPingIt   items_in_ping_,
           KeysLoadPongIt    keys_in_pong_,
           ItemsLoadPongIt   items_in_pong_,
           Size              keys_count_,
           KeysOutputPingIt  keys_out_ping_,
           ItemsOutputPingIt items_out_ping_,
           KeysOutputPongIt  keys_out_pong_,
           ItemsOutputPongIt items_out_pong_,
           CompareOp         compare_op_,
           Size*             merge_partitions_,
           Size              coop_)
          : ping(ping_),
            storage(storage_),
            keys_in_ping(keys_in_ping_),
            items_in_ping(items_in_ping_),
            keys_in_pong(keys_in_pong_),
            items_in_pong(items_in_pong_),
            keys_count(keys_count_),
            keys_out_pong(keys_out_pong_),
            items_out_pong(items_out_pong_),
            keys_out_ping(keys_out_ping_),
            items_out_ping(items_out_ping_),
            compare_op(compare_op_),
            merge_partitions(merge_partitions_),
            coop(coop_)
      {
        // XXX with 8.5 chaging type to Size (or long long) results in error!
        int  tile_idx      = blockIdx.x;
        Size num_tiles     = gridDim.x;
        Size tile_base     = Size(tile_idx) * ITEMS_PER_TILE;
        int tid           = threadIdx.x;
        int items_in_tile = static_cast<int>(min((Size)ITEMS_PER_TILE,
                                                 keys_count - tile_base));
        if (tile_idx < num_tiles - 1)
        {
          consume_tile<true>(tid,
                             tile_idx,
                             tile_base,
                             ITEMS_PER_TILE);
        }
        else
        {
          consume_tile<false>(tid,
                              tile_idx,
                              tile_base,
                              items_in_tile);
        }
      }
    };    // struct impl

    //---------------------------------------------------------------------
    // Agent entry point
    //---------------------------------------------------------------------

    template <typename ActivePtxPlan>
    THRUST_AGENT_ENTRY(bool       ping,
                       KeysIt     keys_ping,
                       ItemsIt    items_ping,
                       Size       keys_count,
                       key_type*  keys_pong,
                       item_type* items_pong,
                       CompareOp  compare_op,
                       Size*      merge_partitions,
                       Size       coop,
                       char*      shmem)
    {
      using temp_storage_t = typename ActivePtxPlan::TempStorage;
      auto &storage        = *reinterpret_cast<temp_storage_t *>(shmem);

      impl<ActivePtxPlan>{
        ping,
        storage,
        core::make_load_iterator(ActivePtxPlan{}, keys_ping),
        core::make_load_iterator(ActivePtxPlan{}, items_ping),
        core::make_load_iterator(ActivePtxPlan{}, keys_pong),
        core::make_load_iterator(ActivePtxPlan{}, items_pong),
        keys_count,
        keys_pong,
        items_pong,
        keys_ping,
        items_ping,
        compare_op,
        merge_partitions,
        coop};
    }
  };    // struct MergeAgent;

  /////////////////////////

  template <class SORT_ITEMS,
            class STABLE,
            class KeysIt,
            class ItemsIt,
            class Size,
            class CompareOp>
  THRUST_RUNTIME_FUNCTION cudaError_t
  doit_step(void*        d_temp_storage,
            size_t&      temp_storage_bytes,
            KeysIt       keys,
            ItemsIt      items,
            Size         keys_count,
            CompareOp    compare_op,
            cudaStream_t stream,
            bool         debug_sync)
  {
    cudaError_t status = cudaSuccess;

    if (!d_temp_storage)
    { // Initialize this for early return.
      temp_storage_bytes = 0;
    }

    if (keys_count == 0)
    {
      return status;
    }

    using key_type  = typename iterator_traits<KeysIt>::value_type;
    using item_type = typename iterator_traits<ItemsIt>::value_type;

    // Declare type aliases for agents, etc:
    using block_sort_agent_t =
      BlockSortAgent<KeysIt, ItemsIt, Size, CompareOp, SORT_ITEMS, STABLE>;
    using partition_agent_t = PartitionAgent<KeysIt, Size, CompareOp>;
    using merge_agent_t =
      MergeAgent<KeysIt, ItemsIt, Size, CompareOp, SORT_ITEMS>;

    // Create PtxPlans and AgentPlans:
    const auto partition_ptx_plan = typename partition_agent_t::PtxPlan{};
    const auto partition_agent_plan = core::AgentPlan{partition_ptx_plan};

    const auto merge_agent_plan =
      core::AgentPlanFromTunings<merge_agent_t>::get();
    const auto block_sort_agent_plan = merge_agent_plan;

    // Figure out temp_storage_size:
    const int tile_size = merge_agent_plan.items_per_tile;
    const Size num_tiles = (keys_count + tile_size - 1) / tile_size;

    const std::size_t temp_storage1 = (1 + num_tiles) * sizeof(Size);
    const std::size_t temp_storage2 = keys_count * sizeof(key_type);
    const std::size_t temp_storage3 = keys_count * sizeof(item_type) *
                                      SORT_ITEMS::value;
    const std::size_t temp_storage4 =
      core::vshmem_size(max(block_sort_agent_plan.shared_memory_size,
                            merge_agent_plan.shared_memory_size),
                        num_tiles);

    void *allocations[4]            = {nullptr, nullptr, nullptr, nullptr};
    std::size_t allocation_sizes[4] = {temp_storage1,
                                       temp_storage2,
                                       temp_storage3,
                                       temp_storage4};

    status = core::alias_storage(d_temp_storage,
                                 temp_storage_bytes,
                                 allocations,
                                 allocation_sizes);
    CUDA_CUB_RET_IF_FAIL(status);

    if (d_temp_storage == NULL)
    {
      return status;
    };

    const int num_passes = static_cast<int>(thrust::detail::log2_ri(num_tiles));
    bool ping = !(1 & num_passes);

    auto *merge_partitions = reinterpret_cast<Size *>(allocations[0]);
    auto *keys_buffer      = reinterpret_cast<key_type *>(allocations[1]);
    auto *items_buffer     = reinterpret_cast<item_type *>(allocations[2]);
    auto *vshmem_ptr       = temp_storage4 > 0
                               ? reinterpret_cast<char *>(allocations[3])
                               : nullptr;

    using block_sort_agent_launcher_t = core::AgentLauncher<block_sort_agent_t>;
    block_sort_agent_launcher_t bsa{block_sort_agent_plan,
                                    keys_count,
                                    stream,
                                    vshmem_ptr,
                                    "block_sort_agent",
                                    debug_sync};
    bsa.launch_ptx_dispatch(typename block_sort_agent_t::Tunings{},
                            ping,
                            keys,
                            items,
                            keys_count,
                            keys_buffer,
                            items_buffer,
                            compare_op);
    CUDA_CUB_RET_IF_FAIL(cudaPeekAtLastError());

    const std::size_t num_partitions = num_tiles + 1;

    using partition_agent_launcher_t = core::AgentLauncher<partition_agent_t>;
    using merge_agent_launcher_t     = core::AgentLauncher<merge_agent_t>;

    partition_agent_launcher_t pa{partition_agent_plan,
                                  num_partitions,
                                  stream,
                                  "partition_agent",
                                  debug_sync};
    merge_agent_launcher_t ma{merge_agent_plan,
                              keys_count,
                              stream,
                              vshmem_ptr,
                              "merge_agent",
                              debug_sync};

    for (int pass = 0; pass < num_passes; ++pass, ping = !ping)
    {
      Size coop = Size(2) << pass;

      pa.launch_ptx_plan(partition_ptx_plan,
                         ping,
                         keys,
                         keys_buffer,
                         keys_count,
                         num_partitions,
                         merge_partitions,
                         compare_op,
                         coop,
                         merge_agent_plan.items_per_tile);
      CUDA_CUB_RET_IF_FAIL(cudaPeekAtLastError());

      ma.launch_ptx_dispatch(typename merge_agent_t::Tunings{},
                             ping,
                             keys,
                             items,
                             keys_count,
                             keys_buffer,
                             items_buffer,
                             compare_op,
                             merge_partitions,
                             coop);
      CUDA_CUB_RET_IF_FAIL(cudaPeekAtLastError());
    }

    return status;
  }

  template <typename SORT_ITEMS,
            typename STABLE,
            typename Derived,
            typename KeysIt,
            typename ItemsIt,
            typename CompareOp>
  THRUST_RUNTIME_FUNCTION
  void merge_sort(execution_policy<Derived>& policy,
                  KeysIt                     keys_first,
                  KeysIt                     keys_last,
                  ItemsIt                    items_first,
                  CompareOp                  compare_op)

  {
    typedef typename iterator_traits<KeysIt>::difference_type size_type;

    size_type count = static_cast<size_type>(thrust::distance(keys_first, keys_last));

    size_t       storage_size = 0;
    cudaStream_t stream       = cuda_cub::stream(policy);
    bool         debug_sync   = THRUST_DEBUG_SYNC_FLAG;

    cudaError_t status;
    status = doit_step<SORT_ITEMS, STABLE>(NULL,
                                           storage_size,
                                           keys_first,
                                           items_first,
                                           count,
                                           compare_op,
                                           stream,
                                           debug_sync);
    cuda_cub::throw_on_error(status, "merge_sort: failed on 1st step");

    // Allocate temporary storage.
    thrust::detail::temporary_array<thrust::detail::uint8_t, Derived>
      tmp(policy, storage_size);
    void *ptr = static_cast<void*>(tmp.data().get());

    status = doit_step<SORT_ITEMS, STABLE>(ptr,
                                           storage_size,
                                           keys_first,
                                           items_first,
                                           count,
                                           compare_op,
                                           stream,
                                           debug_sync);
    cuda_cub::throw_on_error(status, "merge_sort: failed on 2nd step");

    status = cuda_cub::synchronize(policy);
    cuda_cub::throw_on_error(status, "merge_sort: failed to synchronize");
  }
}    // namespace __merge_sort

namespace __radix_sort {

  template <class SORT_ITEMS, class Comparator>
  struct dispatch;

  // sort keys in ascending order
  template <class K>
  struct dispatch<thrust::detail::false_type, thrust::less<K> >
  {
    template <class Key, class Item, class Size>
    THRUST_RUNTIME_FUNCTION static cudaError_t
    doit(void*                    d_temp_storage,
         size_t&                  temp_storage_bytes,
         cub::DoubleBuffer<Key>&  keys_buffer,
         cub::DoubleBuffer<Item>& /*items_buffer*/,
         Size                     count,
         cudaStream_t             stream,
         bool                     debug_sync)
    {
      return cub::DeviceRadixSort::SortKeys(d_temp_storage,
                                            temp_storage_bytes,
                                            keys_buffer,
                                            static_cast<int>(count),
                                            0,
                                            static_cast<int>(sizeof(Key) * 8),
                                            stream,
                                            debug_sync);
    }
  }; // struct dispatch -- sort keys in ascending order;

  // sort keys in descending order
  template <class K>
  struct dispatch<thrust::detail::false_type, thrust::greater<K> >
  {
    template <class Key, class Item, class Size>
    THRUST_RUNTIME_FUNCTION static cudaError_t
    doit(void*                    d_temp_storage,
         size_t&                  temp_storage_bytes,
         cub::DoubleBuffer<Key>&  keys_buffer,
         cub::DoubleBuffer<Item>& /*items_buffer*/,
         Size                     count,
         cudaStream_t             stream,
         bool                     debug_sync)
    {
      return cub::DeviceRadixSort::SortKeysDescending(d_temp_storage,
                                                      temp_storage_bytes,
                                                      keys_buffer,
                                                      static_cast<int>(count),
                                                      0,
                                                      static_cast<int>(sizeof(Key) * 8),
                                                      stream,
                                                      debug_sync);
    }
  }; // struct dispatch -- sort keys in descending order;

  // sort pairs in ascending order
  template <class K>
  struct dispatch<thrust::detail::true_type, thrust::less<K> >
  {
    template <class Key, class Item, class Size>
    THRUST_RUNTIME_FUNCTION static cudaError_t
    doit(void*                    d_temp_storage,
         size_t&                  temp_storage_bytes,
         cub::DoubleBuffer<Key>&  keys_buffer,
         cub::DoubleBuffer<Item>& items_buffer,
         Size                     count,
         cudaStream_t             stream,
         bool                     debug_sync)
    {
      return cub::DeviceRadixSort::SortPairs(d_temp_storage,
                                             temp_storage_bytes,
                                             keys_buffer,
                                             items_buffer,
                                             static_cast<int>(count),
                                             0,
                                             static_cast<int>(sizeof(Key) * 8),
                                             stream,
                                             debug_sync);
    }
  }; // struct dispatch -- sort pairs in ascending order;

  // sort pairs in descending order
  template <class K>
  struct dispatch<thrust::detail::true_type, thrust::greater<K> >
  {
    template <class Key, class Item, class Size>
    THRUST_RUNTIME_FUNCTION static cudaError_t
    doit(void*                    d_temp_storage,
         size_t&                  temp_storage_bytes,
         cub::DoubleBuffer<Key>&  keys_buffer,
         cub::DoubleBuffer<Item>& items_buffer,
         Size                     count,
         cudaStream_t             stream,
         bool                     debug_sync)
    {
      return cub::DeviceRadixSort::SortPairsDescending(d_temp_storage,
                                                       temp_storage_bytes,
                                                       keys_buffer,
                                                       items_buffer,
                                                       static_cast<int>(count),
                                                       0,
                                                       static_cast<int>(sizeof(Key) * 8),
                                                       stream,
                                                       debug_sync);
    }
  }; // struct dispatch -- sort pairs in descending order;

  template <typename SORT_ITEMS,
            typename Derived,
            typename Key,
            typename Item,
            typename Size,
            typename CompareOp>
  THRUST_RUNTIME_FUNCTION
  void radix_sort(execution_policy<Derived>& policy,
                  Key*                       keys,
                  Item*                      items,
                  Size                       count,
                  CompareOp)
  {
    size_t       temp_storage_bytes = 0;
    cudaStream_t stream             = cuda_cub::stream(policy);
    bool         debug_sync         = THRUST_DEBUG_SYNC_FLAG;

    cub::DoubleBuffer<Key>  keys_buffer(keys, NULL);
    cub::DoubleBuffer<Item> items_buffer(items, NULL);

    Size keys_count = count;
    Size items_count = SORT_ITEMS::value ? count : 0;

    cudaError_t status;

    status = dispatch<SORT_ITEMS, CompareOp>::doit(NULL,
                                                   temp_storage_bytes,
                                                   keys_buffer,
                                                   items_buffer,
                                                   keys_count,
                                                   stream,
                                                   debug_sync);
    cuda_cub::throw_on_error(status, "radix_sort: failed on 1st step");

    size_t keys_temp_storage  = core::align_to(sizeof(Key) * keys_count, 128);
    size_t items_temp_storage = core::align_to(sizeof(Item) * items_count, 128);

    size_t storage_size = keys_temp_storage
                        + items_temp_storage
                        + temp_storage_bytes;

    // Allocate temporary storage.
    thrust::detail::temporary_array<thrust::detail::uint8_t, Derived>
      tmp(policy, storage_size);

    keys_buffer.d_buffers[1]  = thrust::detail::aligned_reinterpret_cast<Key*>(
      tmp.data().get()
    );
    items_buffer.d_buffers[1] = thrust::detail::aligned_reinterpret_cast<Item*>(
      tmp.data().get() + keys_temp_storage
    );
    void *ptr = static_cast<void*>(
      tmp.data().get() + keys_temp_storage + items_temp_storage
    );

    status = dispatch<SORT_ITEMS, CompareOp>::doit(ptr,
                                                   temp_storage_bytes,
                                                   keys_buffer,
                                                   items_buffer,
                                                   keys_count,
                                                   stream,
                                                   debug_sync);
    cuda_cub::throw_on_error(status, "radix_sort: failed on 2nd step");

    if (keys_buffer.selector != 0)
    {
      Key* temp_ptr = reinterpret_cast<Key*>(keys_buffer.d_buffers[1]);
      cuda_cub::copy_n(policy, temp_ptr, keys_count, keys);
    }
    THRUST_IF_CONSTEXPR(SORT_ITEMS::value)
    {
      if (items_buffer.selector != 0)
      {
        Item *temp_ptr = reinterpret_cast<Item *>(items_buffer.d_buffers[1]);
        cuda_cub::copy_n(policy, temp_ptr, items_count, items);
      }
    }
  }
}    // __radix_sort

//---------------------------------------------------------------------
// Smart sort picks at compile-time whether to dispatch radix or merge sort
//---------------------------------------------------------------------

namespace __smart_sort {

  template <class Key, class CompareOp>
  struct can_use_primitive_sort
      : thrust::detail::and_<
            thrust::detail::is_arithmetic<Key>,
            thrust::detail::or_<
                thrust::detail::is_same<CompareOp, thrust::less<Key> >,
                thrust::detail::is_same<CompareOp, thrust::greater<Key> > > > {};

  template <class Iterator, class CompareOp>
  struct enable_if_primitive_sort
      : thrust::detail::enable_if<
            can_use_primitive_sort<typename iterator_value<Iterator>::type,
                                   CompareOp>::value> {};

  template <class Iterator, class CompareOp>
  struct enable_if_comparison_sort
      : thrust::detail::disable_if<
            can_use_primitive_sort<typename iterator_value<Iterator>::type,
                                   CompareOp>::value> {};


  template <class SORT_ITEMS,
            class STABLE,
            class Policy,
            class KeysIt,
            class ItemsIt,
            class CompareOp>
  THRUST_RUNTIME_FUNCTION typename enable_if_comparison_sort<KeysIt, CompareOp>::type
  smart_sort(Policy&   policy,
             KeysIt    keys_first,
             KeysIt    keys_last,
             ItemsIt   items_first,
             CompareOp compare_op)
  {
    __merge_sort::merge_sort<SORT_ITEMS, STABLE>(policy,
                                                 keys_first,
                                                 keys_last,
                                                 items_first,
                                                 compare_op);

  }

  template <class SORT_ITEMS,
            class STABLE,
            class Policy,
            class KeysIt,
            class ItemsIt,
            class CompareOp>
  THRUST_RUNTIME_FUNCTION typename enable_if_primitive_sort<KeysIt, CompareOp>::type
  smart_sort(execution_policy<Policy>& policy,
             KeysIt                    keys_first,
             KeysIt                    keys_last,
             ItemsIt                   items_first,
             CompareOp                 compare_op)
  {
    // ensure sequences have trivial iterators
    thrust::detail::trivial_sequence<KeysIt, Policy>
        keys(policy, keys_first, keys_last);

    if (SORT_ITEMS::value)
    {
      thrust::detail::trivial_sequence<ItemsIt, Policy>
          values(policy, items_first, items_first + (keys_last - keys_first));

      __radix_sort::radix_sort<SORT_ITEMS>(
          policy,
          thrust::raw_pointer_cast(&*keys.begin()),
          thrust::raw_pointer_cast(&*values.begin()),
          keys_last - keys_first,
          compare_op);

      if (!is_contiguous_iterator<ItemsIt>::value)
      {
        cuda_cub::copy(policy, values.begin(), values.end(), items_first);
      }
    }
    else
    {
      __radix_sort::radix_sort<SORT_ITEMS>(
          policy,
          thrust::raw_pointer_cast(&*keys.begin()),
          thrust::raw_pointer_cast(&*keys.begin()),
          keys_last - keys_first,
          compare_op);
    }

    // copy results back, if necessary
    if (!is_contiguous_iterator<KeysIt>::value)
    {
      cuda_cub::copy(policy, keys.begin(), keys.end(), keys_first);
    }

    cuda_cub::throw_on_error(
      cuda_cub::synchronize(policy),
      "smart_sort: failed to synchronize");
  }
}    // namespace __smart_sort


//-------------------------
// Thrust API entry points
//-------------------------


__thrust_exec_check_disable__
template <class Derived, class ItemsIt, class CompareOp>
void __host__ __device__
sort(execution_policy<Derived>& policy,
     ItemsIt                    first,
     ItemsIt                    last,
     CompareOp                  compare_op)
{
  if (__THRUST_HAS_CUDART__)
  {
    typedef typename thrust::iterator_value<ItemsIt>::type item_type;
    __smart_sort::smart_sort<thrust::detail::false_type, thrust::detail::false_type>(
        policy, first, last, (item_type*)NULL, compare_op);
  }
  else
  {
#if !__THRUST_HAS_CUDART__
    thrust::sort(cvt_to_seq(derived_cast(policy)), first, last, compare_op);
#endif
  }
}

__thrust_exec_check_disable__
template <class Derived, class ItemsIt, class CompareOp>
void __host__ __device__
stable_sort(execution_policy<Derived>& policy,
            ItemsIt                    first,
            ItemsIt                    last,
            CompareOp                  compare_op)
{
  if (__THRUST_HAS_CUDART__)
  {
    typedef typename thrust::iterator_value<ItemsIt>::type item_type;
    __smart_sort::smart_sort<thrust::detail::false_type, thrust::detail::true_type>(
        policy, first, last, (item_type*)NULL, compare_op);
  }
  else
  {
#if !__THRUST_HAS_CUDART__
    thrust::stable_sort(cvt_to_seq(derived_cast(policy)), first, last, compare_op);
#endif
  }
}

__thrust_exec_check_disable__
template <class Derived, class KeysIt, class ValuesIt, class CompareOp>
void __host__ __device__
sort_by_key(execution_policy<Derived>& policy,
            KeysIt                     keys_first,
            KeysIt                     keys_last,
            ValuesIt                   values,
            CompareOp                  compare_op)
{
  if (__THRUST_HAS_CUDART__)
  {
    __smart_sort::smart_sort<thrust::detail::true_type, thrust::detail::false_type>(
        policy, keys_first, keys_last, values, compare_op);
  }
  else
  {
#if !__THRUST_HAS_CUDART__
    thrust::sort_by_key(
        cvt_to_seq(derived_cast(policy)), keys_first, keys_last, values, compare_op);
#endif
  }
}

__thrust_exec_check_disable__
template <class Derived,
          class KeysIt,
          class ValuesIt,
          class CompareOp>
void __host__ __device__
stable_sort_by_key(execution_policy<Derived> &policy,
            KeysIt                     keys_first,
            KeysIt                     keys_last,
            ValuesIt                   values,
            CompareOp                  compare_op)
{
  if (__THRUST_HAS_CUDART__)
  {
    __smart_sort::smart_sort<thrust::detail::true_type, thrust::detail::true_type>(
        policy, keys_first, keys_last, values, compare_op);
  }
  else
  {
#if !__THRUST_HAS_CUDART__
    thrust::stable_sort_by_key(
        cvt_to_seq(derived_cast(policy)), keys_first, keys_last, values, compare_op);
#endif
  }
}

// API with default comparator

template <class Derived, class ItemsIt>
void __host__ __device__
sort(execution_policy<Derived>& policy,
     ItemsIt                    first,
     ItemsIt                    last)
{
  typedef typename thrust::iterator_value<ItemsIt>::type item_type;
  cuda_cub::sort(policy, first, last, less<item_type>());
}

template <class Derived, class ItemsIt>
void __host__ __device__
stable_sort(execution_policy<Derived>& policy,
            ItemsIt                    first,
            ItemsIt                    last)
{
  typedef typename thrust::iterator_value<ItemsIt>::type item_type;
  cuda_cub::stable_sort(policy, first, last, less<item_type>());
}

template <class Derived, class KeysIt, class ValuesIt>
void __host__ __device__
sort_by_key(execution_policy<Derived>& policy,
            KeysIt                     keys_first,
            KeysIt                     keys_last,
            ValuesIt                   values)
{
  typedef typename thrust::iterator_value<KeysIt>::type key_type;
  cuda_cub::sort_by_key(policy, keys_first, keys_last, values, less<key_type>());
}

template <class Derived, class KeysIt, class ValuesIt>
void __host__ __device__
stable_sort_by_key(
    execution_policy<Derived>& policy, KeysIt keys_first, KeysIt keys_last, ValuesIt values)
{
  typedef typename thrust::iterator_value<KeysIt>::type key_type;
  cuda_cub::stable_sort_by_key(policy, keys_first, keys_last, values, less<key_type>());
}


}    // namespace cuda_cub
} // end namespace thrust
#endif
