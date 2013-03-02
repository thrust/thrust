/*
 *  Copyright 2008-2012 NVIDIA Corporation
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


/*! \file unique.h
 *  \brief Sequential implementations of unique algorithms.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/system/detail/sequential/tag.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/pair.h>

namespace thrust
{
namespace system
{
namespace detail
{
namespace sequential
{


template<typename InputIterator,
         typename OutputIterator,
         typename BinaryPredicate>
__host__ __device__
  OutputIterator unique_copy(tag,
                             InputIterator first,
                             InputIterator last,
                             OutputIterator output,
                             BinaryPredicate binary_pred)
{
  typedef typename thrust::iterator_traits<InputIterator>::value_type T;

  if(first != last)
  {
    T prev = *first;

    for(++first; first != last; ++first)
    {
      T temp = *first;

      if (!binary_pred(prev, temp))
      {
        *output = prev;

        ++output;

        prev = temp;
      }
    }

    *output = prev;
    ++output;
  }

  return output;
} // end unique_copy()


template<typename ForwardIterator,
         typename BinaryPredicate>
__host__ __device__
  ForwardIterator unique(tag seq,
                         ForwardIterator first,
                         ForwardIterator last,
                         BinaryPredicate binary_pred)
{
  // sequential unique_copy permits in-situ operation
  return unique_copy(seq, first, last, first, binary_pred);
} // end unique()


} // end namespace sequential
} // end namespace detail
} // end namespace system
} // end namespace thrust
