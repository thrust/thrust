/*
 *  Copyright 2008-2020 NVIDIA Corporation
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

/*! \file 
 *  \brief Test case for Doxybook rendering.
 */

#pragma once

namespace thrust
{

/*! \addtogroup test Test 
 *  \{
 */

/*! \brief \c test_predefined_friend_class is a class intended to exercise and
 *  test Doxybook rendering.
 */
template <typename... Z>
struct test_predefined_friend_class {};

/*! \brief \c test_class is a class intended to exercise and test Doxybook
 *  rendering.
 *
 *  It does many things.
 *
 *  \see test_function
 */
template <typename T, typename U>
struct test_class
{
  template <typename Z>
  struct test_nested_class {};

  int test_member_variable = 0; ///< A test member variable.

  [[deprecated]] constexpr int test_member_constant = 42; ///< A test member constant.

  template <typename X, typename Y>
  using other = test_class<X, Y>;

  enum class test_enum {
    A = 15, ///< An enumerator. It is equal to 15.
    B,
    C
  };

  /*! \brief Construct an empty test class.
   */
  __host__ __device__ constexpr
  test_class();

  __host__ __device__ constexpr
  int test_member_function();

  template <typename Z>
  friend void test_friend_function();

  template <typename Z>
  friend struct test_friend_class;

  template <typename... Z>
  friend struct test_predefined_friend_class;
};

/*! \brief \c test_function is a function intended to exercise and test Doxybook
 *  rendering.
 */
template <typename T>
void test_function(T const& a, test_class<T, T const>&& b);

/*! \brief \c test_parameter_overflow is a function intended to test Doxybook's
 *  rendering of function and template parameters that exceed the length of a
 *  line.
 */
template <typename T = test_predefined_friend_class<int, int, int, int, int, int, int, int, int, int, int, int>,
  typename U = test_predefined_friend_class<int, int, int, int, int, int, int, int, int, int, int, int>,
  typename V = test_predefined_friend_class<int, int, int, int, int, int, int, int, int, int, int, int>
>
test_predefined_friend_class<int, int, int, int, int, int, int, int, int, int, int, int, int, int, int>
test_function(test_predefined_friend_class<int, int, int, int, int, int, int, int, int, int, int, int> t,
  test_predefined_friend_class<int, int, int, int, int, int, int, int, int, int, int, int> u,
  test_predefined_friend_class<int, int, int, int, int, int, int, int, int, int, int, int> v);

/*! \brief \c test_namespace is a namespace intended to exercise and test
 *  Doxybook rendering.
 */
namespace test_namespace {

inline constexpr int test_constant = 12; 

/*! \brief \c nested_function is a function intended to exercise and test
 *  Doxybook rendering.
 */
template <typename T, typename U>
auto test_nested_function(T t, U u) noexcept(noexcept(t + u)) -> decltype(t + u)
{ return t + u; }
 
} // namespace test_namespace

/*! \brief \c THRUST_TEST_MACRO is a macro intended to exercise and test
 *  Doxybook rendering.
 */
#define THRUST_TEST_MACRO(x, y) thrust::test_namespace::nested_function(x, y)

/*! \} // test 
 */

} // namespace thrust

