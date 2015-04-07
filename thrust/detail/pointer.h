/*
 *  Copyright 2008-2013 NVIDIA Corporation
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

#pragma once

#include <thrust/detail/config.h>
#include <thrust/iterator/iterator_adaptor.h>
#include <thrust/iterator/detail/iterator_traversal_tags.h>
#include <thrust/detail/type_traits/pointer_traits.h>
#include <thrust/detail/type_traits.h>
#include <thrust/detail/reference_forward_declaration.h>
#include <ostream>

namespace thrust
{

// declare pointer with default values of template parameters
template<typename Element, typename Tag, typename Reference = use_default, typename Derived = use_default> class pointer;

} // end thrust


// specialize thrust::iterator_traits to avoid problems with the name of
// pointer's constructor shadowing its nested pointer type
// do this before pointer is defined so the specialization is correctly
// used inside the definition
namespace thrust
{

template<typename Element, typename Tag, typename Reference, typename Derived>
  struct iterator_traits<thrust::pointer<Element,Tag,Reference,Derived> >
{
  private:
    typedef thrust::pointer<Element,Tag,Reference,Derived> ptr;

  public:
    typedef typename ptr::iterator_category iterator_category;
    typedef typename ptr::value_type        value_type;
    typedef typename ptr::difference_type   difference_type;
    // XXX implement this type (the result of operator->) later
    typedef void                             pointer;
    typedef typename ptr::reference         reference;
}; // end iterator_traits

} // end thrust


namespace thrust
{

namespace detail
{

// this metafunction computes the type of iterator_adaptor thrust::pointer should inherit from
template<typename Element, typename Tag, typename Reference, typename Derived>
  struct pointer_base
{
  // void pointers should have no element type
  // note that we remove_cv from the Element type to get the value_type
  typedef typename thrust::detail::eval_if<
    thrust::detail::is_void<typename thrust::detail::remove_const<Element>::type>::value,
    thrust::detail::identity_<void>,
    thrust::detail::remove_cv<Element>
  >::type value_type;

  // if no Derived type is given, just use pointer
  typedef typename thrust::detail::eval_if<
    thrust::detail::is_same<Derived,use_default>::value,
    thrust::detail::identity_<pointer<Element,Tag,Reference,Derived> >,
    thrust::detail::identity_<Derived>
  >::type derived_type;

  // void pointers should have no reference type
  // if no Reference type is given, just use reference
  typedef typename thrust::detail::eval_if<
    thrust::detail::is_void<typename thrust::detail::remove_const<Element>::type>::value,
    thrust::detail::identity_<void>,
    thrust::detail::eval_if<
      thrust::detail::is_same<Reference,use_default>::value,
      thrust::detail::identity_<reference<Element,derived_type> >,
      thrust::detail::identity_<Reference>
    >
  >::type reference_arg;

  typedef thrust::iterator_adaptor<
    derived_type,                        // pass along the type of our Derived class to iterator_adaptor
    Element *,                           // we adapt a raw pointer
    value_type,                          // the value type
    Tag,                                 // system tag
    thrust::random_access_traversal_tag, // pointers have random access traversal
    reference_arg,                       // pass along our Reference type
    std::ptrdiff_t
  > type;
}; // end pointer_base


} // end detail


// the base type for all of thrust's tagged pointers.
// for reasonable pointer-like semantics, derived types should reimplement the following:
// 1. no-argument constructor
// 2. constructor from OtherElement *
// 3. constructor from OtherPointer related by convertibility
// 4. assignment from OtherPointer related by convertibility
// These should just call the corresponding members of pointer.
template<typename Element, typename Tag, typename Reference, typename Derived>
  class pointer
    : public thrust::detail::pointer_base<Element,Tag,Reference,Derived>::type
{
  private:
    typedef typename thrust::detail::pointer_base<Element,Tag,Reference,Derived>::type         super_t;

    typedef typename thrust::detail::pointer_base<Element,Tag,Reference,Derived>::derived_type derived_type;

    // friend iterator_core_access to give it access to dereference
    friend class thrust::iterator_core_access;

    __host__ __device__
    typename super_t::reference dereference() const;

    // don't provide access to this part of super_t's interface
    using super_t::base;
    using typename super_t::base_type;

  public:
    typedef typename super_t::base_type raw_pointer;

    // constructors
    
    __host__ __device__
    pointer();

    // OtherValue shall be convertible to Value
    // XXX consider making the pointer implementation a template parameter which defaults to Element *
    template<typename OtherElement>
    __host__ __device__
    explicit pointer(OtherElement *ptr);

    // OtherPointer's element_type shall be convertible to Element
    // OtherPointer's system shall be convertible to Tag
    template<typename OtherPointer>
    __host__ __device__
    pointer(const OtherPointer &other,
            typename thrust::detail::enable_if_pointer_is_convertible<
              OtherPointer,
              pointer<Element,Tag,Reference,Derived>
            >::type * = 0);

    // assignment
    
    // OtherPointer's element_type shall be convertible to Element
    // OtherPointer's system shall be convertible to Tag
    template<typename OtherPointer>
    __host__ __device__
    typename thrust::detail::enable_if_pointer_is_convertible<
      OtherPointer,
      pointer,
      derived_type &
    >::type
    operator=(const OtherPointer &other);

    // observers

    __host__ __device__
    Element *get() const;
}; // end pointer

// Output stream operator
template<typename Element, typename Tag, typename Reference, typename Derived,
         typename charT, typename traits>
std::basic_ostream<charT, traits> &
operator<<(std::basic_ostream<charT, traits> &os,
           const pointer<Element, Tag, Reference, Derived> &p);

} // end thrust

#include <thrust/detail/pointer.inl>
