# Check all files in thrust to make sure that they use
# THRUST_NAMESPACE_BEGIN/END instead of bare `namespace thrust {}` declarations.
#
# This is run as a ctest test named `thrust.test.cmake.check_namespace`, or
# manually with:
# cmake -D "Thrust_SOURCE_DIR=<thrust project root>" -P check_namespace.cmake

cmake_minimum_required(VERSION 3.15)

set(exclusions
  # This defines the macros and must have bare namespace declarations:
  thrust/detail/config/namespace.h
)

function(count_substrings input search_regex output_var)
  string(REGEX MATCHALL "${search_regex}" matches "${input}")
  list(LENGTH matches num_matches)
  set(${output_var} ${num_matches} PARENT_SCOPE)
endfunction()

set(bare_ns_regex "namespace[ \n\r\t]+thrust[ \n\r\t]*\\{")

# Validation check for the above regex:
count_substrings([=[
namespace thrust{
namespace thrust {
namespace  thrust  {
 namespace thrust {
namespace thrust
{
namespace
thrust
{
]=]
  ${bare_ns_regex} valid_count)
if (NOT valid_count EQUAL 6)
  message(FATAL_ERROR "Validation of bare namespace regex failed: "
                      "Matched ${valid_count} times, expected 6.")
endif()

set(exit_code 0)
file(GLOB_RECURSE thrust_srcs
  RELATIVE "${Thrust_SOURCE_DIR}"
  "${Thrust_SOURCE_DIR}/thrust/*.h"
  "${Thrust_SOURCE_DIR}/thrust/*.inl"
  "${Thrust_SOURCE_DIR}/testing/*.cu"
  "${Thrust_SOURCE_DIR}/examples/*.cu"
)

foreach(src ${thrust_srcs})
  if (${src} IN_LIST exclusions)
    continue()
  endif()

  file(READ "${Thrust_SOURCE_DIR}/${src}" src_contents)

  count_substrings("${src_contents}" "${bare_ns_regex}" bare_ns_count)
  count_substrings("${src_contents}" THRUST_NS_PREFIX prefix_count)
  count_substrings("${src_contents}" THRUST_NS_POSTFIX postfix_count)
  count_substrings("${src_contents}" THRUST_NAMESPACE_BEGIN begin_count)
  count_substrings("${src_contents}" THRUST_NAMESPACE_END end_count)

  if (NOT bare_ns_count EQUAL 0)
    message("'${src}' contains 'namespace thrust {...}'. Replace with THRUST_NAMESPACE macros.")
    set(exit_code 1)
  endif()

  if (NOT prefix_count EQUAL 0)
    message("'${src}' contains 'THRUST_NS_PREFIX'. Replace with THRUST_NAMESPACE macros.")
    set(exit_code 1)
  endif()

  if (NOT postfix_count EQUAL 0)
    message("'${src}' contains 'THRUST_NS_POSTFIX'. Replace with THRUST_NAMESPACE macros.")
    set(exit_code 1)
  endif()

  if (NOT begin_count EQUAL end_count)
    message("'${src}' namespace macros are unbalanced:")
    message(" - THRUST_NAMESPACE_BEGIN occurs ${begin_count} times.")
    message(" - THRUST_NAMESPACE_END   occurs ${end_count} times.")
    set(exit_code 1)
  endif()
endforeach()

if (NOT exit_code EQUAL 0)
  message(FATAL_ERROR "Errors detected.")
endif()
