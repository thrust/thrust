# Thrust: The C++ Parallel Algorithms Library

<table><tr>
<th><b><a href="https://github.com/nvidia/thrust/tree/main/examples">Examples</a></b></th>
<th><b><a href="https://godbolt.org/z/rsdedW">Godbolt</a></b></th>
<th><b><a href="https://nvidia.github.io/thrust">Documentation</a></b></th>
</tr></table>

Thrust is the C++ parallel algorithms library which inspired the introduction
  of parallel algorithms to the C++ Standard Library.
Thrust's **high-level** interface greatly enhances programmer **productivity**
  while enabling performance portability between GPUs and multicore CPUs.
It builds on top of established parallel programming frameworks (such as CUDA,
  TBB, and OpenMP).
It also provides a number of general-purpose facilities similar to those found
  in the C++ Standard Library.

The NVIDIA C++ Standard Library is an open source project; it is available on
  [GitHub] and included in the NVIDIA HPC SDK and CUDA Toolkit.
If you have one of those SDKs installed, no additional installation or compiler
  flags are needed to use libcu++.

## Examples

Thrust is best learned through examples.

The following example generates random numbers serially and then transfers them
  to a parallel device where they are sorted.

```cuda
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/random.h>

int main() {
  // Generate 32M random numbers serially.
  thrust::default_random_engine rng(1337);
  thrust::uniform_int_distribution<int> dist;
  thrust::host_vector<int> h_vec(32 << 20);
  thrust::generate(h_vec.begin(), h_vec.end(), [&] { return dist(rng); });

  // Transfer data to the device.
  thrust::device_vector<int> d_vec = h_vec;

  // Sort data on the device.
  thrust::sort(d_vec.begin(), d_vec.end());

  // Transfer data back to host.
  thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());
}
```

[See it on Godbolt](https://godbolt.org/z/v3fdoE){: .btn }

This example demonstrates computing the sum of some random numbers in parallel:

```cuda
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/random.h>

int main() {
  // Generate random data serially.
  thrust::default_random_engine rng(1337);
  thrust::uniform_real_distribution<double> dist(-50.0, 50.0);
  thrust::host_vector<double> h_vec(32 << 20);
  thrust::generate(h_vec.begin(), h_vec.end(), [&] { return dist(rng); });

  // Transfer to device and compute the sum.
  thrust::device_vector<double> d_vec = h_vec;
  double x = thrust::reduce(d_vec.begin(), d_vec.end(), 0, thrust::plus<int>());
}
```

[See it on Godbolt](https://godbolt.org/z/119jxj){: .btn }

This example show how to perform such a reduction asynchronously:

```cuda
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/async/copy.h>
#include <thrust/async/reduce.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include <numeric>

int main() {
  // Generate 32M random numbers serially.
  thrust::default_random_engine rng(123456);
  thrust::uniform_real_distribution<double> dist(-50.0, 50.0);
  thrust::host_vector<double> h_vec(32 << 20);
  thrust::generate(h_vec.begin(), h_vec.end(), [&] { return dist(rng); });

  // Asynchronously transfer to the device.
  thrust::device_vector<double> d_vec(h_vec.size());
  thrust::device_event e = thrust::async::copy(h_vec.begin(), h_vec.end(),
                                               d_vec.begin());

  // After the transfer completes, asynchronously compute the sum on the device.
  thrust::device_future<double> f0 = thrust::async::reduce(thrust::device.after(e),
                                                           d_vec.begin(), d_vec.end(),
                                                           0.0, thrust::plus<double>());

  // While the sum is being computed on the device, compute the sum serially on
  // the host.
  double f1 = std::accumulate(h_vec.begin(), h_vec.end(), 0.0, thrust::plus<double>());
}
```

[See it on Godbolt](https://godbolt.org/z/rsdedW){: .btn }

## CI Status

<a href='https://gpuci.gpuopenanalytics.com/job/nvidia/job/thrust/job/prb/job/thrust-gpu-build/CXX_TYPE=gcc,CXX_VER=7,OS_TYPE=ubuntu,OS_VER=20.04,SDK_TYPE=cuda,SDK_VER=11.0-devel/'><img src='https://gpuci.gpuopenanalytics.com/job/nvidia/job/thrust/job/prb/job/thrust-gpu-build/CXX_TYPE=gcc,CXX_VER=7,OS_TYPE=ubuntu,OS_VER=20.04,SDK_TYPE=cuda,SDK_VER=11.0-devel/badge/icon?subject=NVCC%2011.0%20%2B%20GCC%207%20build%20and%20device%20tests'></a>

<a href='https://gpuci.gpuopenanalytics.com/job/nvidia/job/thrust/job/prb/job/thrust-cpu-build/CXX_TYPE=gcc,CXX_VER=9,OS_TYPE=ubuntu,OS_VER=20.04,SDK_TYPE=cuda,SDK_VER=11.0-devel/'><img src='https://gpuci.gpuopenanalytics.com/job/nvidia/job/thrust/job/prb/job/thrust-cpu-build/CXX_TYPE=gcc,CXX_VER=9,OS_TYPE=ubuntu,OS_VER=20.04,SDK_TYPE=cuda,SDK_VER=11.0-devel/badge/icon?subject=NVCC%2011.0%20%2B%20GCC%209%20build%20and%20host%20tests'></a>

<a href='https://gpuci.gpuopenanalytics.com/job/nvidia/job/thrust/job/prb/job/thrust-cpu-build/CXX_TYPE=gcc,CXX_VER=8,OS_TYPE=ubuntu,OS_VER=20.04,SDK_TYPE=cuda,SDK_VER=11.0-devel/'><img src='https://gpuci.gpuopenanalytics.com/job/nvidia/job/thrust/job/prb/job/thrust-cpu-build/CXX_TYPE=gcc,CXX_VER=8,OS_TYPE=ubuntu,OS_VER=20.04,SDK_TYPE=cuda,SDK_VER=11.0-devel/badge/icon?subject=NVCC%2011.0%20%2B%20GCC%208%20build%20and%20host%20tests'></a>

<a href='https://gpuci.gpuopenanalytics.com/job/nvidia/job/thrust/job/prb/job/thrust-cpu-build/CXX_TYPE=gcc,CXX_VER=7,OS_TYPE=ubuntu,OS_VER=20.04,SDK_TYPE=cuda,SDK_VER=11.0-devel/'><img src='https://gpuci.gpuopenanalytics.com/job/nvidia/job/thrust/job/prb/job/thrust-cpu-build/CXX_TYPE=gcc,CXX_VER=7,OS_TYPE=ubuntu,OS_VER=20.04,SDK_TYPE=cuda,SDK_VER=11.0-devel/badge/icon?subject=NVCC%2011.0%20%2B%20GCC%207%20build%20and%20host%20tests'></a>

<a href='https://gpuci.gpuopenanalytics.com/job/nvidia/job/thrust/job/prb/job/thrust-cpu-build/CXX_TYPE=gcc,CXX_VER=6,OS_TYPE=ubuntu,OS_VER=20.04,SDK_TYPE=cuda,SDK_VER=11.0-devel/'><img src='https://gpuci.gpuopenanalytics.com/job/nvidia/job/thrust/job/prb/job/thrust-cpu-build/CXX_TYPE=gcc,CXX_VER=6,OS_TYPE=ubuntu,OS_VER=20.04,SDK_TYPE=cuda,SDK_VER=11.0-devel/badge/icon?subject=NVCC%2011.0%20%2B%20GCC%206%20build%20and%20host%20tests'></a>

<a href='https://gpuci.gpuopenanalytics.com/job/nvidia/job/thrust/job/prb/job/thrust-cpu-build/CXX_TYPE=gcc,CXX_VER=5,OS_TYPE=ubuntu,OS_VER=20.04,SDK_TYPE=cuda,SDK_VER=11.0-devel/'><img src='https://gpuci.gpuopenanalytics.com/job/nvidia/job/thrust/job/prb/job/thrust-cpu-build/CXX_TYPE=gcc,CXX_VER=5,OS_TYPE=ubuntu,OS_VER=20.04,SDK_TYPE=cuda,SDK_VER=11.0-devel/badge/icon?subject=NVCC%2011.0%20%2B%20GCC%205%20build%20and%20host%20tests'></a>

<a href='https://gpuci.gpuopenanalytics.com/job/nvidia/job/thrust/job/prb/job/thrust-cpu-build/CXX_TYPE=clang,CXX_VER=9,OS_TYPE=ubuntu,OS_VER=20.04,SDK_TYPE=cuda,SDK_VER=11.0-devel/'><img src='https://gpuci.gpuopenanalytics.com/job/nvidia/job/thrust/job/prb/job/thrust-cpu-build/CXX_TYPE=clang,CXX_VER=9,OS_TYPE=ubuntu,OS_VER=20.04,SDK_TYPE=cuda,SDK_VER=11.0-devel/badge/icon?subject=NVCC%2011.0%20%2B%20Clang%209%20build%20and%20host%20tests'></a>

<a href='https://gpuci.gpuopenanalytics.com/job/nvidia/job/thrust/job/prb/job/thrust-cpu-build/CXX_TYPE=clang,CXX_VER=8,OS_TYPE=ubuntu,OS_VER=20.04,SDK_TYPE=cuda,SDK_VER=11.0-devel/'><img src='https://gpuci.gpuopenanalytics.com/job/nvidia/job/thrust/job/prb/job/thrust-cpu-build/CXX_TYPE=clang,CXX_VER=8,OS_TYPE=ubuntu,OS_VER=20.04,SDK_TYPE=cuda,SDK_VER=11.0-devel/badge/icon?subject=NVCC%2011.0%20%2B%20Clang%208%20build%20and%20host%20tests'></a>

<a href='https://gpuci.gpuopenanalytics.com/job/nvidia/job/thrust/job/prb/job/thrust-cpu-build/CXX_TYPE=clang,CXX_VER=7,OS_TYPE=ubuntu,OS_VER=20.04,SDK_TYPE=cuda,SDK_VER=11.0-devel/'><img src='https://gpuci.gpuopenanalytics.com/job/nvidia/job/thrust/job/prb/job/thrust-cpu-build/CXX_TYPE=clang,CXX_VER=7,OS_TYPE=ubuntu,OS_VER=20.04,SDK_TYPE=cuda,SDK_VER=11.0-devel/badge/icon?subject=NVCC%2011.0%20%2B%20Clang%207%20build%20and%20host%20tests'></a>

<a href='https://gpuci.gpuopenanalytics.com/job/nvidia/job/thrust/job/prb/job/thrust-cpu-build/CXX_TYPE=icc,CXX_VER=latest,OS_TYPE=ubuntu,OS_VER=20.04,SDK_TYPE=cuda,SDK_VER=11.0-devel/'><img src='https://gpuci.gpuopenanalytics.com/job/nvidia/job/thrust/job/prb/job/thrust-cpu-build/CXX_TYPE=icc,CXX_VER=latest,OS_TYPE=ubuntu,OS_VER=20.04,SDK_TYPE=cuda,SDK_VER=11.0-devel/badge/icon?subject=NVCC%2011.0%20%2B%20ICC%20build%20and%20host%20tests'></a>

<a href='https://gpuci.gpuopenanalytics.com/job/nvidia/job/thrust/job/prb/job/thrust-cpu-build/CXX_TYPE=nvcxx,CXX_VER=20.9,OS_TYPE=ubuntu,OS_VER=20.04,SDK_TYPE=nvhpc,SDK_VER=20.9-devel/'><img src='https://gpuci.gpuopenanalytics.com/job/nvidia/job/thrust/job/prb/job/thrust-cpu-build/CXX_TYPE=nvcxx,CXX_VER=20.9,OS_TYPE=ubuntu,OS_VER=20.04,SDK_TYPE=nvhpc,SDK_VER=20.9-devel/badge/icon?subject=NVC%2B%2B%2020.9%20build%20and%20host%20tests'></a>

## Adding Thrust To A CMake Project

Since Thrust is a header library, there is no need to build or install Thrust
  to use it.
The `thrust` directory contains a complete, ready-to-use Thrust
  package upon checkout from GitHub.
If you have the NVIDIA HPC SDK or the CUDA Toolkit installed, then Thrust will
  already been on the include path when using those SDKs.

We provide CMake configuration files that make it easy to include Thrust
  from other CMake projects.
See the [CMake section] for details.

## Development Process

Thrust uses the [CMake build system] to build unit tests, examples, and header
  tests.
To build Thrust as a developer, it is recommended that you use our
  containerized development system:

```bash
# Clone Thrust and CUB repos recursively:
git clone --recursive https://github.com/NVIDIA/thrust.git
cd thrust

# Build and run tests and examples:
ci/local/build.bash
```

That does the equivalent of the following, but in a clean containerized
  environment which has all dependencies installed:

```bash
# Clone Thrust and CUB repos recursively:
git clone --recursive https://github.com/NVIDIA/thrust.git
cd thrust

# Create build directory:
mkdir build
cd build

# Configure -- use one of the following:
cmake ..   # Command line interface.
ccmake ..  # ncurses GUI (Linux only).
cmake-gui  # Graphical UI, set source/build directories in the app.

# Build:
cmake --build . -j ${NUM_JOBS} # Invokes make (or ninja, etc).

# Run tests and examples:
ctest
```

By default, a serial `CPP` host system, `CUDA` accelerated device system, and
  C++14 standard are used.
This can be changed in CMake and via flags to `ci/local/build.bash`

More information on configuring your Thrust build and creating a pull request
  can be found in the [contributing section].

## Licensing

Thrust is an open source project developed on [GitHub].
Thrust is distributed under the [Apache License v2.0 with LLVM Exceptions];
  some parts are distributed under the [Apache License v2.0] and the
  [Boost License v1.0].
See the [licensing section] for more details.


[GitHub]: https://github.com/nvidia/thrust

[CMake section]: https://nvidia.github.io/thrust/setup/cmake.html
[contributing section]: https://nvidia.github.io/thrust/contributing.html
[licensing section]: https://nvidia.github.io/thrust/licensing.html

[CMake build system]: https://cmake.org

[Apache License v2.0 with LLVM Exceptions]: https://llvm.org/LICENSE.txt
[Apache License v2.0]: https://www.apache.org/licenses/LICENSE-2.0.txt
[Boost License v1.0]: https://www.boost.org/LICENSE_1_0.txt

