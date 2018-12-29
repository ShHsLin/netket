// Copyright 2018 The Simons Foundation, Inc. - All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef NETKET_PYSUPERVISED_HPP
#define NETKET_PYSUPERVISED_HPP

#include <mpi.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <complex>
#include <vector>
#include "supervised.hpp"

namespace py = pybind11;

namespace netket {

void AddSupervisedModule(py::module &m) {
  auto subm = m.def_submodule("supervised");

  py::class_<Supervised>(subm, "supervised")
      .def(py::init([](SamplerType &sa, AbstractOptimizer &op, int batch_size,
                       int niter_opt, std::vector<Eigen::VectorXd> samples,
                       std::vector<Eigen::VectorXd> targets,
                       std::string output_file) {
             return Supervised{sa,
                               op,
                               batch_size,
                               niter_opt,
                               std::move(samples),
                               std::move(targets),
                               output_file};
           }),
           py::keep_alive<1, 2>(), py::keep_alive<1, 3>(),
           py::keep_alive<1, 7>(), py::arg("sampler"), py::arg("optimizer"),
           py::arg("batch_size"), py::arg("niter_opt"), py::arg("samples"),
           py::arg("targets"), py::arg("output_file"))
      .def("run", &Supervised::Run);
}

}  // namespace netket

#endif
