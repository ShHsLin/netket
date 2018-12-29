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

#ifndef NETKET_PYHILBERT_HPP
#define NETKET_PYHILBERT_HPP

#include <mpi.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <complex>
#include <vector>
#include "hilbert.hpp"

namespace py = pybind11;

namespace netket {

constexpr int HilbertIndex::MaxStates;

#define ADDHILBERTMETHODS(name)                                  \
                                                                 \
  .def_property_readonly("is_discrete", &name::IsDiscrete)       \
      .def_property_readonly("local_size", &name::LocalSize)     \
      .def_property_readonly("size", &name::Size)                \
      .def_property_readonly("local_states", &name::LocalStates) \
      .def("random_vals", &name ::RandomVals)                    \
      .def("update_conf", &name::UpdateConf)

void AddHilbertModule(py::module &m) {
  auto subm = m.def_submodule("hilbert");

  py::class_<AbstractHilbert>(subm, "Hilbert")
      ADDHILBERTMETHODS(AbstractHilbert);

  py::class_<Spin, AbstractHilbert>(subm, "Spin")
      .def(py::init<const AbstractGraph &, double>(), py::keep_alive<1, 2>(),
           py::arg("graph"), py::arg("s"))
      .def(py::init<const AbstractGraph &, double, double>(),
           py::keep_alive<1, 2>(), py::arg("graph"), py::arg("s"),
           py::arg("total_sz")) ADDHILBERTMETHODS(Spin);

  py::class_<Qubit, AbstractHilbert>(subm, "Qubit")
      .def(py::init<const AbstractGraph &>(), py::keep_alive<1, 2>(),
           py::arg("graph")) ADDHILBERTMETHODS(Qubit);

  py::class_<Boson, AbstractHilbert>(subm, "Boson")
      .def(py::init<const AbstractGraph &, int>(), py::keep_alive<1, 2>(),
           py::arg("graph"), py::arg("n_max"))
      .def(py::init<const AbstractGraph &, int, int>(), py::keep_alive<1, 2>(),
           py::arg("graph"), py::arg("n_max"), py::arg("n_bosons"))
          ADDHILBERTMETHODS(Boson);

  py::class_<CustomHilbert, AbstractHilbert>(subm, "CustomHilbert")
      .def(py::init<const AbstractGraph &, std::vector<double>>(),
           py::keep_alive<1, 2>(), py::arg("graph"), py::arg("local_states"))
          ADDHILBERTMETHODS(CustomHilbert);

  py::class_<HilbertIndex>(subm, "HilbertIndex")
      .def(py::init<const AbstractHilbert &>(), py::arg("hilbert"))
      .def_property_readonly("n_states", &HilbertIndex::NStates)
      .def("number_to_state", &HilbertIndex::NumberToState)
      .def("state_to_number", &HilbertIndex::StateToNumber)
      .def_readonly_static("max_states", &HilbertIndex::MaxStates);

}  // namespace netket

}  // namespace netket

#endif
