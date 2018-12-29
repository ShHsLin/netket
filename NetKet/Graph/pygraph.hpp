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

#ifndef NETKET_PYGRAPH_HPP
#define NETKET_PYGRAPH_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include "netket.hpp"

namespace py = pybind11;

namespace netket {

namespace detail {
/// Given a Python iterable, returns its length if it is known or 0 otherwise.
/// This can safely be used to preallocate storage on the C++ side as calls to
/// `std::vector<T>::reserve(0)` are basically noops.
inline std::size_t LengthHint(py::iterable xs) {
  auto iterator = xs.attr("__iter__")();
  if (py::hasattr(iterator, "__length_hint__")) {
    auto const n = iterator.attr("__length_hint__")().cast<long>();
    assert(n >= 0 && "Bug in Python/pybind11??");
    return static_cast<std::size_t>(n);
  }
  return 0;
}
inline std::size_t LengthHint(py::iterator x) {
  if (py::hasattr(x, "__length_hint__")) {
    auto const n = x.attr("__length_hint__")().cast<long>();
    assert(n >= 0 && "Bug in Python/pybind11??");
    return static_cast<std::size_t>(n);
  }
  return 0;
}

/// Correctly orders site indices and constructs an edge.
// TODO(twesterhout): Should we throw when `x == y`? I.e. edge from a node to
// itself is a questionable concept.
inline AbstractGraph::Edge MakeEdge(int const x, int const y) noexcept {
  using Edge = AbstractGraph::Edge;
  return (x < y) ? Edge{x, y} : Edge{y, x};
}

/// Converts a Python iterable to a list of edges. An exception is thrown if the
/// input iterable contains duplicate edges.
///
/// \postcondition For each edge (i, j) we have i <= j.
/// \postcondition The returned list contains no duplicates.
inline std::vector<AbstractGraph::Edge> Iterable2Edges(py::iterator x) {
  using std::begin;
  using std::end;
  std::vector<AbstractGraph::Edge> edges;
  edges.reserve(LengthHint(x));

  while (x != py::iterator::sentinel()) {
    int i, j;
    std::tie(i, j) = x->template cast<std::tuple<int, int>>();
    edges.push_back(MakeEdge(i, j));
    ++x;
  }

  // NOTE(twesterhout): Yes, I know that this screws up the algorithmic
  // complexity, but it's fast enough to be unnoticeable for all practical
  // purposes.
  std::sort(begin(edges), end(edges));
  if (std::unique(begin(edges), end(edges)) != end(edges)) {
    throw InvalidInputError{"Edge list contains duplicates."};
  }
  return edges;
}

/// Converts a Python iterable to a `ColorMap`. An exception is thrown if the
/// input iterable contains duplicate edges.
///
/// \postcondition For each edge (i, j) we have i <= j.
inline AbstractGraph::ColorMap Iterable2ColorMap(py::iterator x) {
  AbstractGraph::ColorMap colors;
  colors.reserve(LengthHint(x));

  while (x != py::iterator::sentinel()) {
    int i, j, color;
    std::tie(i, j, color) = x->template cast<std::tuple<int, int, int>>();
    if (!colors.emplace(MakeEdge(i, j), color).second) {
      // Failed to insert an edge because it already exists
      throw InvalidInputError{"Edge list contains duplicates."};
    }
    ++x;
  }
  return colors;
}
}  // namespace detail

namespace {
/// Given a Python iterable object constructs the edge list and (optionally)
/// the colour map for the soon to be graph. `callback` is then called in one of
/// the following ways:
/// * `callback(edges, colour_map)` if the iterable contained elements of type
/// `(int, int, int)`.
/// * `callback(edges)` if the iterable contained elements of type `(int, int)`.
template <class Function>
auto WithEdges(py::iterator first, Function&& callback)
    -> decltype(std::forward<Function>(callback)(
        std::declval<std::vector<AbstractGraph::Edge>>())) {
  using std::begin;
  using std::end;

  bool has_colours = false;
  if (first != py::iterator::sentinel()) {
    // We have at least one element, let's determine whether it's an instance
    // of `(int, int)` or `(int, int, int)`.
    try {
      // If the following line succeeds, we have a sequence of `(int, int)`.
      static_cast<void>(first->template cast<std::tuple<int, int>>());
      has_colours = false;
    } catch (py::cast_error& /*unused*/) {
      try {
        // If the following line succeeds, we have a sequence of `(int, int,
        // int)`.
        static_cast<void>(first->template cast<std::tuple<int, int, int>>());
        has_colours = true;
      } catch (py::cast_error& /*unused*/) {
        throw py::cast_error("Unable to cast Python instance of type " +
                             std::string{py::str(first->get_type())} +
                             " to either one of the following C++ types: '" +
                             py::type_id<std::tuple<int, int>>() + "', '" +
                             py::type_id<std::tuple<int, int, int>>() + "'");
      }
    }
  }

  if (has_colours) {
    auto colors = detail::Iterable2ColorMap(first);
    std::vector<AbstractGraph::Edge> edges;
    edges.reserve(colors.size());
    std::transform(
        begin(colors), end(colors), std::back_inserter(edges),
        [](std::pair<AbstractGraph::Edge, int> const& x) { return x.first; });
    return std::forward<Function>(callback)(std::move(edges),
                                            std::move(colors));
  } else {
    return std::forward<Function>(callback)(detail::Iterable2Edges(first));
  }
}
}  // namespace

// Work around the lack of C++11 support for defaulted arguments in lambdas.
namespace {
struct CustomGraphInit {
  using Edge = AbstractGraph::Edge;
  using ColorMap = AbstractGraph::ColorMap;

  std::vector<std::vector<int>> automorphisms;
  bool is_bipartite;

  auto operator()(std::vector<Edge> edges, ColorMap colors = ColorMap{})
      -> std::unique_ptr<CustomGraph> {
    return make_unique<CustomGraph>(std::move(edges), std::move(colors),
                                    std::move(automorphisms), is_bipartite);
  }
};
}  // namespace

void AddGraphModule(py::module& m) {
  auto subm = m.def_submodule("graph");

  py::class_<AbstractGraph>(subm, "Graph")
      .def_property_readonly("n_sites", &AbstractGraph::Nsites,
                             R"EOF(
              Returns the number of vertices in the graph.
           )EOF")
      .def_property_readonly(
          "edges",
          [](AbstractGraph const& x) {
            using vector_type =
                std::remove_reference<decltype(x.Edges())>::type;
            return vector_type{x.Edges()};
          },
          R"EOF(
               Returns the graph edges.
           )EOF")
      .def_property_readonly("adjacency_list", &AbstractGraph::AdjacencyList,
                             R"EOF(
               Returns the adjacency list of the graph where each node is
               represented by an integer in ``[0, n_sites)``
           )EOF")
      .def_property_readonly("is_bipartite", &AbstractGraph::IsBipartite,
                             R"EOF(
               Whether the graph is bipartite.
           )EOF")
      .def_property_readonly("is_connected", &AbstractGraph::IsConnected,
                             R"EOF(
               Whether the graph is connected.
           )EOF")
      .def_property_readonly("distances", &AbstractGraph::AllDistances,
                             R"EOF(
               Returns distances between the nodes. The fact that some node
               may not be reachable from another is represented by -1.
           )EOF")
      .def_property_readonly("symmetry_table", &AbstractGraph::SymmetryTable);

  py::class_<Hypercube, AbstractGraph>(subm, "Hypercube")
      .def(py::init<int, int, bool>(), py::arg("length"), py::arg("n_dim") = 1,
           py::arg("pbc") = true,
           R"EOF(
               Constructs a new ``Hypercube`` given its side length and dimension.

               :param length:
                   side length of the hypercube. It must be always be >=1,
                   but if ``pbc==True`` then the minimal valid length is 3.
               :param n_dim:
                   dimension of the hypercube. It must be at least 1.
               :param pbc:
                   if ``True`` then the constructed hypercube will have periodic
                   boundary conditions, otherwise open boundary conditions are
                   emposed.
           )EOF")
      .def(py::init([](int length, py::iterable xs) {
             auto iterator = xs.attr("__iter__")();
             return Hypercube{length, detail::Iterable2ColorMap(iterator)};
           }),
           py::arg("length"), py::arg("colors"),
           R"EOF(
               Constructs a new `Hypercube` given its side length and edge coloring.

               ``colors`` must be an iterable of ``Tuple[int, int, int]`` where each
               element ``(i, j, c)`` represents an edge ``i <-> j`` of color ``c``.
               Colors must be assigned to __all__ edges.

               :param length:
                   side length of the hypercube. It must be always be >=3 if the
                   hypercube has periodic boundary conditions and >=1 otherwise.
               :param colors:
                   edge colors.
           )EOF");

  py::class_<CustomGraph, AbstractGraph>(subm, "CustomGraph")
      .def(py::init([](py::iterable xs,
                       std::vector<std::vector<int>> automorphisms,
                       bool const is_bipartite) {
             auto iterator = xs.attr("__iter__")();
             return WithEdges(
                 iterator,
                 CustomGraphInit{std::move(automorphisms), is_bipartite});
           }),
           py::arg("edges"),
           py::arg("automorphisms") = std::vector<std::vector<int>>(),
           py::arg("is_bipartite") = false,
           R"EOF(
               Constructs a new graph given a list of edges.

                   * If `edges` has elements of type `Tuple[int, int]` it is treated
                     as a list of edges. Then each element `(i, j)` means a connection
                     between sites `i` and `j`. It is assumed that `0 <= i <= j`. Also,
                     `edges` should contain no duplicates.

                   * If `edges` has elements of type `Tuple[int, int, int]` each
                     element `(i, j, c)` represents an edge between sites `i` and `j`
                     colored into `c`. It is again assumed that `0 <= i <= j` and that
                     there are no duplicate elements in `edges`.
          )EOF");
}

}  // namespace netket

#endif
