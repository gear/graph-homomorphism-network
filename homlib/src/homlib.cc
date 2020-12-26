#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "hom.hh"

using namespace pybind11;

PYBIND11_MODULE(homlib, m) {
    m.doc() = "homomorphism counting library";
    class_<Graph>(m, "Graph")
        .def(init<int>())
        .def("addEdge", &Graph::addEdge);
    m.def("hom", &hom<double>, "homomorphism counting function (double)");
    m.def("homint", &hom<long long>, "homomorphism counting function (int64_t)");
}
