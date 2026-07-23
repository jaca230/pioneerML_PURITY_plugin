#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <map>
#include <vector>

namespace py = pybind11;

template <typename T>
py::array_t<T, py::array::c_style | py::array::forcecast> array_from(
    const py::dict& arrays, const char* key) {
  return arrays[py::str(key)].cast<
      py::array_t<T, py::array::c_style | py::array::forcecast>>();
}

py::tuple build_graph(const py::dict& arrays,
                      py::array_t<float, py::array::c_style | py::array::forcecast> norms) {
  auto ao = array_from<std::int64_t>(arrays, "atar_offsets");
  auto lo = array_from<std::int64_t>(arrays, "lyso_offsets");
  if (ao.ndim() != 1 || lo.ndim() != 1 || ao.shape(0) != lo.shape(0)) {
    throw std::runtime_error("invalid or mismatched event offsets");
  }
  const auto rows = ao.shape(0) - 1;
  auto counts = py::array_t<std::int64_t>(rows);
  auto count = counts.mutable_unchecked<1>();
  auto aoff = ao.unchecked<1>();
  auto loff = lo.unchecked<1>();
  std::int64_t total = 0;
  for (py::ssize_t row = 0; row < rows; ++row) {
    count(row) = (aoff(row + 1) - aoff(row)) + (loff(row + 1) - loff(row));
    total += count(row);
  }

  auto out_array = py::array_t<float>({total, std::int64_t{10}});
  auto out = out_array.mutable_unchecked<2>();
  auto norm = norms.unchecked<1>();

  auto ax = array_from<float>(arrays, "atar_x").unchecked<1>();
  auto ay = array_from<float>(arrays, "atar_y").unchecked<1>();
  auto az = array_from<float>(arrays, "atar_z").unchecked<1>();
  auto ae = array_from<float>(arrays, "atar_E").unchecked<1>();
  auto at = array_from<float>(arrays, "atar_t").unchecked<1>();
  auto av = array_from<std::int32_t>(arrays, "atar_view").unchecked<1>();
  auto as = array_from<std::int32_t>(arrays, "atar_slice").unchecked<1>();
  auto ast = array_from<float>(arrays, "atar_slice_mean_t").unchecked<1>();
  auto lx = array_from<float>(arrays, "lyso_x").unchecked<1>();
  auto ly = array_from<float>(arrays, "lyso_y").unchecked<1>();
  auto lz = array_from<float>(arrays, "lyso_z").unchecked<1>();
  auto le = array_from<float>(arrays, "lyso_E").unchecked<1>();
  auto lt = array_from<float>(arrays, "lyso_t").unchecked<1>();
  auto ls = array_from<std::int32_t>(arrays, "lyso_slice").unchecked<1>();
  auto lst = array_from<float>(arrays, "lyso_slice_mean_t").unchecked<1>();

  std::int64_t dst = 0;
  std::vector<std::int64_t> node_slice_id;
  std::vector<std::int64_t> slice_graph_id;
  std::vector<std::int64_t> slice_counts;
  std::vector<std::int64_t> graph_slice_counts;
  node_slice_id.reserve(total);
  std::int64_t graph_idx = 0;
  for (py::ssize_t row = 0; row < rows; ++row) {
    std::map<std::int64_t, std::int64_t> local_slices;
    const auto graph_slice_base = static_cast<std::int64_t>(slice_counts.size());
    auto register_slice = [&](std::int64_t slice_id) {
      auto [it, inserted] = local_slices.try_emplace(slice_id, 0);
      return it;
    };
    for (std::int64_t i = aoff(row); i < aoff(row + 1); ++i) register_slice(as(i));
    for (std::int64_t i = loff(row); i < loff(row + 1); ++i) register_slice(ls(i));
    std::int64_t local_id = 0;
    for (auto& [slice_id, mapped] : local_slices) {
      mapped = local_id++;
      slice_graph_id.push_back(graph_idx);
      slice_counts.push_back(0);
    }
    if (count(row) > 0) {
      graph_slice_counts.push_back(static_cast<std::int64_t>(local_slices.size()));
    }

    for (std::int64_t i = aoff(row); i < aoff(row + 1); ++i, ++dst) {
      for (int col = 0; col < 10; ++col) out(dst, col) = 0.0F;
      const bool yz = av(i) == 1;
      out(dst, 0) = (yz ? 0.0F : ax(i)) / norm(0);
      out(dst, 1) = (yz ? ay(i) : 0.0F) / norm(0);
      out(dst, 2) = az(i) / norm(0);
      out(dst, 3) = ae(i) / norm(1);
      out(dst, 4) = at(i) / norm(2);
      out(dst, 5) = av(i) == 0 ? 1.0F : 0.0F;
      out(dst, 6) = yz ? 1.0F : 0.0F;
      out(dst, 8) = static_cast<float>(as(i));
      out(dst, 9) = ast(i);
      const auto sid = graph_slice_base + local_slices.at(as(i));
      node_slice_id.push_back(sid);
      ++slice_counts[sid];
    }
    for (std::int64_t i = loff(row); i < loff(row + 1); ++i, ++dst) {
      for (int col = 0; col < 10; ++col) out(dst, col) = 0.0F;
      out(dst, 0) = lx(i) / norm(3);
      out(dst, 1) = ly(i) / norm(3);
      out(dst, 2) = lz(i) / norm(3);
      out(dst, 3) = le(i) / norm(4);
      out(dst, 4) = lt(i) / norm(5);
      out(dst, 7) = 1.0F;
      out(dst, 8) = static_cast<float>(ls(i));
      out(dst, 9) = lst(i);
      const auto sid = graph_slice_base + local_slices.at(ls(i));
      node_slice_id.push_back(sid);
      ++slice_counts[sid];
    }
    if (count(row) > 0) ++graph_idx;
  }

  auto vector_array = [](const std::vector<std::int64_t>& values) {
    py::array_t<std::int64_t> result(values.size());
    std::copy(values.begin(), values.end(), result.mutable_data());
    return result;
  };
  return py::make_tuple(
      std::move(out_array), std::move(counts), vector_array(node_slice_id),
      vector_array(slice_graph_id), vector_array(slice_counts),
      vector_array(graph_slice_counts));
}

PYBIND11_MODULE(_purity_loader_native, module) {
  module.doc() = "Experimental native PURITY inference graph builder";
  module.def("build_graph", &build_graph);
}
