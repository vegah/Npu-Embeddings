//===- npu_device.cpp ---------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- XRT dispatch. See npu_device.hpp.
// SPDX-License-Identifier: Apache-2.0
//
// The buffer-object group ids are not arbitrary: for the mlir-aie DPU kernel,
// argument 1 is the instruction stream and arguments 3.. are the data buffers.
// xrt::kernel::group_id(i) maps those to memory banks, and getting it wrong
// produces a silent wrong answer rather than an error.
//
// Every host write is followed by an explicit sync to the device and every read
// by a sync from it. research/notes/0003 records what happens otherwise on the
// Python side: writes that never reached the device, with only the first
// dispatch looking right.

#include "npu_device.hpp"

#include <chrono>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

namespace npu {
namespace {

double now_s() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::vector<uint8_t> read_file(const std::string &path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::streamsize n = f.tellg();
  f.seekg(0);
  std::vector<uint8_t> buf(static_cast<size_t>(n));
  if (!f.read(reinterpret_cast<char *>(buf.data()), n))
    throw std::runtime_error("cannot read " + path);
  return buf;
}

// Minimal field extraction from design.json. Same reasoning as the .npue
// directory scanner: the file is written by our own build step, so a
// dependency-free reader is the right size of tool for it.
size_t find_key(const std::string &t, const std::string &key) {
  size_t i = t.find("\"" + key + "\"");
  if (i == std::string::npos)
    throw std::runtime_error("design.json: missing " + key);
  return t.find(':', i) + 1;
}

int64_t json_int(const std::string &t, const std::string &key,
                 int64_t fallback, bool required = false) {
  size_t i = t.find("\"" + key + "\"");
  if (i == std::string::npos) {
    if (required) throw std::runtime_error("design.json: missing " + key);
    return fallback;
  }
  return std::stoll(t.substr(t.find(':', i) + 1));
}

std::string json_str(const std::string &t, const std::string &key,
                     const std::string &fallback) {
  size_t i = t.find("\"" + key + "\"");
  if (i == std::string::npos) return fallback;
  i = t.find('"', t.find(':', i) + 1) + 1;
  return t.substr(i, t.find('"', i) - i);
}

std::vector<size_t> json_size_array(const std::string &t,
                                    const std::string &key) {
  std::vector<size_t> out;
  size_t i = t.find("\"" + key + "\"");
  if (i == std::string::npos) return out;
  i = t.find('[', i) + 1;
  while (true) {
    while (i < t.size() && (t[i] == ' ' || t[i] == '\n' || t[i] == ',')) ++i;
    if (i >= t.size() || t[i] == ']') break;
    out.push_back(static_cast<size_t>(std::stoull(t.substr(i))));
    while (i < t.size() && t[i] != ',' && t[i] != ']') ++i;
  }
  return out;
}

}  // namespace

struct Device::Impl {
  xrt::device device{0};
};

Device::Device() : impl_(std::make_unique<Impl>()) {}
Device::~Device() = default;

struct Design::Impl {
  xrt::hw_context ctx;
  xrt::kernel kernel;
  xrt::bo bo_instr;
  std::vector<xrt::bo> alt_instr;        // slots 1..n
  std::vector<size_t> alt_instr_words;
  size_t active_instr = 0;
  std::vector<xrt::bo> bos;               // slot 0 for each argument
  std::vector<std::vector<xrt::bo>> alt;  // per argument, slots 1..n
  std::vector<size_t> active;             // per argument, which slot is bound
  size_t n_instr_words = 0;

  // Every access goes through here, so host_ptr/sync/dispatch can never
  // disagree about which buffer is live.
  xrt::bo &live(size_t i) {
    size_t s = active.at(i);
    return s == 0 ? bos.at(i) : alt.at(i).at(s - 1);
  }
};

Design::Design(Device &dev, const std::string &dir)
    : impl_(std::make_unique<Impl>()) {
  std::ifstream jf(dir + "/design.json");
  if (!jf) throw std::runtime_error("cannot open " + dir + "/design.json");
  std::stringstream ss;
  ss << jf.rdbuf();
  const std::string js = ss.str();

  info_.name = json_str(js, "name", dir);
  info_.kind = json_str(js, "kind", "gemm");
  info_.M = json_int(js, "M", 0);
  info_.K = json_int(js, "K", 0);
  info_.N = json_int(js, "N", 0);
  info_.b_layout_hash = json_str(js, "b_layout_hash", "");

  info_.buffer_bytes = json_size_array(js, "buffers");
  if (info_.buffer_bytes.empty()) {
    // The GEMM designs describe their buffers individually.
    info_.buffer_bytes = {
        static_cast<size_t>(json_int(js, "bytes_a", 0, true)),
        static_cast<size_t>(json_int(js, "bytes_b", 0, true)),
        static_cast<size_t>(json_int(js, "bytes_c", 0, true))};
  }
  if (info_.buffer_bytes.size() < 2)
    throw std::runtime_error(dir + ": fewer than two buffers");
  output_index_ = info_.buffer_bytes.size() - 1;   // output is always last

  auto &d = *dev.impl();
  auto xclbin = xrt::xclbin(dir + "/final.xclbin");
  auto uuid = d.device.register_xclbin(xclbin);
  impl_->ctx = xrt::hw_context(d.device, uuid);

  std::string kname;
  for (const auto &k : xclbin.get_kernels())
    if (k.get_name().rfind("MLIR_AIE", 0) == 0) { kname = k.get_name(); break; }
  if (kname.empty())
    throw std::runtime_error(dir + ": no MLIR_AIE kernel in the xclbin");
  impl_->kernel = xrt::kernel(impl_->ctx, kname);

  auto instr = read_file(dir + "/insts.bin");
  impl_->n_instr_words = instr.size() / sizeof(uint32_t);
  impl_->bo_instr = xrt::bo(d.device, instr.size(), XCL_BO_FLAGS_CACHEABLE,
                            impl_->kernel.group_id(1));
  std::memcpy(impl_->bo_instr.map<void *>(), instr.data(), instr.size());
  impl_->bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // Data buffers start at kernel argument 3.
  for (size_t i = 0; i < info_.buffer_bytes.size(); ++i)
    impl_->bos.emplace_back(d.device, info_.buffer_bytes[i],
                            XRT_BO_FLAGS_HOST_ONLY,
                            impl_->kernel.group_id(static_cast<int>(3 + i)));
  impl_->alt.resize(info_.buffer_bytes.size());
  impl_->active.assign(info_.buffer_bytes.size(), 0);
  dev_ = &dev;
}

Design::~Design() = default;

size_t Design::load_instr(const std::string &path) {
  auto &d = *dev_->impl();
  auto instr = read_file(path);
  xrt::bo bo(d.device, instr.size(), XCL_BO_FLAGS_CACHEABLE,
             impl_->kernel.group_id(1));
  std::memcpy(bo.map<void *>(), instr.data(), instr.size());
  bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  impl_->alt_instr.push_back(std::move(bo));
  impl_->alt_instr_words.push_back(instr.size() / sizeof(uint32_t));
  return impl_->alt_instr.size();       // slot 0 is the original stream
}

void Design::bind_instr(size_t slot) {
  if (slot > impl_->alt_instr.size())
    throw std::runtime_error(info_.name + ": no instr slot " +
                             std::to_string(slot));
  impl_->active_instr = slot;
}

size_t Design::stage(size_t arg_index, const void *data, size_t bytes) {
  // A LARGER buffer than the design declares would mean DMA descriptors
  // reading past what the caller staged -- refuse. SMALLER is legitimate for
  // the unified gemm_rtp design, whose declared sizes are the maximum over
  // its streams: each stream's descriptors only address that stream's bytes.
  if (bytes > info_.buffer_bytes.at(arg_index))
    throw std::runtime_error(info_.name + ": staged buffer for argument " +
                             std::to_string(arg_index) + " is " +
                             std::to_string(bytes) + " bytes, design allows " +
                             std::to_string(info_.buffer_bytes[arg_index]));
  auto &d = *dev_->impl();
  auto &slots = impl_->alt.at(arg_index);
  slots.emplace_back(d.device, bytes, XRT_BO_FLAGS_HOST_ONLY,
                     impl_->kernel.group_id(static_cast<int>(3 + arg_index)));
  std::memcpy(slots.back().map<void *>(), data, bytes);
  slots.back().sync(XCL_BO_SYNC_BO_TO_DEVICE);
  return slots.size();     // slot 0 is the design's own buffer
}

size_t Design::stage_alloc(size_t arg_index, size_t bytes) {
  if (bytes > info_.buffer_bytes.at(arg_index))
    throw std::runtime_error(info_.name + ": stage_alloc for argument " +
                             std::to_string(arg_index) + " is " +
                             std::to_string(bytes) + " bytes, design allows " +
                             std::to_string(info_.buffer_bytes[arg_index]));
  auto &d = *dev_->impl();
  auto &slots = impl_->alt.at(arg_index);
  slots.emplace_back(d.device, bytes, XRT_BO_FLAGS_HOST_ONLY,
                     impl_->kernel.group_id(static_cast<int>(3 + arg_index)));
  return slots.size();
}

void *Design::slot_ptr(size_t arg_index, size_t slot) {
  return (slot == 0 ? impl_->bos.at(arg_index)
                    : impl_->alt.at(arg_index).at(slot - 1))
      .map<void *>();
}

void Design::bind(size_t arg_index, size_t slot) {
  if (slot > impl_->alt.at(arg_index).size())
    throw std::runtime_error(info_.name + ": no staged slot " +
                             std::to_string(slot));
  impl_->active.at(arg_index) = slot;
}

void *Design::host_ptr(size_t index) {
  return impl_->live(index).map<void *>();
}

void Design::sync_to_device(size_t index, size_t bytes) {
  xrt::bo &b = impl_->live(index);
  if (bytes == 0) bytes = b.size();
  b.sync(XCL_BO_SYNC_BO_TO_DEVICE, bytes, 0);
}

void Design::sync_from_device(size_t index, size_t bytes) {
  xrt::bo &b = impl_->live(index);
  if (bytes == 0) bytes = b.size();
  b.sync(XCL_BO_SYNC_BO_FROM_DEVICE, bytes, 0);
}

void Design::dispatch_only() {
  const double t_enter = now_s();
  const size_t ai = impl_->active_instr;
  xrt::bo &instr = ai == 0 ? impl_->bo_instr : impl_->alt_instr[ai - 1];
  const uint32_t n_words = static_cast<uint32_t>(
      ai == 0 ? impl_->n_instr_words : impl_->alt_instr_words[ai - 1]);
  // opcode 3 is "run a DPU instruction sequence" in the mlir-aie host contract.
  xrt::run r;
  switch (impl_->bos.size()) {
    case 2:
      r = impl_->kernel(3, instr, n_words, impl_->live(0), impl_->live(1));
      break;
    case 3:
      r = impl_->kernel(3, instr, n_words, impl_->live(0), impl_->live(1),
                        impl_->live(2));
      break;
    case 4:
      r = impl_->kernel(3, instr, n_words, impl_->live(0), impl_->live(1),
                        impl_->live(2), impl_->live(3));
      break;
    default:
      throw std::runtime_error(info_.name + ": unsupported buffer count");
  }
  // Submit and wait are charged separately: 0010 measured ~150 us of hardware
  // per dispatch and this path costs 1313 us, so the question "is the command
  // expensive to build, or is the kernel actually slow?" decides what to fix.
  t_submit += now_s() - t_enter;
  double t_wait0 = now_s();
  auto state = r.wait();
  t_wait += now_s() - t_wait0;
  ++n_dispatch;
  if (state != ERT_CMD_STATE_COMPLETED)
    throw std::runtime_error(info_.name + ": kernel state " +
                             std::to_string(static_cast<int>(state)));
}

void Design::run(const std::vector<const void *> &inputs, void *output) {
  if (inputs.size() != output_index_)
    throw std::runtime_error(info_.name + ": expected " +
                             std::to_string(output_index_) + " inputs");
  for (size_t i = 0; i < inputs.size(); ++i) {
    std::memcpy(host_ptr(i), inputs[i], info_.buffer_bytes[i]);
    sync_to_device(i);
  }
  dispatch_only();
  sync_from_device(output_index_);
  std::memcpy(output, host_ptr(output_index_),
              info_.buffer_bytes[output_index_]);
}

}  // namespace npu
