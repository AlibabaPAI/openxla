/* Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_ALL_REDUCE_SPLITTER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_ALL_REDUCE_SPLITTER_H_

#include "xla/service/hlo_pass_interface.h"

namespace xla {

class AllReduceSplitter : public HloModulePass {
 public:
  explicit AllReduceSplitter(int64_t combine_threshold_in_bytes,
                                int64_t combine_threshold_count)
      : combine_threshold_in_bytes_(combine_threshold_in_bytes),
        combine_threshold_count_(combine_threshold_count) {}
  absl::string_view name() const override { return "all-reduce-splitter"; }

  using HloPassInterface::Run;
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  int64_t combine_threshold_in_bytes_;
  int64_t combine_threshold_count_;

  int64_t GetCombineThreshloldInBytes();
  int64_t GetCombineThreshloldCount();
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_ALL_REDUCE_SPLITTER_H_
