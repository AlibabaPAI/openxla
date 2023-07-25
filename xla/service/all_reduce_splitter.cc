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

#include "xla/service/all_reduce_splitter.h"

#include "absl/container/flat_hash_map.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/service/computation_placer.h"
#include "xla/service/shape_inference.h"

namespace xla {

namespace {

constexpr char kCombineThresholdInBytes[] =
    "XLA_GPU_ALL_REDUCE_COMBINE_THRESHOLD_IN_BYTES";
constexpr char kCombineThresholdCount[] =
    "XLA_GPU_ALL_REDUCE_COMBINE_THRESHOLD_COUNT";

StatusOr<bool> CombineAllReduces(
    absl::Span<HloInstruction* const> to_combine,
    std::vector<HloInstruction*>& result_recorder) {
  VLOG(1) << "Combined " << to_combine.size() << " CRS ops";
  if (to_combine.size() < 2) {
    if (!to_combine.empty()) {
      result_recorder.push_back(to_combine.front());
    }
    return true;
  }

  HloComputation& computation = *to_combine.back()->parent();
  HloComputation* reduction = to_combine[0]->to_apply();
  const HloOpcode type = reduction->root_instruction()->opcode();

  // Create a single bigger AllReduce of the operands of the smaller
  // AllReduces.
  std::vector<HloInstruction*> operands;
  std::vector<Shape> operand_shapes;
  VLOG(1) << "Combining set";
  for (HloInstruction* hlo : to_combine) {
    VLOG(1) << "Set element: " << hlo->ToString();
    TF_RET_CHECK(hlo->opcode() == HloOpcode::kAllReduce);
    TF_RET_CHECK(hlo->operands().size() == 1);
    TF_RET_CHECK(hlo->to_apply() == reduction ||
                 (hlo->to_apply()->instruction_count() == 3 &&
                  hlo->to_apply()->num_parameters() == 2 &&
                  hlo->to_apply()->root_instruction()->opcode() == type));
    TF_RET_CHECK(hlo->shape().IsArray());
    for (HloInstruction* operand : hlo->operands()) {
      operands.push_back(operand);
      operand_shapes.push_back(operand->shape());
    }
  }

  HloInstruction* combined;
  // AllReduce ops with more than one operand produce a tuple.
  TF_RET_CHECK(operands.size() >= 2);
  combined = computation.AddInstruction(HloInstruction::CreateAllReduce(
      ShapeUtil::MakeTupleShape(operand_shapes), operands, reduction,
      to_combine.front()->replica_groups(),
      /*constrain_layout=*/true, to_combine.front()->channel_id(),
      Cast<HloAllReduceInstruction>(to_combine.front())
          ->use_global_device_ids()));
  result_recorder.push_back(combined);

  // We have to propagate the sharding manually because Domain instructions are
  // not guaranteed to preserve it for side effecting instructions.
  if (to_combine.front()->has_sharding()) {
    combined->set_sharding(to_combine.front()->sharding());
  }
  VLOG(1) << "Replacing with : " << combined->ToString();

  // Replace all the smaller AllReduces with elements of the tuple output
  // of the single bigger AllReduce.
  for (int64_t i = 0; i < to_combine.size(); ++i) {
    auto replace_with = HloInstruction::CreateGetTupleElement(
        to_combine[i]->shape(), combined, i);
    TF_RETURN_IF_ERROR(computation.ReplaceWithNewInstruction(
        to_combine[i], std::move(replace_with)));
  }
  return true;
}

std::vector<std::vector<HloInstruction*>> BreakDownAllReduce(
    int64_t combine_threshold_in_bytes, int64_t combine_threshold_count,
    const std::vector<HloInstruction*>& all_reduce_to_break_down) {
  bool scale_first = true;
  int64_t combined_in_bytes = 0;
  int64_t combined_count = 0;
  std::vector<HloInstruction*> to_combine;
  std::vector<std::vector<HloInstruction*>> to_combine_vector;

  for (HloInstruction* to_combined_all_reduce : all_reduce_to_break_down) {
    int64_t instruction_bytes =
        ShapeUtil::ByteSizeOf(to_combined_all_reduce->shape());
    VLOG(1) << " Got " << to_combined_all_reduce->ToString()
            << " with size in bytes: " << instruction_bytes;
    if (to_combine.empty()) {
      to_combine.push_back(to_combined_all_reduce);
      combined_in_bytes += instruction_bytes;
      combined_count += 1;
      continue;
    }

    if (combined_in_bytes + instruction_bytes >
            combine_threshold_in_bytes / (scale_first ? 2 : 1) ||
        combined_count + 1 > combine_threshold_count / (scale_first ? 2 : 1)) {
      scale_first = false;
      to_combine_vector.push_back(to_combine);
      to_combine.clear();
      combined_in_bytes = 0;
      combined_count = 0;
    }
    to_combine.push_back(to_combined_all_reduce);
    combined_in_bytes += instruction_bytes;
    combined_count += 1;
  }

  if (!to_combine.empty()) {
    to_combine_vector.push_back(to_combine);
  }

  return to_combine_vector;
}

}  // namespace

int64_t AllReduceSplitter::GetCombineThreshloldInBytes() {
  int64_t value;
  char* env = std::getenv(kCombineThresholdInBytes);
  if (env != nullptr && absl::SimpleAtoi(env, &value)) {
    VLOG(1) << "combine_threshold_in_bytes_ has been set to " << value << " by "
            << kCombineThresholdInBytes;
    combine_threshold_in_bytes_ = value;
  }
  return combine_threshold_in_bytes_;
}

int64_t AllReduceSplitter::GetCombineThreshloldCount() {
  int64_t value;
  char* env = std::getenv(kCombineThresholdCount);
  if (env != nullptr && absl::SimpleAtoi(env, &value)) {
    VLOG(1) << "combine_threshold_count_ has been set to " << value << " by "
            << kCombineThresholdCount;
    combine_threshold_count_ = value;
  }
  return combine_threshold_count_;
}

StatusOr<bool> AllReduceSplitter::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation : module->MakeNonfusionComputations()) {
    // Find all all-reduce ops first as we can't modify the instructions while
    // iterating through them.
    std::vector<HloInstruction*> found_allreduces;

    for (HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      if (instruction->opcode() == HloOpcode::kAllReduce) {
        found_allreduces.push_back(instruction);
      }
    }
    if (found_allreduces.empty()) {
      continue;
    }

    for (HloInstruction* instruction : found_allreduces) {
      if (HloAllReduceInstruction* ar =
              DynCast<HloAllReduceInstruction>(instruction)) {
        if (ar->operand_count() < 2) {
          continue;
        }
        // Split allreduce with multiple operands into multiple allreduce with
        // single operand.
        std::vector<HloInstruction*> split_all_reduce_in_order;
        for (size_t i = 0; i < ar->operand_count(); i++) {
          HloInstruction* split_ar =
              computation->AddInstruction(HloInstruction::CreateAllReduce(
                  ar->operand(i)->shape(), {ar->mutable_operand(i)},
                  ar->to_apply(), ar->replica_groups(), ar->constrain_layout(),
                  ar->channel_id(), ar->use_global_device_ids()));
          split_ar->set_metadata(ar->metadata());
          split_ar->set_raw_backend_config_string(
              ar->raw_backend_config_string());
          split_all_reduce_in_order.push_back(split_ar);
        }

        changed = true;

        HloInstruction* result_tuple = computation->AddInstruction(
            HloInstruction::CreateTuple(split_all_reduce_in_order));

        TF_RETURN_WITH_CONTEXT_IF_ERROR(
            computation->ReplaceInstruction(ar, result_tuple),
            "replacing with result tuple", ar->ToShortString());

        // Combine allreduce in a specific order to reduce communication launch
        // overhead. The native combiner uses post order to combine, which
        // causes the computational communication to not overlap well.

        // TODO(lyh271596): DDP also uses the same order (simply reversing
        // model.parameter), however there should be room for improvement:
        // https://pytorch.org/docs/stable/notes/ddp.html#internal-design
        std::reverse(split_all_reduce_in_order.begin(),
                     split_all_reduce_in_order.end());

        std::vector<HloInstruction*> combined_all_reduce_in_order;
        int64_t combine_threshold_in_bytes = GetCombineThreshloldInBytes();
        int64_t combine_threshold_count = GetCombineThreshloldCount();

        std::vector<std::vector<HloInstruction*>> all_combined_group =
            BreakDownAllReduce(combine_threshold_in_bytes,
                               combine_threshold_count,
                               split_all_reduce_in_order);

        std::vector<HloInstruction*> last_combined_group =
            all_combined_group.back();
        std::reverse(last_combined_group.begin(), last_combined_group.end());
        all_combined_group.pop_back();

        std::vector<std::vector<HloInstruction*>>
            all_combined_re_group_for_last = BreakDownAllReduce(
                combine_threshold_in_bytes, combine_threshold_count,
                last_combined_group);

        for (std::vector<HloInstruction*>& all_reduce_group :
             all_combined_re_group_for_last) {
          std::reverse(all_reduce_group.begin(), all_reduce_group.end());
        }

        all_combined_group.insert(all_combined_group.end(),
                                  all_combined_re_group_for_last.rbegin(),
                                  all_combined_re_group_for_last.rend());

        for (std::vector<HloInstruction*> all_reduce_group :
             all_combined_group) {
          auto status =
              CombineAllReduces(all_reduce_group, combined_all_reduce_in_order);
        }

        // Convert to an asynchronous allreduce op.
        std::vector<HloInstruction*> all_reduce_start_in_order;

        for (HloInstruction* ar_to_async : combined_all_reduce_in_order) {
          if (HloAllReduceInstruction* ar_before_async =
                  DynCast<HloAllReduceInstruction>(ar_to_async)) {
            HloInstruction* start = computation->AddInstruction(
                HloInstruction::CreateAllReduceStart(
                    ar_before_async->shape(), ar_before_async->operands(),
                    ar_before_async->to_apply(),
                    ar_before_async->replica_groups(),
                    ar_before_async->constrain_layout(),
                    ar_before_async->channel_id(),
                    ar_before_async->use_global_device_ids()));
            std::unique_ptr<HloInstruction> done = HloInstruction::CreateUnary(
                ar_before_async->shape(), HloOpcode::kAllReduceDone, start);
            start->set_metadata(ar_before_async->metadata());
            start->set_raw_backend_config_string(
                ar_before_async->raw_backend_config_string());
            all_reduce_start_in_order.push_back(start);
            TF_RETURN_WITH_CONTEXT_IF_ERROR(
                computation->ReplaceWithNewInstruction(ar_before_async,
                                                       std::move(done)),
                "replacing in async ", ar_before_async->ToShortString());
          }
        }

        // Ordering AllReduceStart
        HloCollectiveInstruction* prev = nullptr;
        for (HloInstruction* ar_to_add_control_dep :
             all_reduce_start_in_order) {
          auto* next = DynCast<HloCollectiveInstruction>(ar_to_add_control_dep);
          if (prev != nullptr) {
            TF_CHECK_OK(prev->AddControlDependencyTo(next));
            VLOG(1) << "Adding control dependency from " << prev->ToString()
                    << " to " << next->ToString();
          }
          prev = next;
        }

        // Ensure that users of single AllReduce scheduled after the end of the
        // overall AllReduce to maximum overlapping.

        // TODO(lyh271596):There is no overlapping between allreduce and
        // optimizer step under the existing implementation, so there is some
        // room for improvement.
        std::vector<HloInstruction*> all_reduce_end_in_order;
        for (HloInstruction* sub_ar : all_reduce_start_in_order) {
          all_reduce_end_in_order.push_back(sub_ar->users()[0]);
        }
        HloInstruction* all_all_reduce_token = computation->AddInstruction(
            HloInstruction::CreateAfterAll(all_reduce_end_in_order));

        for (HloInstruction* sub_ar : all_reduce_start_in_order) {
          HloInstruction* ar_done = sub_ar->users()[0];
          HloInstruction* ar_done_after_all =
              computation->AddInstruction(HloInstruction::CreateAddDependency(
                  ar_done, all_all_reduce_token));
          for (auto ar_done_user : ar_done->users()) {
            if (ar_done_user != ar_done_after_all &&
                ar_done_user != all_all_reduce_token) {
              auto ar_done_as_user_index = ar_done_user->operand_index(ar_done);
              TF_CHECK_OK(ar_done_user->ReplaceOperandWith(
                  ar_done_as_user_index, ar_done_after_all));
            }
          }
        }
      }
    }
  }
  return changed;
}

}  // namespace xla
