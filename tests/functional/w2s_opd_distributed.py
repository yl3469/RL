# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""W2S-OPD value, gradient, and teacher post-processors under TP2 x CP2.

Needs four GPUs and the mcore extra:

    uv run --extra mcore torchrun --nproc_per_node=4 \
        tests/functional/w2s_opd_distributed.py

Checks the student loss for both transports against a dense oracle through the
production ``LossPostProcessor`` (CP normalization included), that only TP0/CP0
keeps a teacher payload, and that the negative teacher's
``W2SProxyPostProcessor`` publishes the exact proxy support.
"""

import os
from unittest.mock import patch

import torch
import torch.distributed as dist

from nemo_rl.algorithms.loss import ClippedPGLossConfig, ClippedPGLossFn
from nemo_rl.algorithms.opd import W2SOPDConfig
from nemo_rl.algorithms.w2s_opd import (
    w2s_compact_proxy,
    w2s_reverse_kl_from_hidden,
)
from nemo_rl.data_plane.schema import W2S_COMPONENT_FIELDS, W2S_PROXY_FIELDS
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.model_utils import _get_tokens_on_this_cp_rank


def main():
    use_cuda = torch.cuda.is_available()
    device = (
        torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        if use_cuda
        else torch.device("cpu")
    )
    if use_cuda:
        torch.cuda.set_device(device)
    dist.init_process_group("nccl" if use_cuda else "gloo")
    rank = dist.get_rank()
    assert dist.get_world_size() == 4, "This validation requires exactly four ranks"
    tp_groups = [dist.new_group([0, 1]), dist.new_group([2, 3])]
    cp_groups = [dist.new_group([0, 2]), dist.new_group([1, 3])]
    tp_rank, cp_rank = rank % 2, rank // 2
    tp_group, cp_group = tp_groups[cp_rank], cp_groups[tp_rank]
    torch.manual_seed(703)
    # Vocabulary 11 padded to 12, odd teacher prefix padded to 8 for CP.
    dense_student = torch.randn(2, 8, 12, device=device, requires_grad=True)
    hidden = [torch.randn(2, 7, h, device=device) for h in (3, 5, 5)]
    heads = [torch.randn(12, h, device=device) for h in (3, 5, 5)]
    padded_hidden = [torch.nn.functional.pad(h, (0, 0, 0, 1)) for h in hidden]
    projected = [h @ w.t() for h, w in zip(padded_hidden, heads)]
    proxy = projected[0] + 0.75 * (projected[1] - projected[2])
    ids = proxy[..., :11].topk(4, dim=-1).indices
    p = dense_student.gather(-1, ids).softmax(-1)
    q = proxy.gather(-1, ids).softmax(-1)
    oracle = (p * (p.log() - q.log())).sum(-1)
    # Distinct weights reveal incorrect CP reconstruction/order and masking.
    mask = torch.tensor(
        [
            [1.0, 0.0, 0.2, 1.0, 0.4, 0.0, 1.0, 0.0],
            [0.0, 0.3, 1.0, 0.0, 1.0, 0.5, 0.7, 0.0],
        ],
        device=device,
    )
    (oracle * mask).sum().backward()
    local_logits = (
        _get_tokens_on_this_cp_rank(
            dense_student.detach()[..., tp_rank * 6 : (tp_rank + 1) * 6],
            cp_rank,
            2,
            seq_dim=1,
        )
        .clone()
        .requires_grad_()
    )
    local_heads = tuple(h[tp_rank * 6 : (tp_rank + 1) * 6] for h in heads)
    actual = w2s_reverse_kl_from_hidden(
        local_logits,
        tuple(hidden),
        local_heads,
        alpha=0.75,
        top_k=4,
        vocab_size=11,
        chunk_size=2,
        vocab_parallel_group=tp_group,
        context_parallel_group=cp_group,
    )
    torch.testing.assert_close(actual, oracle, atol=6e-6, rtol=6e-6)
    # Exercise the production loss preparation and CP normalization. The raw
    # CP gather sums gradients across its replicas; LossPostProcessor cancels
    # that replication before Megatron's schedule applies its own scaling.
    from nemo_rl.models.megatron import train

    expected_gradient = _get_tokens_on_this_cp_rank(
        dense_student.grad[..., tp_rank * 6 : (tp_rank + 1) * 6], cp_rank, 2, seq_dim=1
    )
    support, target = w2s_compact_proxy(
        tuple(hidden),
        tuple(heads),
        alpha=0.75,
        top_k=4,
        vocab_size=11,
        chunk_size=2,
        vocab_parallel_group=None,
    )
    columns_by_transport = {
        "hidden_states": dict(zip(W2S_COMPONENT_FIELDS, hidden, strict=True)),
        "proxy_support": dict(
            zip(W2S_PROXY_FIELDS, (support.to(torch.int32), target), strict=True)
        ),
    }
    for transport, columns in columns_by_transport.items():
        config = W2SOPDConfig(
            enabled=True,
            anchor_model="/frozen/student",
            alpha=0.75,
            top_k=4,
            vocab_size=11,
            chunk_size=2,
            transport=transport,
        )
        loss_fn = ClippedPGLossFn(
            ClippedPGLossConfig(disable_ppo_ratio=True, reference_policy_kl_penalty=0),
            opd_full=config,
        )
        data = BatchedDataDict(
            {
                "input_ids": torch.zeros(2, 8, dtype=torch.long, device=device),
                "token_mask": torch.cat(
                    [torch.zeros(2, 1, device=device), mask[:, :-1]], dim=1
                ),
                "sample_mask": torch.ones(2, device=device),
                **columns,
            }
        )
        logits = local_logits.detach().clone().requires_grad_()
        microbatches = 3
        with (
            patch.object(train, "get_context_parallel_world_size", return_value=2),
            patch.object(train, "get_context_parallel_group", return_value=cp_group),
            patch.object(train, "get_tensor_model_parallel_rank", return_value=tp_rank),
            patch.object(
                train, "get_tensor_model_parallel_group", return_value=tp_group
            ),
        ):
            post = train.LossPostProcessor(
                loss_fn,
                {"sequence_packing": {"enabled": False}},
                num_microbatches=microbatches,
                teacher_output_layer_weight_by_index=(
                    dict(enumerate(local_heads))
                    if transport == "hidden_states"
                    else None
                ),
            )
            wrapped = post(
                data,
                global_valid_seqs=torch.tensor(1.0, device=device),
                global_valid_toks=torch.tensor(1.0, device=device),
            )
            loss, metrics = wrapped(logits)
            # Match the schedule's cp_size / num_microbatches factor, which the
            # postprocessor explicitly counteracts in train.py.
            effective_loss = loss * 2 / microbatches
            torch.testing.assert_close(
                effective_loss * 2, (oracle * mask).sum(), atol=6e-6, rtol=6e-6
            )
            assert (
                abs(metrics["w2s_opd_reverse_kl"] - (oracle * mask).sum().item()) < 2e-5
            )
            effective_loss.backward()
        torch.testing.assert_close(logits.grad, expected_gradient, atol=6e-6, rtol=6e-6)
    # Every TP/CP rank joins reconstruction, but only TP0/CP0 keeps
    # the CPU payload. Exercise two packed sequences with distinct boundaries.
    packed_local = torch.cat(
        [
            _get_tokens_on_this_cp_rank(h[None], cp_rank, 2, seq_dim=1)
            for h in padded_hidden[0]
        ],
        dim=1,
    )
    fake_cfg = {
        "sequence_packing": {"enabled": True},
        "megatron_cfg": {"context_parallel_size": 2},
    }

    def scalar_processor(**kwargs):
        return lambda output: (
            output.new_zeros(()),
            {"logprobs": output.new_zeros((2, 7))},
        )

    with (
        patch.object(train, "get_context_parallel_group", return_value=cp_group),
        patch.object(train, "get_context_parallel_rank", return_value=cp_rank),
        patch.object(train, "get_tensor_model_parallel_rank", return_value=tp_rank),
        patch.object(train, "LogprobsPostProcessor", return_value=scalar_processor),
    ):
        processor = train.TeacherFullPayloadPostProcessor(
            fake_cfg, "hidden_states", torch.float32
        )
        run = processor(
            BatchedDataDict(
                {
                    "input_ids": torch.zeros(2, 7, dtype=torch.long, device=device),
                    "input_lengths": torch.tensor([7, 7], device=device),
                }
            ),
            torch.zeros(1, 8, dtype=torch.long, device=device),
            torch.tensor([0, 8, 16], dtype=torch.int32, device=device),
            7,
            hidden_states=packed_local.transpose(0, 1),
        )
        _, outputs = run(torch.zeros(1, 8, 6, device=device))
        if tp_rank == 0 and cp_rank == 0:
            torch.testing.assert_close(outputs["teacher_full_payload"], hidden[0].cpu())
        else:
            assert outputs["teacher_full_payload"] is None

    # Compact transport: the negative teacher folds anchor/positive rows (from
    # the arm source) with its own captured hidden states into the proxy
    # support. Two packed rows of different lengths; only TP0/CP0 keeps it.
    lengths, spans = [7, 5], [(0, 8), (8, 16)]
    arm_rows = [
        torch.cat([hidden[i][0], hidden[i][1, :5]]) for i in (0, 1)
    ]  # [12, H] true-length rows back to back
    negative_full = torch.zeros(1, 16, 5, device=device)
    negative_full[0, :7] = hidden[2][0]
    negative_full[0, 8:13] = hidden[2][1, :5]
    negative_local = torch.cat(
        [
            _get_tokens_on_this_cp_rank(negative_full[:, a:b], cp_rank, 2, seq_dim=1)
            for a, b in spans
        ],
        dim=1,
    )
    seen = []

    def arm_source(data_dict):
        seen.append(data_dict["input_lengths"].tolist())
        return arm_rows[0], arm_rows[1]

    with (
        patch.object(train, "get_context_parallel_group", return_value=cp_group),
        patch.object(train, "get_context_parallel_rank", return_value=cp_rank),
        patch.object(train, "get_tensor_model_parallel_rank", return_value=tp_rank),
        patch.object(train, "get_tensor_model_parallel_group", return_value=tp_group),
        patch.object(train, "LogprobsPostProcessor", return_value=scalar_processor),
    ):
        processor = train.W2SProxyPostProcessor(
            fake_cfg,
            heads=local_heads,
            alpha=0.75,
            top_k=4,
            vocab_size=11,
            chunk_size=3,
            payload_dtype=torch.float32,
            arm_source=arm_source,
        )
        run = processor(
            BatchedDataDict(
                {
                    "input_ids": torch.zeros(2, 7, dtype=torch.long, device=device),
                    "input_lengths": torch.tensor(lengths, device=device),
                }
            ),
            torch.zeros(1, 8, dtype=torch.long, device=device),
            torch.tensor([0, 8, 16], dtype=torch.int32, device=device),
            7,
            hidden_states=negative_local.transpose(0, 1),
        )
        _, outputs = run(torch.zeros(1, 8, 6, device=device))
        assert seen == [lengths]
        if tp_rank == 0 and cp_rank == 0:
            support, target = outputs["w2s_proxy"]
            assert support.dtype == torch.int32 and support.shape == (2, 7, 4)
            for row, length in enumerate(lengths):
                arms = [h[row, :length] for h in hidden]
                dense = [a @ w.t() for a, w in zip(arms, heads)]
                values, ids = (dense[0] + 0.75 * (dense[1] - dense[2]))[:, :11].topk(
                    4, dim=-1
                )
                assert torch.equal(support[row, :length].long(), ids.cpu())
                torch.testing.assert_close(
                    target[row, :length], values.log_softmax(-1).cpu()
                )
                assert support[row, length:].count_nonzero() == 0
        else:
            assert outputs["w2s_proxy"] is None
    dist.barrier()
    if rank == 0:
        print("W2S_OPD_TP2_CP2_VALUE_GRADIENT_PASS", flush=True)
        print("W2S_PROXY_POSTPROCESSOR_TP2_CP2_PASS", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
