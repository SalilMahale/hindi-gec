# -*- coding: utf-8 -*-
"""Fine-tune IndicBART as a Hindi corrector (PLAN §4.3).

The task as seq2seq
-------------------
Grammar correction is treated as "translation" from erroneous Hindi into
correct Hindi. IndicBART is an encoder–decoder: the encoder reads the source
sentence, the decoder writes the corrected sentence one sub-word at a time,
attending to the encoder. Most output tokens are copies of input tokens; the
model has to learn *when not to copy*.

Training signal
---------------
For target ids  t = [<2hi>, w1 … wn, </s>]  the decoder reads t[:-1] and is
trained to predict t[1:] (teacher forcing). The loss is the cross-entropy

    L = −(1/N) Σ_i Σ_v q_i(v) · log p_θ(v | source, t_<i)

where p_θ is the model's distribution over the 64 014-word vocabulary at
position i, N the number of real (non-pad) target positions, and q_i the
training distribution: with label smoothing ε = 0.1,
q_i = (1 − ε) on the correct token plus ε spread evenly over the vocabulary.
Smoothing stops the model from becoming over-confident in copying — useful
when 80 % of target tokens equal source tokens.

Optimisation
------------
AdamW (weight decay 0.01, none on biases/LayerNorm), learning rate warmed up
linearly over the first 6 % of steps then decayed linearly to 0, gradient
norm clipped at 1.0. Mixed precision: forward pass in fp16 (or bf16) for
speed and memory on the 4060, master weights and loss in fp32; with fp16 a
GradScaler multiplies the loss so small gradients don't underflow to zero.
batch_size × grad_accum sentences contribute to each optimiser step.

Robustness: a full checkpoint (weights, optimiser, scheduler,
scaler, position in the epoch) is written atomically every
``save_every_steps`` and at each epoch end, and ``--resume`` continues from
it, so a killed run loses minutes, not the run. No DataLoader workers are
used (Windows): batches are built in the main process from pre-tokenised ids.
"""

from __future__ import annotations  # modern type-hint syntax

import math  # math.ceil for counting batches
import os  # os.replace for safe file saving
import random  # seeded shuffling
import time  # timing
from pathlib import Path  # file paths
from typing import Any, Dict, List, Optional, Sequence, Tuple  # type hints

import torch  # PyTorch: tensors, GPU, training
import torch.nn.functional as F  # functional API (cross_entropy loss)

from .seq2seq_format import SpecialIds, encode, load_tokenizer, special_ids  # IndicBART format helpers

Example = Tuple[List[int], List[int]]          # (source ids, target ids)


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
def make_example(tok: Any, ids: SpecialIds, source: str, target: str,
                 max_src: int, max_tgt: int) -> Example:
    """Source "<s-words> </s> <2hi>", target "<2hi> <t-words> </s>".

    Words are encoded on their own and the specials appended as ids, so
    truncation can never cut off </s> or the language tag.
    """
    s = encode(tok, source)[: max_src - 2] + [ids.eos, ids.lang]  # source ids (max 62) + </s> + <2hi>
    t = [ids.lang] + encode(tok, target)[: max_tgt - 2] + [ids.eos]  # <2hi> + target ids (max 62) + </s>
    return s, t


def tokenize_split(rows: Sequence[Dict[str, Any]], tok: Any, ids: SpecialIds,
                   max_src: int, max_tgt: int) -> List[Example]:  # turn every (source, target) row into id lists
    return [make_example(tok, ids, r["source"], r["target"], max_src, max_tgt) for r in rows]


def collate(batch: Sequence[Example], pad: int, device: str) -> Dict[str, torch.Tensor]:
    """Pad a batch; labels are the target shifted left, pads ignored (-100)."""
    ls = max(len(s) for s, _ in batch)  # longest source in the batch
    lt = max(len(t) for _, t in batch) - 1  # longest target minus 1 (teacher forcing drops one token)
    src = torch.full((len(batch), ls), pad, dtype=torch.long)  # encoder input, padded
    att = torch.zeros((len(batch), ls), dtype=torch.long)  # attention mask (1 = real token)
    dec = torch.full((len(batch), lt), pad, dtype=torch.long)  # decoder input, padded
    lab = torch.full((len(batch), lt), -100, dtype=torch.long)  # labels; -100 = "ignore this position in the loss"
    for i, (s, t) in enumerate(batch):  # fill row i
        src[i, : len(s)] = torch.tensor(s)  # source ids
        att[i, : len(s)] = 1  # mark real tokens
        dec[i, : len(t) - 1] = torch.tensor(t[:-1])  # decoder reads t[:-1]
        lab[i, : len(t) - 1] = torch.tensor(t[1:])  # and must predict t[1:] (the next token)
    return {k: v.to(device, non_blocking=True) for k, v in  # move all tensors to the GPU
            dict(input_ids=src, attention_mask=att, decoder_input_ids=dec, labels=lab).items()}


def batches_for_epoch(n: int, batch_size: int, seed: int, epoch: int) -> List[List[int]]:
    """Deterministic shuffle per epoch, so a resumed run sees the same order."""
    order = list(range(n))  # indices 0..n-1
    random.Random(seed * 1000 + epoch).shuffle(order)  # shuffle with a seed unique to this epoch
    return [order[i:i + batch_size] for i in range(0, n, batch_size)]  # cut into batches of batch_size


# ----------------------------------------------------------------------
# Loss and evaluation
# ----------------------------------------------------------------------
def seq2seq_loss(model: Any, b: Dict[str, torch.Tensor], smoothing: float,
                 dtype: Optional[torch.dtype]) -> torch.Tensor:  # compute the training loss for one batch
    with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):  # mixed precision: run the model in bf16/fp16
        logits = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                       decoder_input_ids=b["decoder_input_ids"]).logits  # scores for every vocab word at every position: [batch, length, vocab]
    return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), b["labels"].reshape(-1),  # flatten to [batch*length, vocab] and compare with labels (in fp32)
                           ignore_index=-100, label_smoothing=smoothing)  # skip padding; apply label smoothing (0.1)


@torch.no_grad()  # decorator: no gradients inside this function (evaluation only)
def evaluate_loss(model: Any, data: Sequence[Example], batch_size: int, pad: int,
                  device: str, dtype: Optional[torch.dtype]) -> float:
    """Mean un-smoothed cross-entropy per target token on ``data``."""
    model.eval()  # evaluation mode (dropout off)
    total, n_tok = 0.0, 0  # sum of losses, number of tokens
    for i in range(0, len(data), batch_size):  # go through data in batches
        b = collate(data[i:i + batch_size], pad, device)
        n = int((b["labels"] != -100).sum())  # number of real target tokens in this batch
        total += float(seq2seq_loss(model, b, 0.0, dtype)) * n  # batch mean loss × tokens = batch total loss
        n_tok += n
    model.train()  # back to training mode
    return total / max(n_tok, 1)  # average loss per token


# ----------------------------------------------------------------------
# Checkpointing
# ----------------------------------------------------------------------
def save_checkpoint(path: Path, state: Dict[str, Any]) -> None:
    """Write to a temp file then rename: a kill mid-save never corrupts the last one."""
    path.parent.mkdir(parents=True, exist_ok=True)  # make sure the folder exists
    tmp = path.with_suffix(".tmp")  # temporary file name
    torch.save(state, tmp)  # write everything to the temp file
    os.replace(tmp, path)  # atomic rename: the real file is replaced in one step


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
def train(
    cfg: Any, train_rows: Sequence[Dict[str, Any]], val_rows: Sequence[Dict[str, Any]],
    out_dir: Path, max_steps: Optional[int] = None, resume: bool = False,
    log: Any = print,
) -> Dict[str, Any]:
    """Train; returns a summary dict (steps, timing, VRAM, losses)."""
    from transformers import MBartForConditionalGeneration, get_linear_schedule_with_warmup  # model class + LR scheduler

    cc = cfg["corrector"]  # corrector hyperparameters from config
    device = "cuda"  # training runs on the GPU
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": None}[cc["precision"]]  # chosen precision (bf16 in our config)
    torch.manual_seed(cfg.seed)  # reproducible weight init / dropout

    tok = load_tokenizer(cc["model_name"], local_files_only=True)  # IndicBART tokenizer (from local cache)
    ids = special_ids(tok, cc["lang_tag"])  # ids of </s>, <2hi>, <pad> ...
    t0 = time.monotonic()  # start timer for tokenisation
    train_data = tokenize_split(train_rows, tok, ids, cc["max_source_len"], cc["max_target_len"])  # tokenize training pairs once
    val_data = tokenize_split(val_rows, tok, ids, cc["max_source_len"], cc["max_target_len"])  # tokenize validation pairs
    log(f"  tokenised      : train {len(train_data)}, val {len(val_data)} in {time.monotonic() - t0:.0f} s")

    model = MBartForConditionalGeneration.from_pretrained(cc["model_name"], local_files_only=True)  # load pre-trained IndicBART
    if cc["gradient_checkpointing"]:  # memory-saving option (off by default)
        model.gradient_checkpointing_enable()  # recompute activations instead of storing them
        model.config.use_cache = False  # cache is incompatible with gradient checkpointing
    model.to(device).train()  # move to GPU; training mode (dropout on)

    no_decay = ("bias", "layer_norm.weight", "layernorm_embedding.weight")  # parameters that should not get weight decay
    groups = [  # two parameter groups for the optimizer
        {"params": [p for n, p in model.named_parameters() if not n.endswith(no_decay)],
         "weight_decay": cc["weight_decay"]},  # normal weights: decay 0.01
        {"params": [p for n, p in model.named_parameters() if n.endswith(no_decay)],
         "weight_decay": 0.0},  # biases and LayerNorm: no decay
    ]
    opt = torch.optim.AdamW(groups, lr=cc["lr"])  # AdamW optimizer, learning rate 5e-5
    bs, accum = cc["batch_size"], cc["grad_accum"]  # 16 sentences per micro-batch, 2 micro-batches per step
    micro_per_epoch = math.ceil(len(train_data) / bs)  # micro-batches in one epoch
    steps_per_epoch = math.ceil(micro_per_epoch / accum)  # optimizer steps in one epoch
    total_steps = steps_per_epoch * cc["epochs"]  # total optimizer steps
    sched = get_linear_schedule_with_warmup(opt, int(cc["warmup_ratio"] * total_steps), total_steps)  # LR rises for 6 % of steps, then falls linearly to 0
    scaler = torch.amp.GradScaler("cuda", enabled=dtype is torch.float16)  # loss scaling, only needed for fp16 (disabled for bf16)

    ckpt_path = out_dir / "checkpoint_last.pt"  # where the resumable checkpoint is saved
    step, start_epoch, start_micro = 0, 0, 0  # starting position
    if resume and ckpt_path.exists():  # continue an interrupted run
        st = torch.load(ckpt_path, map_location="cpu", weights_only=False)  # load the saved state
        model.load_state_dict(st["model"])  # restore weights
        opt.load_state_dict(st["optimizer"])  # restore optimizer moments
        sched.load_state_dict(st["scheduler"])  # restore LR schedule position
        scaler.load_state_dict(st["scaler"])  # restore loss scaler
        step, start_epoch, start_micro = st["step"], st["epoch"], st["next_micro"]  # restore position in training
        log(f"  resumed        : step {step}, epoch {start_epoch + 1}, micro-batch {start_micro}")

    log(f"  plan           : {cc['epochs']} epochs × {steps_per_epoch} steps = {total_steps} steps "
        f"(batch {bs} × accum {accum} = {bs * accum} pairs/step, {cc['precision']})"
        + (f"; stopping at {max_steps} (trial)" if max_steps else ""))  # print the training plan
    torch.cuda.reset_peak_memory_stats()  # start measuring GPU memory from here
    run_loss, run_n, nan_streak = 0.0, 0, 0  # running loss for logging; count of consecutive NaN losses
    t_start, step_at_start = time.monotonic(), step  # for speed / ETA calculation
    losses: List[Tuple[int, float]] = []  # (step, average loss) history
    stop = False  # set True to end early (trial runs)

    def state(epoch: int, next_micro: int) -> Dict[str, Any]:  # everything needed to resume training later
        return {"model": model.state_dict(), "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
                "step": step, "epoch": epoch, "next_micro": next_micro}

    for epoch in range(start_epoch, cc["epochs"]):  # loop over epochs (full passes over the data)
        batches = batches_for_epoch(len(train_data), bs, cfg.seed, epoch)  # shuffled batches for this epoch
        first = start_micro if epoch == start_epoch else 0  # when resuming, skip already-done micro-batches
        opt.zero_grad(set_to_none=True)  # clear old gradients
        for mi in range(first, len(batches)):  # each micro-batch
            b = collate([train_data[k] for k in batches[mi]], ids.pad, device)  # build padded tensors on GPU
            loss = seq2seq_loss(model, b, cc["label_smoothing"], dtype)  # forward pass -> loss
            if not torch.isfinite(loss):  # loss is NaN or infinite (numerical overflow)
                nan_streak += 1
                if nan_streak >= 3:  # three bad batches in a row -> stop with advice
                    raise RuntimeError("loss is NaN/inf 3 times in a row - set corrector.precision "
                                       "to \"bf16\" in configs/config.yaml and rerun")
                opt.zero_grad(set_to_none=True)  # discard this batch's gradients
                continue
            nan_streak = 0
            scaler.scale(loss / accum).backward()  # backpropagation; divide by accum so accumulated gradients average out
            run_loss += loss.item()  # add to running loss for logging
            run_n += 1
            if (mi + 1) % accum and mi + 1 != len(batches):  # not yet accumulated enough micro-batches
                continue  # keep accumulating gradients
            scaler.unscale_(opt)  # undo loss scaling before clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), cc["max_grad_norm"])  # clip gradient norm to 1.0 (prevents explosions)
            scaler.step(opt)  # update the weights (optimizer step)
            scaler.update()  # update scaling factor (fp16 only)
            opt.zero_grad(set_to_none=True)  # clear gradients for the next step
            sched.step()  # update the learning rate
            step += 1  # one more optimizer step done
            if step % cc["log_every_steps"] == 0 or step == max_steps:  # every 25 steps, print progress
                el = time.monotonic() - t_start  # seconds elapsed
                rate = (step - step_at_start) / el  # steps per second
                eta = (total_steps - step) / rate if rate else 0  # estimated seconds remaining
                avg = run_loss / max(run_n, 1)  # average loss since last log
                losses.append((step, avg))
                log(f"  step {step:6d}/{total_steps}  ep {epoch + 1}  loss {avg:.4f}  "
                    f"lr {sched.get_last_lr()[0]:.2e}  {rate:.2f} steps/s  "
                    f"VRAM peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB  "
                    f"ETA {eta / 60:.0f} min")
                run_loss, run_n = 0.0, 0  # reset running loss
            if not max_steps and step % cc["save_every_steps"] == 0:  # every 1000 steps (full runs only)
                save_checkpoint(ckpt_path, state(epoch, mi + 1))  # save a resumable checkpoint
                log(f"  checkpoint     : step {step} -> {ckpt_path}")
            if max_steps and step >= max_steps:  # trial run reached its limit
                stop = True
                break
        if stop:
            break
        save_checkpoint(ckpt_path, state(epoch + 1, 0))  # checkpoint at the end of each epoch
        vl = evaluate_loss(model, val_data, bs * 2, ids.pad, device, dtype)  # validation loss
        log(f"  epoch {epoch + 1} done : val loss {vl:.4f}  (checkpoint saved)")

    elapsed = time.monotonic() - t_start  # total training time
    summary = {  # statistics returned to the script
        "steps_done": step - step_at_start, "seconds": elapsed,
        "steps_per_sec": (step - step_at_start) / elapsed if elapsed else 0.0,
        "total_steps_full_run": total_steps, "steps_per_epoch": steps_per_epoch,
        "vram_peak_alloc_gib": torch.cuda.max_memory_allocated() / 2**30,  # peak GPU memory used (GiB)
        "vram_peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,  # peak GPU memory reserved (GiB)
        "losses": losses,
    }
    summary["val_loss"] = evaluate_loss(model, val_data, bs * 2, ids.pad, device, dtype)  # final validation loss
    if not max_steps:  # full run (not a trial)
        final = out_dir / "final"  # folder for the finished model
        model.save_pretrained(final)  # save weights + config
        tok.save_pretrained(final)  # save tokenizer next to it (so the demo is self-contained)
        log(f"  final model    : {final}")
    return summary
