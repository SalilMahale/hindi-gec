# -*- coding: utf-8 -*-
"""Correct a Hindi sentence and explain the correction (PLAN §4.3–4.4).

Pipeline tonight (the MuRIL detector is cut, so detection is derived from the
corrector):

    raw text -> bio_clean normalisation -> IndicBART beam search
             -> hallucination guard -> edit extractor -> typed edits

Hallucination guard (PLAN §4.3)
-------------------------------
A seq2seq corrector that is unsure tends to *rewrite* rather than leave text
alone — and every unrequested rewrite is a false positive that F0.5 punishes
heavily. So the output is rejected, and the input returned unchanged, when its
token-level edit distance from the input is too large:

    reject if  d(source, output) > max(max_token_edits, max_edit_ratio · |source|)

where d is the Levenshtein distance over tokens (insertions, deletions and
substitutions of whole words), |source| the number of source tokens, and the
two limits come from config (3 tokens, 25 %). The max() reading lets a long
sentence carry a two-word agreement fix plus another error; the plan's
wording is ambiguous and the evaluation reports guard-on and guard-off.

Sentence confidence: beam search returns a length-normalised log-probability s
for the chosen output; exp(s) is the geometric mean of its per-token
probabilities, shown as a rough confidence for the whole sentence.

Per-edit confidence
---------------------------------
One sentence score cannot tell a sure fix from a guess in the same sentence,
so every edit gets its own number. After beam search, the chosen output
y = (y_1 … y_T) is fed back through the model once (teacher forcing), which
gives for every generated subword

    p_t = P(y_t | y_1 … y_(t-1), x)

the probability the model gave to the subword it wrote, having read the input x
and its own output so far. SentencePiece subwords that start with "▁" begin a
new word, so the subwords are grouped into the output's words. For an edit
whose corrected words are W (its target span),

    confidence(edit) = (1 / |S|) · Σ_{t ∈ S} p_t,   S = the subwords of the words in W

i.e. the mean probability of the subwords the model wrote for the correction.
A deletion has no corrected words; it uses the subwords of the word that now
follows the deleted ones (or of </s> at the end): how sure the model was about
continuing without them. If the subwords cannot be matched to the words
(never seen on our data, but possible), every edit falls back to the sentence
confidence.

Alternatives: beam search keeps 5 hypotheses; the best
``app.num_alternatives`` are returned. The first is the correction; the
others, after the same guard, are shown as "other possible corrections" when
they differ from it. Asking for 3 hypotheses does not change the best one
(checked on synthetic val, scripts/12_check_phase_b.py).

Offline: weights load from a local directory with ``local_files_only``, so the
demo works with Wi-Fi off.

Paragraph mode
---------------------------------
The model was trained on single sentences, so a paragraph is split into
sentences first (``split_sentences``: after every । ? !) and all of them go
through the model in one padded GPU batch (``correct_batch``). Padding does not
change the result: the attention mask hides the pad positions.
"""

from __future__ import annotations  # modern type-hint syntax

import os  # environment variables
import re  # regular expressions (sentence splitting)
import time  # timing (latency measurement)
from dataclasses import dataclass, field  # record class
from pathlib import Path  # file paths
from typing import Any, Dict, List, Optional, Sequence, Tuple  # type hints

from .domain_filter import clean_tokens  # same text cleaning used on the training data
from .edit_extractor import extract_edits  # finds and types the differences between input and output
from .explain import explain  # bilingual reason per edit
from .names import load_gazetteer  # person names -> honorific vs plural
from .seq2seq_format import decode, encode, load_tokenizer, special_ids  # IndicBART format helpers

FINAL_PUNCT = ("।", "?", "!")  # valid sentence-ending marks
_SENTENCE = re.compile(r"[^।?!]+[।?!]+|[^।?!]+$")  # a run of text up to (and including) its end mark(s), or a last piece with none


def split_sentences(text: str) -> List[str]:
    """Split a paragraph after every । ? ! ; a last piece without an end mark is kept too."""
    return [s.strip() for s in _SENTENCE.findall(text) if s.strip()]  # drop empty pieces (spaces between sentences)


def token_edit_distance(a: Sequence[str], b: Sequence[str]) -> int:
    """Levenshtein distance between two token lists (unit costs)."""
    prev = list(range(len(b) + 1))  # row for empty a: turning "" into b[:j] costs j insertions
    for i, x in enumerate(a, 1):  # each word of a (i starts at 1)
        cur = [i] + [0] * len(b)  # turning a[:i] into "" costs i deletions
        for j, y in enumerate(b, 1):  # each word of b
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))  # min of: delete, insert, substitute (free if same word)
        prev = cur  # move to the next row (dynamic programming keeps only 2 rows)
    return prev[-1]  # minimum number of word edits to turn a into b


def guard_rejects(source: Sequence[str], output: Sequence[str],
                  max_token_edits: int, max_edit_ratio: float) -> bool:
    """True if the output changes too much of the source to be trusted."""
    limit = max(max_token_edits, max_edit_ratio * len(source))  # allowed changes: at least 3, or 25 % of the sentence length
    return token_edit_distance(source, output) > limit  # too many changes -> reject (model is probably hallucinating)


def prepare_input(text: str) -> List[str]:
    """bio_clean normalisation; add a danda if the user left it off (training format)."""
    toks = clean_tokens(text)  # clean + normalise + tokenize
    if toks and toks[-1] not in FINAL_PUNCT:  # sentence doesn't end with । ? !
        toks.append("।")  # add a danda, because every training sentence ended with one
    return toks


@dataclass
class Result:  # everything the app needs to show for one correction
    input: str  # cleaned input sentence
    corrected: str  # final corrected sentence
    edits: List[Dict[str, Any]] = field(default_factory=list)  # edits: span, wrong, right, type, confidence, explanation
    raw_output: str = ""  # what the model produced before the guard
    guarded: bool = False  # True if the guard rejected the model output
    confidence: float = 0.0  # sentence confidence: exp(length-normalised beam score)
    latency_ms: float = 0.0  # time taken in milliseconds
    system: str = "C"  # which system produced it ("C" = IndicBART corrector)
    alternatives: List[Dict[str, Any]] = field(default_factory=list)  # other beams: text, score, edits


@dataclass
class Hypothesis:  # one beam-search output
    ids: List[int]  # generated ids, starting with the decoder start <2hi>
    words: List[str]  # decoded words
    score: float  # exp(length-normalised log-probability)


def group_pieces(pieces: Sequence[str]) -> List[List[int]]:
    """Indices of the subword pieces making up each word ("▁" starts a new word)."""
    groups: List[List[int]] = []
    for k, p in enumerate(pieces):
        if p.startswith("▁") or not groups:
            groups.append([k])
        else:
            groups[-1].append(k)
    return groups


def edit_confidence(tgt_span: Sequence[int], word_probs: List[List[float]], eos_prob: float) -> float:
    """Mean subword probability of the edit's corrected words (see module docstring)."""
    a, b = tgt_span
    if a == b:  # deletion: the word that now follows, or </s> at the end
        probs = word_probs[a] if a < len(word_probs) else [eos_prob]
    else:
        probs = [p for w in word_probs[a:b] for p in w]
    return sum(probs) / len(probs)


class Corrector:
    """IndicBART corrector, loaded once; ``correct`` is the public entry point."""

    def __init__(self, cfg: Any, model_dir: Optional[Path] = None, device: Optional[str] = None,
                 extended: Optional[bool] = None) -> None:
        """``extended``: type edits with PRON (default: corrector.extended_types, true for the trained model)."""
        os.environ.setdefault("HF_HUB_OFFLINE", "1")  # never try to download (works without internet)
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        import torch  # PyTorch (imported here so importing this module stays fast)
        from transformers import MBartForConditionalGeneration  # IndicBART uses the mBART architecture (encoder-decoder)

        self.torch = torch  # keep a reference for use in other methods
        self.cc = cfg["corrector"]  # corrector section of the config
        path = Path(model_dir) if model_dir else cfg.path("model")      # models/corrector
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")  # use the GPU if available
        self.tok = load_tokenizer(str(path), local_files_only=True)  # load the saved tokenizer
        self.ids = special_ids(self.tok, self.cc["lang_tag"])  # ids of <s>, </s>, <pad>, <2hi>
        dtype = torch.bfloat16 if self.device == "cuda" and self.cc["precision"] == "bf16" else torch.float32  # bf16 on GPU (faster, less memory), fp32 on CPU
        self.model = MBartForConditionalGeneration.from_pretrained(  # load the fine-tuned weights
            str(path), local_files_only=True, dtype=dtype).to(self.device).eval()  # move to device; eval() turns off dropout
        g = self.cc["hallucination_guard"]  # guard limits from config
        self.max_edits, self.max_ratio = g["max_token_edits"], g["max_edit_ratio"]  # 3 and 0.25
        self.names = load_gazetteer(cfg.path("name_gazetteer")) if "name_gazetteer" in cfg["paths"] else frozenset()
        self.n_alt = cfg.get("app", {}).get("num_alternatives", 1)  # beams returned by correct()
        self.extended = self.cc.get("extended_types", False) if extended is None else extended

    def _encode_batch(self, sources: Sequence[Sequence[str]]) -> Tuple[Any, Any]:
        """Padded encoder input ids and attention mask for a batch of token lists."""
        torch, ids = self.torch, self.ids
        enc = [encode(self.tok, " ".join(s))[: self.cc["max_source_len"] - 2] + [ids.eos, ids.lang]  # sentence ids (truncated) + </s> + <2hi>
               for s in sources]
        width = max(len(e) for e in enc)  # longest input in the batch
        inp = torch.full((len(enc), width), ids.pad, dtype=torch.long)  # batch tensor filled with <pad>
        att = torch.zeros((len(enc), width), dtype=torch.long)  # attention mask: 1 = real token, 0 = padding
        for i, e in enumerate(enc):
            inp[i, : len(e)] = torch.tensor(e)  # copy the ids into row i
            att[i, : len(e)] = 1  # mark them as real tokens
        return inp.to(self.device), att.to(self.device)

    def hypotheses(self, sources: Sequence[Sequence[str]], n: int = 1,
                   num_beams: Optional[int] = None) -> List[List[Hypothesis]]:
        """The n best beam-search outputs per input, best first, no guard."""
        torch, ids = self.torch, self.ids
        inp, att = self._encode_batch(sources)
        beams = num_beams or self.cc["num_beams"]
        with torch.no_grad():  # no gradients needed for prediction (saves memory/time)
            out = self.model.generate(  # beam-search decoding
                input_ids=inp, attention_mask=att,
                num_beams=beams, num_return_sequences=min(n, beams), max_length=self.cc["max_target_len"],  # 5 beams, max 64 tokens
                early_stopping=True, pad_token_id=ids.pad, bos_token_id=ids.bos, eos_token_id=ids.eos,  # stop when beams finish; correct special ids
                decoder_start_token_id=ids.lang, output_scores=True, return_dict_in_generate=True)  # decoder starts from <2hi>; also return scores
        k = min(n, beams)
        scores = out.sequences_scores.float().exp().tolist() if out.sequences_scores is not None \
            else [0.0] * len(out.sequences)  # exp(average log-prob) = confidence
        hyps = [Hypothesis(ids=[t for t in seq.tolist()], words=decode(self.tok, seq).split(), score=sc)
                for seq, sc in zip(out.sequences, scores)]
        return [hyps[i * k:(i + 1) * k] for i in range(len(sources))]

    def generate(self, sources: Sequence[Sequence[str]], num_beams: Optional[int] = None
                 ) -> List[Tuple[List[str], float]]:
        """Raw best outputs (tokens, confidence) for a batch of token lists, no guard (used by evaluation)."""
        return [(h[0].words, h[0].score) for h in self.hypotheses(sources, 1, num_beams)]

    def word_probs(self, sources: Sequence[Sequence[str]], hyps: Sequence[Hypothesis]
                   ) -> List[Optional[Tuple[List[List[float]], float]]]:
        """Per output word, the probabilities of its subwords (teacher forcing), plus p(</s>).

        None for a hypothesis whose subwords do not regroup into its decoded words.
        """
        torch, ids = self.torch, self.ids
        inp, att = self._encode_batch(sources)
        # <2hi> w … </s>: cut at the first </s>; in a batch, shorter outputs are padded with
        # repeated </s> (not <pad>) by beam search.
        seqs = [h.ids[: h.ids.index(ids.eos, 1) + 1] if ids.eos in h.ids[1:] else [t for t in h.ids if t != ids.pad]
                for h in hyps]
        width = max(len(s) for s in seqs)
        dec = torch.full((len(seqs), width), ids.pad, dtype=torch.long)
        for i, s in enumerate(seqs):
            dec[i, : len(s)] = torch.tensor(s)
        dec = dec.to(self.device)
        with torch.no_grad():
            logits = self.model(input_ids=inp, attention_mask=att, decoder_input_ids=dec[:, :-1]).logits  # predict t[1:]
            probs = torch.softmax(logits.float(), dim=-1).gather(-1, dec[:, 1:].unsqueeze(-1)).squeeze(-1)  # p of each written token
        out: List[Optional[Tuple[List[List[float]], float]]] = []
        for i, s in enumerate(seqs):
            p = probs[i, : len(s) - 1].tolist()  # aligned with s[1:]
            body, eos_p = s[1:-1], (p[-1] if s[-1] == ids.eos else 0.0)  # drop the closing </s>
            keep = [k for k, t in enumerate(body) if t not in (ids.bos, ids.lang)]  # specials never form words
            pieces = self.tok.convert_ids_to_tokens([body[k] for k in keep])
            groups = group_pieces(pieces)
            words = ["".join(pieces[k] for k in g).replace("▁", "") for g in groups]
            if words != hyps[i].words:  # cannot align subwords to words: caller falls back
                out.append(None)
                continue
            out.append(([[p[keep[k]] for k in g] for g in groups], eos_p))
        return out

    def apply_guard(self, source: Sequence[str], output: Sequence[str]) -> Tuple[List[str], bool]:  # hallucination guard
        if guard_rejects(source, output, self.max_edits, self.max_ratio):  # too many changes
            return list(source), True  # keep the input unchanged; flag "guarded"
        return list(output), False  # accept the model output

    def _edits(self, src: List[str], out: List[str]) -> List[Dict[str, Any]]:
        """Typed edits with explanations (no confidence yet)."""
        edits = []
        for e in extract_edits(src, out, names=self.names, extended=self.extended):  # diff between input and output
            d = {"span": e.src_span, "tgt_span": e.tgt_span, "src": e.src, "tgt": e.tgt, "type": e.type}
            d["explanation"] = explain(d, out, self.names)
            edits.append(d)
        return edits

    def _result(self, src: List[str], hyps: List[Hypothesis],
                probs: Optional[Tuple[List[List[float]], float]], latency_ms: float) -> Result:
        """Guard, edits, per-edit confidence and alternatives for one input."""
        best = hyps[0]
        out, guarded = self.apply_guard(src, best.words)
        edits = self._edits(src, out)
        for e in edits:  # per-edit confidence; sentence confidence if subwords could not be aligned
            e["confidence"] = round(edit_confidence(e["tgt_span"], *probs) if probs else best.score, 3)
        alts, seen = [], {" ".join(out)}
        for h in hyps[1:]:  # other beams, guarded the same way, only if they differ
            alt, g = self.apply_guard(src, h.words)
            text = " ".join(alt)
            if g or text in seen:
                continue
            seen.add(text)
            alts.append({"text": text, "score": round(h.score, 3), "edits": self._edits(src, alt)})
        return Result(input=" ".join(src), corrected=" ".join(out), edits=edits, raw_output=" ".join(best.words),
                      guarded=guarded, confidence=best.score, latency_ms=latency_ms, alternatives=alts)

    def _run(self, srcs: List[List[str]]) -> List[Result]:  # one batch: beams + teacher-forced probabilities
        hyps = self.hypotheses(srcs, self.n_alt)
        probs = self.word_probs(srcs, [h[0] for h in hyps])
        return [self._result(s, h, p, 0.0) for s, h, p in zip(srcs, hyps, probs)]

    def correct(self, sentence: str) -> Result:
        """Correct one raw sentence; edits carry span, wrong/right text, type, confidence, explanation."""
        t0 = time.perf_counter()  # start timer
        src = prepare_input(sentence)  # clean + tokenize
        if not src:  # nothing left after cleaning
            return Result(input=sentence, corrected=sentence)
        (r,) = self._run([src])
        r.latency_ms = (time.perf_counter() - t0) * 1000  # elapsed time in ms
        return r

    def correct_batch(self, sentences: Sequence[str], batch_size: int = 16) -> List[Result]:
        """Correct many sentences in padded GPU batches; results keep the input order.

        latency_ms of each result is the batch's total time divided by the number
        of sentences (the time per sentence the user actually waited).
        """
        t0 = time.perf_counter()
        srcs = [prepare_input(s) for s in sentences]  # clean + tokenize each sentence
        todo = sorted((i for i, s in enumerate(srcs) if s), key=lambda i: len(srcs[i]))  # non-empty ones, shortest first (less padding)
        done: Dict[int, Result] = {}
        for k in range(0, len(todo), batch_size):  # one model call per batch
            idx = todo[k:k + batch_size]
            for i, r in zip(idx, self._run([srcs[i] for i in idx])):
                done[i] = r
        per = (time.perf_counter() - t0) * 1000 / max(len(sentences), 1)  # ms per sentence
        results = [done.get(i, Result(input=s, corrected=s)) for i, s in enumerate(sentences)]
        for r in results:
            r.latency_ms = per
        return results
