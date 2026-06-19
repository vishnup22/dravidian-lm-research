from __future__ import annotations



import argparse

import csv

import json

import math

import statistics

from collections import Counter, defaultdict

from pathlib import Path

from typing import Any



import torch

from transformers import AutoModelForCausalLM, AutoTokenizer



try:

    from datasets import load_dataset

except ImportError:  # pragma: no cover - optional at runtime

    load_dataset = None





TELUGU_BLOCK_START = 0x0C00

TELUGU_BLOCK_END = 0x0C7F





def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(

        description="Telugu evaluation runner for causal language models."

    )

    parser.add_argument("--model", required=True, help="HF model id or local model path.")

    parser.add_argument(

        "--tokenizer",

        default=None,

        help="Optional tokenizer id/path. Defaults to --model.",

    )

    parser.add_argument(

        "--device",

        default="cuda" if torch.cuda.is_available() else "cpu",

        help="Device to use, for example cuda, cuda:0, or cpu.",

    )

    parser.add_argument(

        "--dtype",

        default="auto",

        choices=["auto", "float32", "float16", "bfloat16"],

        help="Model dtype.",

    )

    parser.add_argument(

        "--output",

        default=None,

        help="Optional output JSON path for metrics and samples.",

    )



    subparsers = parser.add_subparsers(dest="command", required=True)



    p_belebele = subparsers.add_parser(

        "belebele", help="Evaluate Telugu Belebele-style multiple-choice accuracy."

    )

    p_belebele.add_argument("--dataset", default="facebook/belebele")

    p_belebele.add_argument("--dataset_config", default="tel_Telu")

    p_belebele.add_argument("--split", default="test")

    p_belebele.add_argument("--limit", type=int, default=None)

    p_belebele.add_argument("--fewshot_file", default=None)

    p_belebele.add_argument("--max_examples", type=int, default=None)

    p_belebele.add_argument("--output", default=None)



    p_mcq = subparsers.add_parser(

        "mcq", help="Evaluate a local or HF multiple-choice dataset."

    )

    add_dataset_args(p_mcq)

    p_mcq.add_argument("--split", default="test")

    p_mcq.add_argument("--prompt_field", default="prompt")

    p_mcq.add_argument("--choices_field", default="choices")

    p_mcq.add_argument("--answer_field", default="answer")

    p_mcq.add_argument("--answer_is_1_indexed", action="store_true")

    p_mcq.add_argument("--limit", type=int, default=None)

    p_mcq.add_argument("--output", default=None)



    p_cls = subparsers.add_parser(

        "classification", help="Evaluate zero-shot prompt-based classification."

    )

    add_dataset_args(p_cls)

    p_cls.add_argument("--split", default="test")

    p_cls.add_argument("--text_field", default="text")

    p_cls.add_argument("--label_field", default="label")

    p_cls.add_argument(

        "--label_names",

        required=True,

        help="JSON list or pipe-delimited label names, e.g. positive|neutral|negative",

    )

    p_cls.add_argument(

        "--prompt_template",

        default="క్రింది వాక్యానికి సరైన వర్గాన్ని ఎంచుకోండి.\n\nవాక్యం: {text}\nవర్గం:",

        help="Prompt template with {text}.",

    )

    p_cls.add_argument("--limit", type=int, default=None)

    p_cls.add_argument("--output", default=None)



    p_ref = subparsers.add_parser(

        "reference-gen",

        help="Generate outputs and score them against references.",

    )

    add_dataset_args(p_ref)

    p_ref.add_argument("--split", default="test")

    p_ref.add_argument("--prompt_field", default="prompt")

    p_ref.add_argument("--reference_field", default="reference")

    p_ref.add_argument("--limit", type=int, default=None)

    add_generation_args(p_ref)

    p_ref.add_argument("--output", default=None)



    p_gen = subparsers.add_parser(

        "generation", help="Generate continuations and compute quality diagnostics."

    )

    p_gen.add_argument(

        "--prompts_file",

        required=True,

        help="TXT or JSONL file with prompts. JSONL must contain a prompt field.",

    )

    p_gen.add_argument("--prompt_field", default="prompt")

    p_gen.add_argument("--limit", type=int, default=None)

    add_generation_args(p_gen)

    p_gen.add_argument("--output", default=None)



    p_human = subparsers.add_parser(

        "human-pack",

        help="Export generations for manual review in CSV/JSONL form.",

    )

    p_human.add_argument("--prompts_file", required=True)

    p_human.add_argument("--prompt_field", default="prompt")

    p_human.add_argument("--limit", type=int, default=None)

    p_human.add_argument("--csv_out", default="te_human_eval.csv")

    p_human.add_argument("--jsonl_out", default="te_human_eval.jsonl")

    add_generation_args(p_human)

    p_human.add_argument("--output", default=None)



    p_all = subparsers.add_parser(

        "all",

        help="Run the full Telugu evaluation suite in one command.",

    )

    p_all.add_argument("--belebele_dataset", default="facebook/belebele")

    p_all.add_argument("--belebele_config", default="tel_Telu")

    p_all.add_argument("--belebele_split", default="test")

    p_all.add_argument("--belebele_limit", type=int, default=None)

    p_all.add_argument("--belebele_fewshot_file", default=None)

    p_all.add_argument("--classification_dataset_source", choices=["local", "hf"], default=None)

    p_all.add_argument("--classification_dataset_path", default=None)

    p_all.add_argument("--classification_dataset_config", default=None)

    p_all.add_argument("--classification_split", default="test")

    p_all.add_argument("--classification_text_field", default="text")

    p_all.add_argument("--classification_label_field", default="label")

    p_all.add_argument("--classification_label_names", default=None)

    p_all.add_argument(

        "--classification_prompt_template",

        default="క్రింది వాక్యానికి సరైన వర్గాన్ని ఎంచుకోండి.\n\nవాక్యం: {text}\nవర్గం:",

    )

    p_all.add_argument("--classification_limit", type=int, default=None)

    p_all.add_argument("--reference_dataset_source", choices=["local", "hf"], default=None)

    p_all.add_argument("--reference_dataset_path", default=None)

    p_all.add_argument("--reference_dataset_config", default=None)

    p_all.add_argument("--reference_split", default="test")

    p_all.add_argument("--reference_prompt_field", default="prompt")

    p_all.add_argument("--reference_reference_field", default="reference")

    p_all.add_argument("--reference_limit", type=int, default=None)

    p_all.add_argument("--prompts_file", default=None)

    p_all.add_argument("--prompt_field", default="prompt")

    p_all.add_argument("--generation_limit", type=int, default=None)

    p_all.add_argument("--human_limit", type=int, default=None)

    p_all.add_argument("--csv_out", default="te_human_eval.csv")

    p_all.add_argument("--jsonl_out", default="te_human_eval.jsonl")

    add_generation_args(p_all)

    p_all.add_argument("--output", default=None)



    return parser.parse_args()





def add_dataset_args(parser: argparse.ArgumentParser) -> None:

    parser.add_argument(

        "--dataset_source",

        choices=["local", "hf"],

        required=True,

        help="Whether to read the dataset from a local file or Hugging Face datasets.",

    )

    parser.add_argument(

        "--dataset_path",

        required=True,

        help="Local JSONL/JSON file path or HF dataset name.",

    )

    parser.add_argument(

        "--dataset_config",

        default=None,

        help="Optional HF dataset config name.",

    )





def add_generation_args(parser: argparse.ArgumentParser) -> None:

    parser.add_argument("--max_new_tokens", type=int, default=64)

    parser.add_argument("--temperature", type=float, default=0.8)

    parser.add_argument("--top_p", type=float, default=0.95)

    parser.add_argument("--do_sample", action="store_true")

    parser.add_argument("--num_beams", type=int, default=1)





def get_torch_dtype(name: str) -> torch.dtype | None:

    if name == "auto":

        return None

    if name == "float32":

        return torch.float32

    if name == "float16":

        return torch.float16

    if name == "bfloat16":

        return torch.bfloat16

    raise ValueError(f"Unsupported dtype: {name}")





class TeluguEvaluator:

    def __init__(self, model_name: str, tokenizer_name: str | None, device: str, dtype: str):

        self.model_name = model_name

        self.tokenizer_name = tokenizer_name or model_name

        self.device = device



        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

        torch_dtype = get_torch_dtype(dtype)

        model_kwargs: dict[str, Any] = {}

        if torch_dtype is not None:

            model_kwargs["torch_dtype"] = torch_dtype

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)

        self.model.to(self.device)

        self.model.eval()



        if self.tokenizer.pad_token is None:

            if self.tokenizer.eos_token is not None:

                self.tokenizer.pad_token = self.tokenizer.eos_token

            else:

                self.tokenizer.add_special_tokens({"pad_token": "<pad>"})

                self.model.resize_token_embeddings(len(self.tokenizer))



    def score_continuation(self, prompt: str, continuation: str) -> float:

        prompt_ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)

        full_ids = self.tokenizer(

            prompt + continuation,

            return_tensors="pt",

            add_special_tokens=False,

        )

        input_ids = full_ids.input_ids.to(self.device)

        attention_mask = full_ids.attention_mask.to(self.device)

        prompt_len = prompt_ids.input_ids.size(1)

        target_ids = input_ids.clone()

        target_ids[:, :prompt_len] = -100



        with torch.no_grad():

            outputs = self.model(

                input_ids=input_ids,

                attention_mask=attention_mask,

                labels=target_ids,

            )



        valid_tokens = (target_ids != -100).sum().item()

        return -outputs.loss.item() * valid_tokens



    def predict_choice(self, prompt: str, choices: list[str]) -> tuple[int, list[float]]:

        scores = [self.score_continuation(prompt, " " + choice) for choice in choices]

        pred = max(range(len(scores)), key=lambda idx: scores[idx])

        return pred, scores



    def generate(self, prompt: str, max_new_tokens: int, temperature: float, top_p: float,

                 do_sample: bool, num_beams: int) -> str:

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():

            output = self.model.generate(

                **inputs,

                max_new_tokens=max_new_tokens,

                temperature=temperature,

                top_p=top_p,

                do_sample=do_sample,

                num_beams=num_beams,

                pad_token_id=self.tokenizer.pad_token_id,

                eos_token_id=self.tokenizer.eos_token_id,

            )

        generated = output[0][inputs["input_ids"].size(1):]

        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()





def load_records(

    dataset_source: str,

    dataset_path: str,

    dataset_config: str | None,

    split: str,

) -> list[dict[str, Any]]:

    if dataset_source == "hf":

        if load_dataset is None:

            raise ImportError("datasets is required for --dataset_source hf")

        ds = load_dataset(dataset_path, dataset_config, split=split)

        return [dict(row) for row in ds]



    path = Path(dataset_path)

    if not path.exists():

        raise FileNotFoundError(f"Dataset file not found: {path}")

    if path.suffix.lower() == ".jsonl":

        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    if path.suffix.lower() == ".json":

        data = json.loads(path.read_text(encoding="utf-8"))

        if isinstance(data, list):

            return data

        raise ValueError("JSON dataset must be a list of records.")

    raise ValueError("Local datasets must be .jsonl or .json")





def load_prompts_file(path_str: str, prompt_field: str) -> list[str]:

    path = Path(path_str)

    if not path.exists():

        raise FileNotFoundError(f"Prompts file not found: {path}")

    if path.suffix.lower() == ".txt":

        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    if path.suffix.lower() == ".jsonl":

        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

        return [str(row[prompt_field]).strip() for row in rows if str(row.get(prompt_field, "")).strip()]

    raise ValueError("Prompts file must be .txt or .jsonl")





def maybe_limit(records: list[Any], limit: int | None) -> list[Any]:

    return records if limit is None else records[:limit]





def format_belebele_prompt(row: dict[str, Any], fewshot_examples: list[dict[str, Any]]) -> tuple[str, list[str], int]:

    context = (

        row.get("flores_passage")

        or row.get("passage")

        or row.get("context")

        or row.get("question_passage")

    )

    question = row.get("question")

    choices = [

        row.get("mc_answer1"),

        row.get("mc_answer2"),

        row.get("mc_answer3"),

        row.get("mc_answer4"),

    ]

    if any(choice is None for choice in choices):

        choices = row.get("choices")

    if context is None or question is None or choices is None:

        raise KeyError("Belebele row is missing expected fields.")



    shots = []

    for ex in fewshot_examples:

        ex_context = ex.get("flores_passage") or ex.get("passage") or ex.get("context")

        ex_question = ex.get("question")

        ex_choices = [

            ex.get("mc_answer1"),

            ex.get("mc_answer2"),

            ex.get("mc_answer3"),

            ex.get("mc_answer4"),

        ]

        if any(choice is None for choice in ex_choices):

            ex_choices = ex.get("choices")

        ex_answer = normalize_answer_index(ex.get("correct_answer_num") or ex.get("answer"), one_indexed=True)

        answer_text = ex_choices[ex_answer]

        shots.append(

            "ప్రాసంగిక భాగం:\n"

            f"{ex_context}\n\n"

            f"ప్రశ్న: {ex_question}\n"

            f"A. {ex_choices[0]}\n"

            f"B. {ex_choices[1]}\n"

            f"C. {ex_choices[2]}\n"

            f"D. {ex_choices[3]}\n"

            f"సరైన సమాధానం: {answer_text}\n"

        )



    prompt = ""

    if shots:

        prompt += "\n".join(shots) + "\n\n"

    prompt += (

        "ప్రాసంగిక భాగం:\n"

        f"{context}\n\n"

        f"ప్రశ్న: {question}\n"

        f"A. {choices[0]}\n"

        f"B. {choices[1]}\n"

        f"C. {choices[2]}\n"

        f"D. {choices[3]}\n"

        "సరైన సమాధానం:"

    )

    answer = normalize_answer_index(row.get("correct_answer_num"), one_indexed=True)

    return prompt, [choices[0], choices[1], choices[2], choices[3]], answer





def normalize_answer_index(value: Any, one_indexed: bool) -> int:

    if isinstance(value, str):

        cleaned = value.strip()

        if cleaned.upper() in {"A", "B", "C", "D"}:

            return "ABCD".index(cleaned.upper())

        value = int(cleaned)

    idx = int(value)

    return idx - 1 if one_indexed else idx





def evaluate_belebele(evaluator: TeluguEvaluator, args: argparse.Namespace) -> dict[str, Any]:

    if load_dataset is None:

        raise ImportError("datasets is required for Belebele evaluation.")

    ds = load_dataset(args.dataset, args.dataset_config, split=args.split)

    rows = [dict(row) for row in ds]

    rows = maybe_limit(rows, args.limit or args.max_examples)

    fewshot_examples = []

    if args.fewshot_file:

        fewshot_records = load_records("local", args.fewshot_file, None, "unused")

        fewshot_examples = fewshot_records



    samples = []

    correct = 0

    for row in rows:

        prompt, choices, gold = format_belebele_prompt(row, fewshot_examples)

        pred, scores = evaluator.predict_choice(prompt, choices)

        correct += int(pred == gold)

        samples.append(

            {

                "question": row.get("question"),

                "gold": gold,

                "pred": pred,

                "choices": choices,

                "scores": scores,

            }

        )



    metrics = {

        "task": "belebele",

        "dataset": args.dataset,

        "dataset_config": args.dataset_config,

        "split": args.split,

        "n": len(rows),

        "accuracy": safe_div(correct, len(rows)),

        "samples": samples,

    }

    return metrics





def evaluate_mcq(evaluator: TeluguEvaluator, args: argparse.Namespace) -> dict[str, Any]:

    rows = load_records(args.dataset_source, args.dataset_path, args.dataset_config, args.split)

    rows = maybe_limit(rows, args.limit)



    correct = 0

    samples = []

    for row in rows:

        prompt = str(row[args.prompt_field])

        choices = list(row[args.choices_field])

        gold = normalize_answer_index(row[args.answer_field], args.answer_is_1_indexed)

        pred, scores = evaluator.predict_choice(prompt, choices)

        correct += int(pred == gold)

        samples.append(

            {"prompt": prompt, "gold": gold, "pred": pred, "choices": choices, "scores": scores}

        )



    return {

        "task": "mcq",

        "dataset_path": args.dataset_path,

        "split": args.split,

        "n": len(rows),

        "accuracy": safe_div(correct, len(rows)),

        "samples": samples,

    }





def parse_label_names(raw: str) -> list[str]:

    text = raw.strip()

    if text.startswith("["):

        return list(json.loads(text))

    return [part.strip() for part in text.split("|") if part.strip()]





def evaluate_classification(evaluator: TeluguEvaluator, args: argparse.Namespace) -> dict[str, Any]:

    rows = load_records(args.dataset_source, args.dataset_path, args.dataset_config, args.split)

    rows = maybe_limit(rows, args.limit)

    label_names = parse_label_names(args.label_names)



    golds: list[int] = []

    preds: list[int] = []

    samples = []

    for row in rows:

        text = str(row[args.text_field])

        prompt = args.prompt_template.format(text=text)

        pred, scores = evaluator.predict_choice(prompt, label_names)

        gold = normalize_label(row[args.label_field], label_names)

        golds.append(gold)

        preds.append(pred)

        samples.append(

            {"text": text, "gold": gold, "pred": pred, "labels": label_names, "scores": scores}

        )



    cm = confusion_matrix(golds, preds, len(label_names))

    return {

        "task": "classification",

        "dataset_path": args.dataset_path,

        "split": args.split,

        "n": len(rows),

        "accuracy": accuracy(golds, preds),

        "macro_f1": macro_f1(golds, preds, len(label_names)),

        "label_names": label_names,

        "confusion_matrix": cm,

        "samples": samples,

    }





def evaluate_reference_gen(evaluator: TeluguEvaluator, args: argparse.Namespace) -> dict[str, Any]:

    rows = load_records(args.dataset_source, args.dataset_path, args.dataset_config, args.split)

    rows = maybe_limit(rows, args.limit)



    references: list[str] = []

    predictions: list[str] = []

    samples = []

    for row in rows:

        prompt = str(row[args.prompt_field])

        reference = str(row[args.reference_field]).strip()

        prediction = evaluator.generate(

            prompt,

            max_new_tokens=args.max_new_tokens,

            temperature=args.temperature,

            top_p=args.top_p,

            do_sample=args.do_sample,

            num_beams=args.num_beams,

        )

        references.append(reference)

        predictions.append(prediction)

        samples.append({"prompt": prompt, "reference": reference, "prediction": prediction})



    metrics = generation_diagnostics(predictions)

    metrics.update(reference_metrics(references, predictions))

    metrics.update(

        {

            "task": "reference-gen",

            "dataset_path": args.dataset_path,

            "split": args.split,

            "n": len(rows),

            "samples": samples,

        }

    )

    return metrics





def evaluate_generation(evaluator: TeluguEvaluator, args: argparse.Namespace) -> dict[str, Any]:

    prompts = maybe_limit(load_prompts_file(args.prompts_file, args.prompt_field), args.limit)

    samples = []

    generations = []

    for prompt in prompts:

        prediction = evaluator.generate(

            prompt,

            max_new_tokens=args.max_new_tokens,

            temperature=args.temperature,

            top_p=args.top_p,

            do_sample=args.do_sample,

            num_beams=args.num_beams,

        )

        samples.append({"prompt": prompt, "prediction": prediction})

        generations.append(prediction)



    metrics = generation_diagnostics(generations)

    metrics.update({"task": "generation", "n": len(prompts), "samples": samples})

    return metrics





def export_human_pack(evaluator: TeluguEvaluator, args: argparse.Namespace) -> dict[str, Any]:

    prompts = maybe_limit(load_prompts_file(args.prompts_file, args.prompt_field), args.limit)

    rows = []

    for prompt in prompts:

        prediction = evaluator.generate(

            prompt,

            max_new_tokens=args.max_new_tokens,

            temperature=args.temperature,

            top_p=args.top_p,

            do_sample=args.do_sample,

            num_beams=args.num_beams,

        )

        rows.append(

            {

                "prompt": prompt,

                "prediction": prediction,

                "fluency_1_5": "",

                "coherence_1_5": "",

                "relevance_1_5": "",

                "script_correctness_1_5": "",

                "repetition_1_5": "",

                "notes": "",

            }

        )



    csv_path = Path(args.csv_out)

    jsonl_path = Path(args.jsonl_out)



    with csv_path.open("w", encoding="utf-8", newline="") as f:

        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["prompt", "prediction"])

        writer.writeheader()

        writer.writerows(rows)



    with jsonl_path.open("w", encoding="utf-8") as f:

        for row in rows:

            f.write(json.dumps(row, ensure_ascii=False) + "\n")



    metrics = generation_diagnostics([row["prediction"] for row in rows])

    metrics.update(

        {

            "task": "human-pack",

            "n": len(rows),

            "csv_out": str(csv_path),

            "jsonl_out": str(jsonl_path),

            "samples": rows,

        }

    )

    return metrics





def evaluate_all(evaluator: TeluguEvaluator, args: argparse.Namespace) -> dict[str, Any]:

    results: dict[str, Any] = {"task": "all"}



    belebele_args = argparse.Namespace(

        dataset=args.belebele_dataset,

        dataset_config=args.belebele_config,

        split=args.belebele_split,

        limit=args.belebele_limit,

        max_examples=args.belebele_limit,

        fewshot_file=args.belebele_fewshot_file,

    )

    results["belebele"] = evaluate_belebele(evaluator, belebele_args)



    if args.classification_dataset_source and args.classification_dataset_path and args.classification_label_names:

        cls_args = argparse.Namespace(

            dataset_source=args.classification_dataset_source,

            dataset_path=args.classification_dataset_path,

            dataset_config=args.classification_dataset_config,

            split=args.classification_split,

            text_field=args.classification_text_field,

            label_field=args.classification_label_field,

            label_names=args.classification_label_names,

            prompt_template=args.classification_prompt_template,

            limit=args.classification_limit,

        )

        results["classification"] = evaluate_classification(evaluator, cls_args)

    else:

        results["classification"] = {"skipped": True, "reason": "classification dataset arguments not provided"}



    if args.reference_dataset_source and args.reference_dataset_path:

        ref_args = argparse.Namespace(

            dataset_source=args.reference_dataset_source,

            dataset_path=args.reference_dataset_path,

            dataset_config=args.reference_dataset_config,

            split=args.reference_split,

            prompt_field=args.reference_prompt_field,

            reference_field=args.reference_reference_field,

            limit=args.reference_limit,

            max_new_tokens=args.max_new_tokens,

            temperature=args.temperature,

            top_p=args.top_p,

            do_sample=args.do_sample,

            num_beams=args.num_beams,

        )

        results["reference_gen"] = evaluate_reference_gen(evaluator, ref_args)

    else:

        results["reference_gen"] = {"skipped": True, "reason": "reference generation dataset arguments not provided"}



    if args.prompts_file:

        gen_args = argparse.Namespace(

            prompts_file=args.prompts_file,

            prompt_field=args.prompt_field,

            limit=args.generation_limit,

            max_new_tokens=args.max_new_tokens,

            temperature=args.temperature,

            top_p=args.top_p,

            do_sample=args.do_sample,

            num_beams=args.num_beams,

        )

        human_args = argparse.Namespace(

            prompts_file=args.prompts_file,

            prompt_field=args.prompt_field,

            limit=args.human_limit,

            csv_out=args.csv_out,

            jsonl_out=args.jsonl_out,

            max_new_tokens=args.max_new_tokens,

            temperature=args.temperature,

            top_p=args.top_p,

            do_sample=args.do_sample,

            num_beams=args.num_beams,

        )

        results["generation"] = evaluate_generation(evaluator, gen_args)

        results["human_pack"] = export_human_pack(evaluator, human_args)

    else:

        skip = {"skipped": True, "reason": "prompts_file not provided"}

        results["generation"] = skip

        results["human_pack"] = skip



    return results





def accuracy(golds: list[int], preds: list[int]) -> float:

    if not golds:

        return 0.0

    return sum(int(g == p) for g, p in zip(golds, preds)) / len(golds)





def macro_f1(golds: list[int], preds: list[int], num_labels: int) -> float:

    scores = []

    for label in range(num_labels):

        tp = sum(1 for g, p in zip(golds, preds) if g == label and p == label)

        fp = sum(1 for g, p in zip(golds, preds) if g != label and p == label)

        fn = sum(1 for g, p in zip(golds, preds) if g == label and p != label)

        precision = safe_div(tp, tp + fp)

        recall = safe_div(tp, tp + fn)

        if precision + recall == 0:

            scores.append(0.0)

        else:

            scores.append(2 * precision * recall / (precision + recall))

    return sum(scores) / max(1, len(scores))





def confusion_matrix(golds: list[int], preds: list[int], num_labels: int) -> list[list[int]]:

    matrix = [[0 for _ in range(num_labels)] for _ in range(num_labels)]

    for gold, pred in zip(golds, preds):

        matrix[gold][pred] += 1

    return matrix





def normalize_label(value: Any, label_names: list[str]) -> int:

    if isinstance(value, int):

        return value

    if isinstance(value, str):

        cleaned = value.strip()

        if cleaned.isdigit():

            return int(cleaned)

        for idx, label in enumerate(label_names):

            if cleaned == label:

                return idx

    raise ValueError(f"Could not normalize label: {value}")





def generation_diagnostics(predictions: list[str]) -> dict[str, Any]:

    tokenized = [whitespace_tokens(text) for text in predictions]

    token_lengths = [len(tokens) for tokens in tokenized]

    char_lengths = [len(text) for text in predictions]



    distinct1 = distinct_n(tokenized, 1)

    distinct2 = distinct_n(tokenized, 2)

    repeated_bigram_rate = repeated_ngram_rate(tokenized, 2)

    repeated_trigram_rate = repeated_ngram_rate(tokenized, 3)

    telugu_char_ratio = mean([telugu_ratio(text) for text in predictions])



    return {

        "avg_tokens": mean(token_lengths),

        "median_tokens": median(token_lengths),

        "avg_chars": mean(char_lengths),

        "distinct_1": distinct1,

        "distinct_2": distinct2,

        "repeated_bigram_rate": repeated_bigram_rate,

        "repeated_trigram_rate": repeated_trigram_rate,

        "telugu_char_ratio": telugu_char_ratio,

        "empty_rate": safe_div(sum(1 for text in predictions if not text.strip()), len(predictions)),

    }





def reference_metrics(references: list[str], predictions: list[str]) -> dict[str, Any]:

    rouge_scores = [rouge_l_f1(ref, pred) for ref, pred in zip(references, predictions)]

    chrf_scores = [chrf_f1(ref, pred) for ref, pred in zip(references, predictions)]

    exact_match = [

        int(normalize_space(ref) == normalize_space(pred)) for ref, pred in zip(references, predictions)

    ]

    return {

        "rouge_l_f1": mean(rouge_scores),

        "chrf_2": mean(chrf_scores),

        "exact_match": mean(exact_match),

    }





def whitespace_tokens(text: str) -> list[str]:

    return [tok for tok in text.strip().split() if tok]





def distinct_n(tokenized_texts: list[list[str]], n: int) -> float:

    all_ngrams: list[tuple[str, ...]] = []

    total = 0

    for tokens in tokenized_texts:

        ngrams = list(make_ngrams(tokens, n))

        all_ngrams.extend(ngrams)

        total += len(ngrams)

    if total == 0:

        return 0.0

    return len(set(all_ngrams)) / total





def repeated_ngram_rate(tokenized_texts: list[list[str]], n: int) -> float:

    counts = []

    for tokens in tokenized_texts:

        grams = list(make_ngrams(tokens, n))

        if not grams:

            counts.append(0.0)

            continue

        repeated = sum(1 for gram, count in Counter(grams).items() if count > 1)

        counts.append(repeated / len(grams))

    return mean(counts)





def make_ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:

    if len(tokens) < n:

        return []

    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]





def telugu_ratio(text: str) -> float:

    letters = [ch for ch in text if not ch.isspace()]

    if not letters:

        return 0.0

    telugu = sum(1 for ch in letters if TELUGU_BLOCK_START <= ord(ch) <= TELUGU_BLOCK_END)

    return telugu / len(letters)





def rouge_l_f1(reference: str, prediction: str) -> float:

    ref_tokens = whitespace_tokens(reference)

    pred_tokens = whitespace_tokens(prediction)

    if not ref_tokens or not pred_tokens:

        return 0.0

    lcs = lcs_length(ref_tokens, pred_tokens)

    precision = lcs / len(pred_tokens)

    recall = lcs / len(ref_tokens)

    if precision + recall == 0:

        return 0.0

    return 2 * precision * recall / (precision + recall)





def lcs_length(a: list[str], b: list[str]) -> int:

    dp = [0] * (len(b) + 1)

    for token_a in a:

        prev = 0

        for j, token_b in enumerate(b, start=1):

            temp = dp[j]

            if token_a == token_b:

                dp[j] = prev + 1

            else:

                dp[j] = max(dp[j], dp[j - 1])

            prev = temp

    return dp[-1]





def chrf_f1(reference: str, prediction: str, max_order: int = 6, beta: float = 2.0) -> float:

    ref = normalize_space(reference)

    pred = normalize_space(prediction)

    if not ref or not pred:

        return 0.0

    precisions = []

    recalls = []

    for order in range(1, max_order + 1):

        ref_counts = Counter(char_ngrams(ref, order))

        pred_counts = Counter(char_ngrams(pred, order))

        overlap = sum(min(count, pred_counts[gram]) for gram, count in ref_counts.items())

        precisions.append(safe_div(overlap, sum(pred_counts.values())))

        recalls.append(safe_div(overlap, sum(ref_counts.values())))

    precision = mean(precisions)

    recall = mean(recalls)

    if precision == 0 and recall == 0:

        return 0.0

    beta_sq = beta * beta

    return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)





def char_ngrams(text: str, n: int) -> list[str]:

    if len(text) < n:

        return []

    return [text[i:i + n] for i in range(len(text) - n + 1)]





def normalize_space(text: str) -> str:

    return " ".join(text.split())





def mean(values: list[float | int]) -> float:

    return float(sum(values) / len(values)) if values else 0.0





def median(values: list[float | int]) -> float:

    return float(statistics.median(values)) if values else 0.0





def safe_div(numerator: float, denominator: float) -> float:

    return numerator / denominator if denominator else 0.0





def save_output(output_path: str | None, result: dict[str, Any]) -> None:

    if output_path is None:

        return

    path = Path(output_path)

    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")





def print_summary(result: dict[str, Any]) -> None:

    summary = {k: v for k, v in result.items() if k != "samples"}

    print(json.dumps(summary, ensure_ascii=False, indent=2))





def main() -> None:

    args = parse_args()

    evaluator = TeluguEvaluator(args.model, args.tokenizer, args.device, args.dtype)



    if args.command == "belebele":

        result = evaluate_belebele(evaluator, args)

    elif args.command == "mcq":

        result = evaluate_mcq(evaluator, args)

    elif args.command == "classification":

        result = evaluate_classification(evaluator, args)

    elif args.command == "reference-gen":

        result = evaluate_reference_gen(evaluator, args)

    elif args.command == "generation":

        result = evaluate_generation(evaluator, args)

    elif args.command == "human-pack":

        result = export_human_pack(evaluator, args)

    elif args.command == "all":

        result = evaluate_all(evaluator, args)

    else:  # pragma: no cover

        raise ValueError(f"Unknown command: {args.command}")



    save_output(args.output, result)

    print_summary(result)





if __name__ == "__main__":

    main()

