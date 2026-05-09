import json
import re
from collections import defaultdict

import numpy as np


CONTRACTIONS = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hes": "he's",
    "im": "i'm",
    "isnt": "isn't",
    "itll": "it'll",
    "ive": "i've",
    "lets": "let's",
    "shouldnt": "shouldn't",
    "thats": "that's",
    "theres": "there's",
    "theyre": "they're",
    "wasnt": "wasn't",
    "werent": "weren't",
    "wont": "won't",
    "wouldnt": "wouldn't",
    "youre": "you're",
    "youve": "you've",
}

NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}

ARTICLES = {"a", "an", "the"}
PUNCTUATION = [
    ";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_",
    "-", ">", "<", "@", "`", ",", "?", "!",
]
PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
COMMA_STRIP = re.compile(r"(?<=\d)(,)+(?=\d)")


def normalize_answer(answer):
    """Normalize an answer string with the standard VQA-style text rules."""
    answer = str(answer).replace("\n", " ").replace("\t", " ").strip().lower()
    answer = COMMA_STRIP.sub("", answer)
    for punct in PUNCTUATION:
        if punct + " " in answer or " " + punct in answer or COMMA_STRIP.search(answer):
            answer = answer.replace(punct, "")
        else:
            answer = answer.replace(punct, " ")
    answer = PERIOD_STRIP.sub("", answer)

    words = []
    for word in answer.split():
        word = NUMBER_WORDS.get(word, word)
        if word in ARTICLES:
            continue
        words.append(CONTRACTIONS.get(word, word))
    return " ".join(words)


def vqa_accuracy(predicted, answers):
    """Official VQA soft accuracy averaged over annotator subsets."""
    normalized_prediction = normalize_answer(predicted)
    normalized_answers = [
        normalize_answer(answer["answer"] if isinstance(answer, dict) else answer)
        for answer in answers
    ]
    if not normalized_answers:
        return 0.0

    per_answer_scores = []
    for index in range(len(normalized_answers)):
        other_answers = normalized_answers[:index] + normalized_answers[index + 1:]
        matching = sum(1 for answer in other_answers if answer == normalized_prediction)
        per_answer_scores.append(min(1.0, matching / 3.0))
    return float(np.mean(per_answer_scores))


def load_vqav2(questions_path, annotations_path):
    with open(questions_path) as f:
        questions_data = json.load(f)
    with open(annotations_path) as f:
        annotations_data = json.load(f)

    annotations_by_question = {
        annotation["question_id"]: annotation
        for annotation in annotations_data["annotations"]
    }
    questions = []
    for question in questions_data["questions"]:
        annotation = annotations_by_question.get(question["question_id"], {})
        questions.append({
            "question_id": question["question_id"],
            "image_id": question["image_id"],
            "question": question["question"],
            "answers": annotation.get("answers", []),
            "answer_type": annotation.get("answer_type"),
            "question_type": annotation.get("question_type"),
            "multiple_choice_answer": annotation.get("multiple_choice_answer"),
        })
    return questions


def summarize_scores(scores):
    if not scores:
        return {"accuracy": None, "n": 0}
    return {"accuracy": float(np.mean(scores)), "n": len(scores)}


class VQAEvaluation:
    """Loads VQAv2 annotations and accumulates ViLT question accuracy."""

    def __init__(self, questions_path, annotations_path, variants):
        self.variants = tuple(variants)
        self.accuracies = defaultdict(list)
        self.answer_type_accuracies = defaultdict(lambda: defaultdict(list))
        self.questions_by_image = self._load_questions_by_image(questions_path, annotations_path)

    @staticmethod
    def _load_questions_by_image(questions_path, annotations_path):
        questions_by_image = defaultdict(list)
        for question in load_vqav2(questions_path, annotations_path):
            questions_by_image[question["image_id"]].append(question)
        return questions_by_image

    def add_image_results(self, variant, image_id, image_array, runner):
        for question in self.questions_by_image.get(image_id, []):
            predicted = runner.predict(image_array, question["question"])
            accuracy = vqa_accuracy(predicted, question["answers"])
            self.accuracies[variant].append(accuracy)
            answer_type = question.get("answer_type") or "unknown"
            self.answer_type_accuracies[variant][answer_type].append(accuracy)

    def summary_rows(self):
        rows = []
        for variant in self.variants:
            summary = summarize_scores(self.accuracies[variant])
            rows.append({
                "variant": variant,
                "accuracy": summary["accuracy"],
                "questions": summary["n"],
            })
        return rows

    def answer_type_summary(self, variant):
        return {
            answer_type: summarize_scores(scores)
            for answer_type, scores in sorted(self.answer_type_accuracies[variant].items())
        }

    def print_report(self):
        print("\n  VQA Accuracy - ViLT on VQAv2")
        print(f"  {'Fixation':<22} {'Accuracy':>10}  {'N questions':>12}")
        print(f"  {'-'*46}")
        for row in self.summary_rows():
            if row["accuracy"] is None:
                print(f"  {row['variant']:<22} {'-':>10}  {'0':>12}")
            else:
                print(f"  {row['variant']:<22} {row['accuracy']*100:>9.2f}%  {row['questions']:>12}")

    def to_json(self):
        return {
            variant: {
                **summarize_scores(self.accuracies[variant]),
                "answer_types": self.answer_type_summary(variant),
            }
            for variant in self.variants
            if self.accuracies[variant]
        }
