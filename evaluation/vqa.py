import json
from collections import defaultdict

import numpy as np


def vqa_accuracy(predicted, answers):
    return min(sum(1 for a in answers if a["answer"].lower() == predicted.lower()) / 3.0, 1.0)


class VQAEvaluation:
    """Loads VQAv2 annotations and accumulates ViLT question accuracy."""

    def __init__(self, questions_path, annotations_path, variants):
        self.variants = tuple(variants)
        self.accuracies = defaultdict(list)
        self.questions_by_image = self._load_questions(questions_path, annotations_path)

    @staticmethod
    def _load_questions(questions_path, annotations_path):
        with open(questions_path) as f:
            vqa_questions = json.load(f)
        with open(annotations_path) as f:
            vqa_annotations = json.load(f)

        answer_lookup = {
            annotation["question_id"]: annotation["answers"]
            for annotation in vqa_annotations["annotations"]
        }
        questions_by_image = defaultdict(list)
        for question in vqa_questions["questions"]:
            questions_by_image[question["image_id"]].append({
                "question_id": question["question_id"],
                "question": question["question"],
                "answers": answer_lookup.get(question["question_id"], []),
            })
        return questions_by_image

    def add_image_results(self, variant, image_id, image_array, runner):
        for question in self.questions_by_image.get(image_id, []):
            predicted = runner.predict(image_array, question["question"])
            self.accuracies[variant].append(vqa_accuracy(predicted, question["answers"]))

    def summary_rows(self):
        rows = []
        for variant in self.variants:
            accs = self.accuracies[variant]
            rows.append({
                "variant": variant,
                "accuracy": float(np.mean(accs)) if accs else None,
                "questions": len(accs),
            })
        return rows

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
                "accuracy": float(np.mean(self.accuracies[variant])),
                "n": len(self.accuracies[variant]),
            }
            for variant in self.variants
            if self.accuracies[variant]
        }
