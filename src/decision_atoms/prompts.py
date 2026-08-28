from __future__ import annotations

import string
from typing import Any, Sequence

from .types import DecisionAtom, Sample


CLASSIFICATION_VARIANTS = (
    "Classify the image using only the listed classes. Return exactly the code of the best class.",
    "Which listed class best matches the image? Answer with one class code only.",
    "Inspect the image and select its most likely class from the options. Output only the code.",
)

ATOM_MEASUREMENT_VARIANTS = (
    "Does the visible evidence for {concept} more strongly match side A ({a_side}), side B ({b_side}), or is it unclear? Return only A, B, or U.",
    "Judge only {concept}. Choose A if it looks like {a_side}, B if it looks like {b_side}, and U when the evidence is not visible or ambiguous. Return one letter.",
    "For the property {concept}, decide whether this image supports A: {a_side}; B: {b_side}; or U: unclear. Answer exactly A, B, or U.",
)


def class_prompt(labels: list[str], variant: int = 0) -> tuple[str, dict[str, str]]:
    # Qwen tokenizes individual ASCII letters as one token. Keeping up to 52
    # classes on this path makes Oxford-Pets-scale scoring one forward pass instead
    # of one full multimodal forward per class. The numbered fallback keeps the
    # method defined for larger label spaces.
    single_token_codes = string.ascii_uppercase + string.ascii_lowercase
    codes = {
        label: (single_token_codes[index] if len(labels) <= len(single_token_codes) else f"C{index:03d}")
        for index, label in enumerate(labels)
    }
    options = "\n".join(f"{code}: {label}" for label, code in codes.items())
    instruction = CLASSIFICATION_VARIANTS[variant % len(CLASSIFICATION_VARIANTS)]
    return f"{instruction}\nOptions:\n{options}\nAnswer:", codes


def discovery_prompt(
    label_a: str,
    label_b: str,
    atoms_per_bootstrap: int,
    source_mode: str,
) -> str:
    if source_mode == "shift_audit":
        return f"""You are auditing visual evidence learned by a frozen classifier.
Group A images have class {label_a}; group B images have class {label_b}.
List up to {atoms_per_bootstrap} independently judgeable visual properties that differ between the two groups. Search neutrally across every visible part of the image: the main object, other objects, surroundings, composition, capture conditions, and style. Do not assume in advance which properties define the class and which are accidental correlations; a separate held-out test will decide that. Each property must be answerable on one unseen image and must describe both sides concretely. Do not combine unrelated cues.
For each property, also classify its semantic location as "object" when it is intrinsic morphology or a part of the main subject, "context" when it is surroundings, habitat, co-occurring objects, capture/style, or an accessory, and "uncertain" otherwise. This role is only a semantic prior; a separate quantitative audit will still decide whether the property is predictive.
Return only a JSON array. Every item must have exactly these string fields:
{{"concept": "short property name", "a_side": "what A visibly looks like", "b_side": "what B visibly looks like", "role": "object or context or uncertain"}}
"""
    difficulty = (
        "These are difficult, low-margin examples for the classifier. Focus on subtle properties "
        "that separate the groups among these difficult examples; ignore properties shared by both groups."
        if source_mode == "boundary"
        else "Compare the two image groups using consistently visible evidence."
    )
    return f"""You are analyzing a frozen visual classifier, not writing a generic class guide.
Group A images have class {label_a}; group B images have class {label_b}.
{difficulty}
Identify up to {atoms_per_bootstrap} independently judgeable visual properties. Each property must be answerable on one unseen image. Prefer object morphology, parts, texture, shape, markings, pose, and also report background/context only when it is consistently used by these examples. Do not combine unrelated cues in one property.
For each property, also classify its semantic location as "object" when it is intrinsic morphology or a part of the main subject, "context" when it is surroundings, habitat, co-occurring objects, capture/style, or an accessory, and "uncertain" otherwise.
Return only a JSON array. Every item must have exactly these string fields:
{{"concept": "short property name", "a_side": "what A visibly looks like", "b_side": "what B visibly looks like", "role": "object or context or uncertain"}}
"""


def atom_prompt(atom: DecisionAtom, variant: int) -> str:
    template = ATOM_MEASUREMENT_VARIANTS[variant % len(ATOM_MEASUREMENT_VARIANTS)]
    return template.format(concept=atom.concept, a_side=atom.a_side, b_side=atom.b_side)


def icl_prompt(labels: list[str], demo_labels: list[str]) -> tuple[list[str], str, dict[str, str]]:
    single_token_codes = string.ascii_uppercase + string.ascii_lowercase
    codes = {
        label: (single_token_codes[index] if len(labels) <= len(single_token_codes) else f"C{index:03d}")
        for index, label in enumerate(labels)
    }
    demo_texts = [f"Label: {codes[label]} ({label})" for label in demo_labels]
    options = "\n".join(f"{code}: {label}" for label, code in codes.items())
    query = (
        "Classify the final query image using the demonstrations and the options below. "
        "Return only the best class code.\nOptions:\n" + options + "\nAnswer:"
    )
    return demo_texts, query, codes


def choice_codes(options: Sequence[str]) -> dict[str, str]:
    """Return stable one-character codes for a closed candidate set."""
    normalized = [str(option) for option in options]
    if len(normalized) < 2:
        raise ValueError("a task sample needs at least two candidate options")
    if len(set(normalized)) != len(normalized):
        raise ValueError("candidate options must be unique")
    alphabet = string.ascii_uppercase + string.ascii_lowercase
    if len(normalized) > len(alphabet):
        raise ValueError("paper task scoring supports at most 52 options")
    return {option: alphabet[index] for index, option in enumerate(normalized)}


def sample_options(sample: Sample, default_labels: Sequence[str]) -> list[str]:
    options = list(sample.options) if sample.options else [str(value) for value in default_labels]
    if sample.label not in options:
        raise ValueError(f"sample {sample.id} label {sample.label!r} is absent from its options")
    return options


def _option_text(options: Sequence[str], codes: dict[str, str]) -> str:
    return "\n".join(f"{codes[option]}: {option}" for option in options)


def task_icl_content(
    query: Sample,
    demos: Sequence[Sample],
    default_labels: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Build the model-family-neutral logical prompt for paper experiments.

    Images use an abstract ``image`` block. Backends translate this block into
    their processor's native representation while leaving every text block
    byte-identical across target-model families.
    """
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "For every image below, choose only from that image's listed options."
            ),
        }
    ]
    for index, demo in enumerate(demos, start=1):
        options = sample_options(demo, default_labels)
        codes = choice_codes(options)
        context = f"\nContext: {demo.text}" if demo.text else ""
        content.extend(
            [
                {"type": "text", "text": f"Demonstration {index}:{context}"},
                {"type": "image", "image": demo.image},
                {
                    "type": "text",
                    "text": (
                        f"Options:\n{_option_text(options, codes)}\n"
                        f"Answer: {codes[demo.label]}"
                    ),
                },
            ]
        )

    options = sample_options(query, default_labels)
    codes = choice_codes(options)
    context = f"\nContext: {query.text}" if query.text else ""
    content.extend(
        [
            {"type": "text", "text": f"Final query:{context}"},
            {"type": "image", "image": query.image},
            {
                "type": "text",
                "text": (
                    "Choose the best option for the final query image. Return exactly one code.\n"
                    f"Options:\n{_option_text(options, codes)}\nAnswer:"
                ),
            },
        ]
    )
    return content, codes
