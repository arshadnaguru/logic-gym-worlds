"""
src/evaluation/llm_interface.py

LLM evaluation interface for Logic Gym Worlds.
Supports: OpenAI (GPT-4), Anthropic (Claude), and a rule-based oracle for testing.
"""

import os
import re
from abc import ABC, abstractmethod
from typing import Optional


VALID_LABELS = {"entails", "contradicts", "unknown"}


# ── Abstract base ──────────────────────────────────────────────────────────

class LLMInterface(ABC):
    @abstractmethod
    def query(self, prompt: str) -> str:
        pass

    def query_with_label(self, prompt: str) -> str:
        """Query and return a cleaned label."""
        raw = self.query(prompt)
        return extract_answer(raw)


# ── Implementations ────────────────────────────────────────────────────────

class OpenAIInterface(LLMInterface):
    def __init__(self, model: str = "gpt-4"):
        import openai
        self.client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = model

    def query(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=512,
        )
        return response.choices[0].message.content


class AnthropicInterface(LLMInterface):
    def __init__(self, model: str = "claude-sonnet-4-6"):
        import anthropic
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model = model

    def query(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text


class OracleInterface(LLMInterface):
    """
    Returns gold labels directly. Used for pipeline testing.
    In real eval, replace with a real LLM.
    """
    def __init__(self, gold_map: dict):
        self.gold_map = gold_map  # query_id -> gold_label

    def query(self, prompt: str) -> str:
        # Extract query id from prompt (brittle, for testing only)
        match = re.search(r"\[QID:(\S+)\]", prompt)
        if match:
            return self.gold_map.get(match.group(1), "unknown")
        return "unknown"


# ── Prompt builder ─────────────────────────────────────────────────────────

def build_prompt(instance, query, style: str = "direct") -> str:
    """
    Build an evaluation prompt for a given instance and query.

    styles:
      "direct" — just answer with a label
      "cot"    — chain-of-thought reasoning
      "few_shot" — with 1 worked example
    """
    rules_text = "\n".join(
        f"  {i+1}. {rule.natural_language}"
        for i, rule in enumerate(instance.rule_sheet.rules)
    )
    facts_text = "\n".join(
        f"  - {f['args'][0]} is {instance.rule_sheet.predicates[f['predicate']]}."
        for f in instance.facts
        if f["positive"]
    )

    if style == "direct":
        return f"""Given the following rules and facts, answer the question.

RULES:
{rules_text}

FACTS:
{facts_text}

QUESTION: {query.natural_language}

Answer with exactly one word: ENTAILS, CONTRADICTS, or UNKNOWN.
- ENTAILS: the statement must be true given the rules and facts.
- CONTRADICTS: the statement must be false given the rules and facts.
- UNKNOWN: cannot be determined from the given information.

Your answer:"""

    elif style == "cot":
        return f"""Given the following rules and facts, answer the question step by step.

RULES:
{rules_text}

FACTS:
{facts_text}

QUESTION: {query.natural_language}

Think step by step:
1. Which rules are relevant to this question?
2. Trace the chain of reasoning from the facts through the rules.
3. Can you definitively conclude the statement is true, false, or undetermined?

After your reasoning, give your final answer as exactly one of: ENTAILS, CONTRADICTS, or UNKNOWN."""

    elif style == "few_shot":
        example = """
EXAMPLE:
Rules: If something is a bird, it can fly. If something is a penguin, it is a bird. If something is a penguin, it cannot fly.
Facts: Tweety is a bird. Opus is a penguin.
Question: Can Opus fly?
Answer: CONTRADICTS (Opus is a penguin → cannot fly, overriding the bird → can fly rule.)

---
Now answer:
"""
        return f"""{example}
RULES:
{rules_text}

FACTS:
{facts_text}

QUESTION: {query.natural_language}

Answer (ENTAILS / CONTRADICTS / UNKNOWN):"""

    else:
        raise ValueError(f"Unknown prompt style: {style}")


# ── Label extractor ────────────────────────────────────────────────────────

def extract_answer(response: str) -> str:
    """Parse the model's free-text response into a label."""
    r = response.upper()
    # Prefer the first match to handle CoT responses
    for label in ["ENTAILS", "CONTRADICTS", "UNKNOWN"]:
        if label in r:
            return label.lower()
    return "invalid"


# ── Factory ────────────────────────────────────────────────────────────────

def get_llm(model_name: str) -> LLMInterface:
    """Instantiate the right LLM interface from a model string."""
    if "gpt" in model_name.lower():
        return OpenAIInterface(model=model_name)
    elif "claude" in model_name.lower():
        return AnthropicInterface(model=model_name)
    else:
        raise ValueError(f"Unknown model: {model_name}. Use a GPT or Claude model string.")
