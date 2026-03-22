"""
Dataset builder for PHI-3.5 agentic fine-tuning.
Now generates separate train and eval datasets.
"""

import json
import random
from typing import List, Dict, Any, Tuple
from pathlib import Path
from dataclasses import dataclass


@dataclass
class AgenticExample:
    instruction: str
    input: str
    output: str
    tools_used: List[str]
    complexity: str


class AgenticDatasetBuilder:
    def __init__(self, output_dir: str = "./data/processed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.available_tools = {
            "web_search": {
                "description": "Search the web for information",
                "parameters": ["query", "max_results"],
            },
            "file_reader": {
                "description": "Read and analyze files",
                "parameters": ["file_path", "operation"],
            },
        }

    # =========================
    # SINGLE STEP
    # =========================
    def create_tool_usage_example(self, tool_name: str, scenario: str) -> AgenticExample:
        instruction = f"Help me with: {scenario}"

        thought = f"I need to use {tool_name} to gather information about {scenario}."
        tool_call = self._generate_tool_call(tool_name, scenario)
        observation = self._generate_observation(tool_name, scenario)

        final_answer = f"Based on the gathered information, I have completed the task related to {scenario}."

        output = f"""
Thought: {thought}

Action:
{tool_call}

Observation: {observation}

Final Answer: {final_answer}
"""

        return AgenticExample(
            instruction=instruction,
            input="",
            output=output.strip(),
            tools_used=[tool_name],
            complexity="simple",
        )

    # =========================
    # MULTI STEP
    # =========================
    def create_multi_step_example(self, scenario: str, tools: List[str]) -> AgenticExample:
        instruction = f"Help me with this task: {scenario}"

        output_parts = []

        for i, tool in enumerate(tools):
            thought = f"Step {i+1}: I should use {tool} to progress on {scenario}."
            tool_call = self._generate_tool_call(tool, scenario)
            observation = self._generate_observation(tool, scenario)

            output_parts.append(f"Thought: {thought}")
            output_parts.append(f"Action:\n{tool_call}")
            output_parts.append(f"Observation: {observation}\n")

        output_parts.append(
            f"Final Answer: I have completed the multi-step task for {scenario} using the available tools."
        )

        return AgenticExample(
            instruction=instruction,
            input="",
            output="\n".join(output_parts),
            tools_used=tools,
            complexity="complex",
        )

    # =========================
    # TOOL CALL
    # =========================
    def _generate_tool_call(self, tool_name: str, scenario: str) -> str:
        params = self._generate_sample_parameters(tool_name, scenario)

        return f"""<tool_use>
<tool_name>{tool_name}</tool_name>
<parameters>
{json.dumps(params, indent=2)}
</parameters>
</tool_use>"""

    # =========================
    # OBSERVATION
    # =========================
    def _generate_observation(self, tool_name: str, scenario: str) -> str:
        if tool_name == "web_search":
            return random.choice([
                f"Found multiple sources discussing {scenario}.",
                f"Search results indicate updates regarding {scenario}.",
                f"Relevant articles highlight insights about {scenario}.",
            ])

        if tool_name == "file_reader":
            return random.choice([
                f"The file contains structured data related to {scenario}.",
                f"Logs indicate patterns related to {scenario}.",
                f"Configuration reveals parameters linked to {scenario}.",
            ])

        return "Tool execution completed."

    # =========================
    # PARAMETERS
    # =========================
    def _generate_sample_parameters(self, tool_name: str, scenario: str) -> Dict[str, Any]:
        return {
            "web_search": lambda s: {
                "query": s,
                "max_results": random.choice([3, 5, 10]),
            },
            "file_reader": lambda s: {
                "file_path": f"/data/{s.replace(' ', '_')}_{random.randint(1,100)}.txt",
                "operation": random.choice(["read", "summarize", "analyze"]),
            },
        }.get(tool_name, lambda s: {})(scenario)

    # =========================
    # SCENARIOS
    # =========================
    def _generate_random_scenario(self, tool_name: str) -> str:
        topics = ["AI", "finance", "healthcare", "sports", "climate change"]
        actions = ["analyze", "compare", "summarize", "investigate"]
        entities = ["India", "USA", "Tokyo", "Bangalore"]

        return f"{random.choice(actions)} {random.choice(topics)} trends in {random.choice(entities)}"

    def _generate_multi_step_scenario(self) -> str:
        tasks = [
            "analyze cost of living between cities",
            "evaluate system performance using logs",
            "compare financial performance across companies",
            "analyze user behavior patterns",
        ]
        return random.choice(tasks)

    # =========================
    # DATASET GENERATION
    # =========================
    def _to_dict(self, example: AgenticExample) -> Dict[str, Any]:
        
        return {
            "instruction": example.instruction,
            "input": example.input,
            "output": example.output,
            "tools_used": example.tools_used,
            "complexity": example.complexity,
        }
    def generate_dataset(self, num_examples: int = 1000) -> List[Dict[str, Any]]:
        examples = []

        single_tool_count = int(num_examples * 0.6)
        for _ in range(single_tool_count):
            tool = random.choice(list(self.available_tools.keys()))
            scenario = self._generate_random_scenario(tool)
            example = self.create_tool_usage_example(tool, scenario)
            examples.append(self._to_dict(example))

        multi_step_count = num_examples - single_tool_count
        for _ in range(multi_step_count):
            num_tools = random.randint(2, min(3, len(self.available_tools)))
            tools = random.sample(list(self.available_tools.keys()), num_tools)
            scenario = self._generate_multi_step_scenario()
            example = self.create_multi_step_example(scenario, tools)
            examples.append(self._to_dict(example))

        return examples

    # =========================
    # TRAIN / EVAL SPLIT
    # =========================
    def generate_train_eval_split(
        self, num_examples: int = 10000, train_ratio: float = 0.9
    ) -> Tuple[List[Dict], List[Dict]]:
        dataset = self.generate_dataset(num_examples)

        # 🔥 CRITICAL: shuffle before split
        random.shuffle(dataset)

        split_idx = int(len(dataset) * train_ratio)

        train_data = dataset[:split_idx]
        eval_data = dataset[split_idx:]

        return train_data, eval_data

    # =========================
    # SAVE
    # =========================
    def save_dataset(self, examples: List[Dict[str, Any]], filename: str):
        filepath = self.output_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(examples, f, indent=2, ensure_ascii=False)

        print(f"Saved {len(examples)} examples to {filepath}")

    def save_train_eval_datasets(self, train_data, eval_data):
        self.save_dataset(train_data, "train_dataset.json")
        self.save_dataset(eval_data, "eval_dataset.json")

        print("\nDataset split summary:")
        print(f"Train samples: {len(train_data)}")
        print(f"Eval samples: {len(eval_data)}")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    builder = AgenticDatasetBuilder()

    train_data, eval_data = builder.generate_train_eval_split(
        num_examples=10000,
        train_ratio=0.9,
    )

    builder.save_train_eval_datasets(train_data, eval_data)