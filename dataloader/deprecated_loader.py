"""
DataLoader Implementations
各个数���集的具体加载器实现
"""
import gzip
import json
from pathlib import Path
from typing import List

import pandas as pd

from .base import BaseDataLoader, HF_DATASET_IDS
from utils.DataItem import DataItem


class AegisV1Loader(BaseDataLoader):
    """
    Aegis V1 数据集加载器

    HF Dataset ID: nvidia/Aegis-AI-Content-Safety-Dataset-1.0

    数据集字段:
        - id: str, 样本唯一标识
        - text: str, 对话文本内容 (prompt/response/组合)
        - text_type: str, 文本类型 (user_message|llm_response|combined|multi_turn)
        - labels_0~labels_4: str, 5个标注者的标签 (Safe|Needs Caution|风险类别|None)
        - num_annotations: int, 实际标注者数量 (3~5)
    """
    DATASET_NAME = "aegis-v1"
    HF_DATASET_ID = HF_DATASET_IDS[DATASET_NAME]
    HF_SPLIT = "train"

    def load(self) -> List[DataItem]:
        """
        加载 Aegis V1 数据集

        DataItem 字段映射:
            - id: "{DATASET_NAME}-{idx:06d}"
            - query: text
            - query_harmful_label: majority_vote(labels_0:labels_4) != "Safe"
            - category: majority_vote(labels_0:labels_4)
            - extra.text_type: text_type
            - extra.labels: [labels_0, labels_1, labels_2, labels_3, labels_4]
        """
        dataset = self._load_dataset()

        items = []
        for idx, row in enumerate(dataset, start=1):
            items.append(self._parse_row(idx, row))
        return items

    def _parse_row(self, idx: int, row) -> DataItem:
        """解析单行数据为 DataItem"""
        # 收集所有标签 (包括 None)
        labels = [row.get(f"labels_{i}") for i in range(5)]

        # 过滤有效标签用于投票
        valid_labels = [lb for lb in labels if lb and lb != "None"]

        # 多数投票
        voted_label = self._majority_vote(valid_labels)

        # Safe -> False, 其他 -> True
        is_harmful = voted_label != "Safe" if voted_label else None

        return DataItem(
            id=self._generate_id(idx),
            query=row.get("text", ""),
            query_harmful_label=is_harmful,
            category=voted_label,
            extra={
                "text_type": row.get("text_type"),
                "labels": labels,
            },
        )

    def _generate_id(self, idx: int) -> str:
        """生成 ID: aegis_v1-000001"""
        return f"{self.DATASET_NAME.replace('-', '_')}-{idx:06d}"

    @staticmethod
    def _majority_vote(labels: list[str]) -> str | None:
        """多数投票，返回出现次数最多的标签"""
        if not labels:
            return None
        from collections import Counter
        counter = Counter(labels)
        return counter.most_common(1)[0][0]


class AegisV2Loader(BaseDataLoader):
    """
    Aegis V2 数据集加载器 (Aegis 2.0 / Nemotron Content Safety Dataset V2)

    HF Dataset ID: nvidia/Aegis-AI-Content-Safety-Dataset-2.0
    """
    HF_DATASET_ID = "nvidia/Aegis-AI-Content-Safety-Dataset-2.0"
    HF_SPLIT = "train"
    DATASET_NAME = "aegis-v2"

    def __init__(
        self,
        dataset_path: str | Path = None,
        hf_dataset_id: str = None,
        hf_split: str = None,
        split: str = "train",  # train, validation, test
    ):
        super().__init__(dataset_path, hf_dataset_id, hf_split)
        self.split = split

    def load(self) -> List[DataItem]:
        """
        加载 Aegis V2 数据集

        Aegis 2.0 包含 33,416 个标注样本，涵盖 12 个主要风险类别

        Returns:
            List[DataItem]: 数据项列表
        """
        items = []

        # 优先从 HuggingFace 加载
        if self.hf_dataset_id or self.HF_DATASET_ID:
            try:
                dataset = self._load_from_hf(
                    self.hf_dataset_id or self.HF_DATASET_ID,
                    self.split,  # 使用用户指定的 split
                )
                for idx, row in enumerate(dataset, start=1):
                    # prompt_label/response_label: "safe"/"unsafe"/null
                    item = DataItem(
                        id=self._generate_id(idx),
                        query=row.get("prompt", ""),
                        response=row.get("response", ""),
                        query_harmful_label=self._parse_label(row.get("prompt_label")),
                        response_harmful_label=self._parse_label(row.get("response_label")),
                        category=row.get("violated_categories"),
                        extra={
                            "prompt_label_source": row.get("prompt_label_source"),
                            "response_label_source": row.get("response_label_source"),
                            "reconstruction_id_if_redacted": row.get("reconstruction_id_if_redacted"),
                            "split": self.split,
                            "original_id": row.get("id"),
                        },
                    )
                    items.append(item)
                return items
            except Exception as e:
                print(f"Warning: Failed to load from HF ({e}), falling back to local files")

        # 备选：从本地文件加载
        if self.dataset_path:
            self._validate_path()
            parquet_files = list(self.dataset_path.glob("*.parquet"))
            for file in parquet_files:
                df = pd.read_parquet(file)
                for idx, (_, row) in enumerate(df.iterrows(), start=1):
                    item = DataItem(
                        id=self._generate_id(idx),
                        query=row.get("prompt", ""),
                        response=row.get("response", ""),
                        query_harmful_label=self._parse_label(row.get("prompt_label")),
                        response_harmful_label=self._parse_label(row.get("response_label")),
                        category=row.get("violated_categories"),
                        extra={
                            "original_id": row.get("id"),
                        },
                    )
                    items.append(item)

        return items

    @staticmethod
    def _parse_label(label: str) -> bool | None:
        """将标签字符串转换为 bool"""
        if label is None:
            return None
        if isinstance(label, bool):
            return label
        if isinstance(label, str):
            return label.lower() == "unsafe"
        return None


class BeavertailsLoader(BaseDataLoader):
    """Beavertails 数据集加载器"""

    def load(self) -> List[DataItem]:
        self._validate_path()
        items = []

        # 查找所有 jsonl.gz 文件
        jsonl_files = list(self.dataset_path.rglob("*.jsonl.gz"))
        for file in jsonl_files:
            with gzip.open(file, "rt", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    item = DataItem(
                        query=data.get("prompt", data.get("instruction", "")),
                        response=data.get("response", data.get("output", "")),
                        query_harmful_label=data.get("harmful", None),
                    )
                    items.append(item)

        return items


class BingoGuardLoader(BaseDataLoader):
    """BingoGuard 数据集加载器"""

    def load(self) -> List[DataItem]:
        self._validate_path()
        items = []

        # TODO: 根据 BingoGuard 实际格式实现
        jsonl_files = list(self.dataset_path.rglob("*.jsonl*"))
        for file in jsonl_files:
            opener = gzip.open if file.suffix == ".gz" else open
            mode = "rt" if file.suffix == ".gz" else "r"
            with opener(file, mode, encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    item = DataItem(
                        query=data.get("prompt", ""),
                        response=data.get("response", ""),
                    )
                    items.append(item)

        return items


class HarmBenchLoader(BaseDataLoader):
    """HarmBench 数据集加载器"""

    def load(self) -> List[DataItem]:
        self._validate_path()
        items = []

        # TODO: 根据 HarmBench 实际格式实现
        parquet_files = list(self.dataset_path.rglob("*.parquet"))
        for file in parquet_files:
            df = pd.read_parquet(file)
            for _, row in df.iterrows():
                item = DataItem(
                    query=row.get("prompt", ""),
                    category=row.get("category", None),
                )
                items.append(item)

        return items


class JailbreakBenchLoader(BaseDataLoader):
    """JailbreakBench 数据集加载器"""

    def load(self) -> List[DataItem]:
        self._validate_path()
        items = []

        # TODO: 根据 JailbreakBench 实际格式实现
        json_files = list(self.dataset_path.rglob("*.json"))
        for file in json_files:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for entry in data:
                        item = DataItem(
                            query=entry.get("prompt", ""),
                            category=entry.get("category", None),
                        )
                        items.append(item)
                elif isinstance(data, dict):
                    if "data" in data and isinstance(data["data"], list):
                        for entry in data["data"]:
                            item = DataItem(
                                query=entry.get("prompt", ""),
                                category=entry.get("category", None),
                            )
                            items.append(item)

        return items


class OpenAIModerationLoader(BaseDataLoader):
    """OpenAI Moderation 数据集加载器"""

    def load(self) -> List[DataItem]:
        self._validate_path()
        items = []

        # TODO: 根据 OpenAI Moderation 实际格式实现
        jsonl_files = list(self.dataset_path.rglob("*.jsonl*"))
        for file in jsonl_files:
            opener = gzip.open if file.suffix == ".gz" else open
            mode = "rt" if file.suffix == ".gz" else "r"
            with opener(file, mode, encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    item = DataItem(
                        query=data.get("text", data.get("prompt", "")),
                        query_harmful_label=data.get("flagged", None),
                    )
                    items.append(item)

        return items


class OrBenchLoader(BaseDataLoader):
    """OrBench 数据集加载器"""

    def load(self) -> List[DataItem]:
        self._validate_path()
        items = []

        # TODO: 根据 OrBench 实际格式实现
        json_files = list(self.dataset_path.rglob("*.json"))
        for file in json_files:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for entry in data:
                        item = DataItem(
                            query=entry.get("prompt", ""),
                            category=entry.get("category", None),
                        )
                        items.append(item)

        return items


class SaferLHFLoader(BaseDataLoader):
    """SaferLHF 数据集加载器"""

    def load(self) -> List[DataItem]:
        self._validate_path()
        items = []

        # TODO: 根据 SaferLHF 实际格式实现
        jsonl_files = list(self.dataset_path.rglob("*.jsonl*"))
        for file in jsonl_files:
            opener = gzip.open if file.suffix == ".gz" else open
            mode = "rt" if file.suffix == ".gz" else "r"
            with opener(file, mode, encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    item = DataItem(
                        query=data.get("prompt", ""),
                        response=data.get("response", ""),
                    )
                    items.append(item)

        return items


class SEvalLoader(BaseDataLoader):
    """S-Eval 数据集加载器"""

    def load(self) -> List[DataItem]:
        self._validate_path()
        items = []

        # TODO: 根据 S-Eval 实际格式实现
        json_files = list(self.dataset_path.rglob("*.json"))
        for file in json_files:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for entry in data:
                        item = DataItem(
                            query=entry.get("prompt", ""),
                            category=entry.get("category", None),
                        )
                        items.append(item)

        return items


class SimpleSafetyTestsLoader(BaseDataLoader):
    """SimpleSafetyTests 数据集加载器"""

    def load(self) -> List[DataItem]:
        self._validate_path()
        items = []

        # TODO: 根据 SimpleSafetyTests 实际格式实现
        json_files = list(self.dataset_path.rglob("*.json"))
        for file in json_files:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for entry in data:
                        item = DataItem(
                            query=entry.get("prompt", ""),
                            category=entry.get("category", None),
                        )
                        items.append(item)

        return items


class SorryBenchLoader(BaseDataLoader):
    """SorryBench 数据集加载器"""

    def load(self) -> List[DataItem]:
        self._validate_path()
        items = []

        # TODO: 根据 SorryBench 实际格式实现
        json_files = list(self.dataset_path.rglob("*.json"))
        for file in json_files:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for entry in data:
                        item = DataItem(
                            query=entry.get("prompt", ""),
                            category=entry.get("category", None),
                        )
                        items.append(item)

        return items


class SorryBenchHumanJudgementLoader(BaseDataLoader):
    """SorryBench-Human-Judgement 数据集加载器"""

    def load(self) -> List[DataItem]:
        self._validate_path()
        items = []

        # TODO: 根据 SorryBench-Human-Judgement 实际格式实现
        json_files = list(self.dataset_path.rglob("*.json"))
        for file in json_files:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for entry in data:
                        item = DataItem(
                            query=entry.get("prompt", ""),
                            response=entry.get("response", ""),
                            category=entry.get("category", None),
                        )
                        items.append(item)

        return items


class StrongRejectLoader(BaseDataLoader):
    """StrongReject 数据集加载器"""

    def load(self) -> List[DataItem]:
        self._validate_path()
        items = []

        # TODO: 根据 StrongReject 实际格式实现
        json_files = list(self.dataset_path.rglob("*.json"))
        for file in json_files:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for entry in data:
                        item = DataItem(
                            query=entry.get("prompt", ""),
                            category=entry.get("category", None),
                        )
                        items.append(item)

        return items


class ThinkLoader(BaseDataLoader):
    """Think 数据集加载器"""

    def load(self) -> List[DataItem]:
        self._validate_path()
        items = []

        # TODO: 根据 Think 实际格式实现
        json_files = list(self.dataset_path.rglob("*.json"))
        for file in json_files:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for entry in data:
                        item = DataItem(
                            query=entry.get("prompt", ""),
                            response=entry.get("response", ""),
                        )
                        items.append(item)

        return items


class ToxicChatLoader(BaseDataLoader):
    """
    ToxicChat 数据集加载器

    HF Dataset ID: lmsys/toxic-chat
    Subset: toxicchat0124 (推荐)
    """
    HF_DATASET_ID = "lmsys/toxic-chat"
    HF_SPLIT = "train"
    DATASET_NAME = "toxicchat"

    def __init__(
        self,
        dataset_path: str | Path = None,
        hf_dataset_id: str = None,
        hf_split: str = None,
        subset: str = "toxicchat0124",  # toxicchat0124 或 toxicchat1123
    ):
        super().__init__(dataset_path, hf_dataset_id, hf_split)
        self.subset = subset

    def load(self) -> List[DataItem]:
        """
        加载 ToxicChat 数据集

        优先使用 HuggingFace dataset，如果不可用则使用本地文件

        Returns:
            List[DataItem]: 数据项列表
        """
        items = []

        # 优先从 HuggingFace 加载
        if self.hf_dataset_id or self.HF_DATASET_ID:
            try:
                dataset = self._load_from_hf(
                    self.hf_dataset_id or self.HF_DATASET_ID,
                    self.hf_split or self.HF_SPLIT,
                    name=self.subset,  # subset 作为 config name
                )
                for idx, row in enumerate(dataset, start=1):
                    item = DataItem(
                        id=self._generate_id(idx),
                        query=row.get("user_input", ""),
                        response=row.get("model_output", ""),
                        query_harmful_label=bool(row.get("toxicity", 0)) if row.get("toxicity") is not None else None,
                        extra={
                            "jailbreaking": row.get("jailbreaking"),
                            "openai_moderation": row.get("openai_moderation"),
                            "conv_id": row.get("conv_id"),
                            "human_annotation": row.get("human_annotation"),
                            "subset": self.subset,
                        },
                    )
                    items.append(item)
                return items
            except Exception as e:
                print(f"Warning: Failed to load from HF ({e}), falling back to local files")

        # 备选：从本地文件加载
        if self.dataset_path:
            self._validate_path()
            # 本地 parquet 文件
            parquet_files = list(self.dataset_path.rglob("*.parquet"))
            for file in parquet_files:
                df = pd.read_parquet(file)
                for idx, (_, row) in enumerate(df.iterrows(), start=1):
                    item = DataItem(
                        id=self._generate_id(idx),
                        query=row.get("user_input", ""),
                        response=row.get("model_output", ""),
                        query_harmful_label=bool(row.get("toxicity", 0)) if pd.notna(row.get("toxicity")) else None,
                        extra={
                            "jailbreaking": row.get("jailbreaking"),
                            "openai_moderation": row.get("openai_moderation"),
                            "conv_id": row.get("conv_id"),
                            "human_annotation": row.get("human_annotation"),
                        },
                    )
                    items.append(item)

        return items


class WildGuardLoader(BaseDataLoader):
    """WildGuard 数据集加载器"""

    def load(self) -> List[DataItem]:
        self._validate_path()
        items = []

        # TODO: 根据 WildGuard 实际格式实现
        parquet_files = list(self.dataset_path.rglob("*.parquet"))
        for file in parquet_files:
            df = pd.read_parquet(file)
            for _, row in df.iterrows():
                item = DataItem(
                    query=row.get("prompt", ""),
                    response=row.get("response", ""),
                    query_harmful_label=row.get("prompt_harmful", None),
                )
                items.append(item)

        return items


class WildJailbreakLoader(BaseDataLoader):
    """WildJailbreak 数据集加载器"""

    def load(self) -> List[DataItem]:
        self._validate_path()
        items = []

        # TODO: 根据 WildJailbreak 实际格式实现
        json_files = list(self.dataset_path.rglob("*.json"))
        for file in json_files:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for entry in data:
                        item = DataItem(
                            query=entry.get("prompt", ""),
                            category=entry.get("category", None),
                        )
                        items.append(item)

        return items


class XSTestLoader(BaseDataLoader):
    """XSTest 数据集加载器"""

    def load(self) -> List[DataItem]:
        self._validate_path()
        items = []

        # XSTest 使用 parquet 格式
        parquet_files = list(self.dataset_path.rglob("*.parquet"))
        for file in parquet_files:
            df = pd.read_parquet(file)
            for _, row in df.iterrows():
                item = DataItem(
                    query=row.get("prompt", ""),
                    category=row.get("category", None),
                )
                items.append(item)

        return items


class XSTestResponseLoader(BaseDataLoader):
    """XSTest-Response 数据集加载器"""

    def load(self) -> List[DataItem]:
        self._validate_path()
        items = []

        # TODO: 根据 XSTest-Response 实际格式实现
        parquet_files = list(self.dataset_path.rglob("*.parquet"))
        for file in parquet_files:
            df = pd.read_parquet(file)
            for _, row in df.iterrows():
                item = DataItem(
                    query=row.get("prompt", ""),
                    response=row.get("response", ""),
                )
                items.append(item)

        return items


__all__ = [
    "AegisV1Loader",
    "AegisV2Loader",
    "BeavertailsLoader",
    "BingoGuardLoader",
    "HarmBenchLoader",
    "JailbreakBenchLoader",
    "OpenAIModerationLoader",
    "OrBenchLoader",
    "SaferLHFLoader",
    "SEvalLoader",
    "SimpleSafetyTestsLoader",
    "SorryBenchLoader",
    "SorryBenchHumanJudgementLoader",
    "StrongRejectLoader",
    "ThinkLoader",
    "ToxicChatLoader",
    "WildGuardLoader",
    "WildJailbreakLoader",
    "XSTestLoader",
    "XSTestResponseLoader",
]
