from dataloader.base import BaseDataLoader, DataLoaderRegistry, DataItem
from typing import Dict, Any, List
from collections import Counter

@DataLoaderRegistry.register("aegis-v1")
class AegisV1Loader(BaseDataLoader):
    """
    Hugging Face ID: nvidia/Aegis-AI-Content-Safety-Dataset-1.0
    
    数据集字段及含义:
    - id: 样本唯一标识
    - text: 对话文本内容 (prompt、response 或两者组合)
    - text_type: 文本类型 (user_message, llm_response, combined, multi_turn)
    - num_annotations: 标注者数量
    - labels_0 ~ labels_4: 5 个标注者各自的标签，未标注为 "None"
    """

    def _get_majority_label(self, row: Dict[str, Any]) -> str:
        """
        通过多数投票获取标签。
        只使用非 null 且非 "None" 的标注参与投票。
        """
        valid_labels = [
            row.get(f"labels_{i}") for i in range(5)
            if row.get(f"labels_{i}") and row.get(f"labels_{i}") != "None"
        ]
        
        if not valid_labels:
            return "None"
            
        # 统计频次并取最高者
        counts = Counter(valid_labels)
        majority_label, _ = counts.most_common(1)[0]
        return majority_label

    def mapping(self, row: Dict[str, Any]) -> DataItem:
        majority_label = self._get_majority_label(row)
        
        # 风险判定逻辑
        labels = [row.get(f"labels_{i}") for i in range(5) if row.get(f"labels_{i}") is not None]
        is_harmful = all(label != "Safe" for label in labels)

        return DataItem(
            query=row.get("text"),
            query_harmful_label=is_harmful,
            category=majority_label,
            extra={
                "text_type": row.get("text_type"),
                "labels": [row.get(f"labels_{i}") for i in range(5)]
            }
        )

    def load(self, **load_kwargs) -> List[DataItem]:
        """
        加载并映射 Aegis-v1 数据。

        DataItem 映射逻辑:
        - id: f"aegis-v1-{idx}" (由基类生成)
        - query: row['text']
        - query_harmful_label: True if all labels in [labels_0:labels_4] != "Safe"
        - category: 基于 labels_0~labels_4 多数投票的结果
        - extra: 包含 text_type 和原始 labels 列表
        """
        return super().load(**load_kwargs)


@DataLoaderRegistry.register("aegis-v2")
class AegisV2Loader(BaseDataLoader):
    """
    Hugging Face ID: nvidia/Aegis-AI-Content-Safety-Dataset-2.0
    
    数据集字段及含义:
    - id: 样本唯一标识 (UUID)
    - prompt: 用户输入文本
    - response: 模型回复文本
    - prompt_label: 安全标签 ("safe"/"unsafe")
    - response_label: 安全标签 ("safe"/"unsafe")
    - violated_categories: 违反的类别 (逗号分隔)
    - prompt_label_source: 标签来源 (human)
    - response_label_source: 标签来源 (human, llm_jury, refusal_data_augmentation)
    """

    def mapping(self, row: Dict[str, Any]) -> DataItem:
        # 1. 基础标签转换
        q_harmful = row.get("prompt_label") == "unsafe"
        r_harmful = row.get("response_label") == "unsafe"
        
        # 2. 拒绝标签判定：根据 response_label_source 识别拒绝增强数据
        is_refusal = row.get("response_label_source") == "refusal_data_augmentation"

        return DataItem(
            query=row.get("prompt"),
            response=row.get("response"),
            query_harmful_label=q_harmful,
            response_harmful_label=r_harmful,
            response_refusal_label=is_refusal,
            category=row.get("violated_categories", ""),
            extra={
                "reconstruction_id": row.get("reconstruction_id_if_redacted"),
                "prompt_label_source": row.get("prompt_label_source"),
                "response_label_source": row.get("response_label_source")
            }
        )

    def load(self, **load_kwargs) -> List[DataItem]:
        """
        加载并映射 Aegis-v2 数据。
        
        DataItem 映射逻辑:
        - id: f"aegis-v2-{idx}" (由基类生成)
        - query: row['prompt']
        - response: row['response']
        - query_harmful_label: True if row['prompt_label'] == "unsafe"
        - response_harmful_label: True if row['response_label'] == "unsafe"
        - response_refusal_label: True if row['response_label_source'] == "refusal_data_augmentation"
        - category: row['violated_categories']
        """
        return super().load(**load_kwargs)
