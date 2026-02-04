from dataloader.base import BaseDataLoader, DataLoaderRegistry, DataItem
from typing import Dict, Any, List, Optional, Tuple
import re

@DataLoaderRegistry.register("bingoguard")
class BingoGuardLoader(BaseDataLoader):
    """
    Hugging Face ID: (请在 CONST_HF_DATASET_IDS 中配置对应的路径)
    
    数据集字段及含义:
    - prompt: 用户输入文本
    - response: 模型回复文本
    - label: 通常与 prompt_label 相同，或者是严重程度标记 (#level: X)
    - prompt_label: 安全标签，包含 safe 或 unsafe + 类别
    """

    def _extract_severity(self, label_str: str) -> Optional[int]:
        """专门从 label 字段提取 #level 数值"""
        if label_str and "#level:" in label_str:
            match = re.search(r"#level:\s*(\d+)", label_str)
            return int(match.group(1)) if match else None
        return None

    def _parse_safety(self, prompt_label: str) -> Tuple[bool, str]:
        """从 prompt_label 中解析是否违规及类别"""
        if not prompt_label:
            return False, "safe"
        
        is_harmful = "unsafe" in prompt_label
        category = "safe"
        
        if ":" in prompt_label:
            category = prompt_label.split(":")[-1].strip()
        elif is_harmful:
            # 处理没有冒号但标了 unsafe 的异常情况
            category = "unsafe"
            
        return is_harmful, category

    def mapping(self, row: Dict[str, Any]) -> DataItem:
        p_label_raw = row.get("prompt_label", "")
        r_label_raw = row.get("label", "")
        response_text = row.get("response")

        # 1. 核心判定逻辑：均由 prompt_label 决定
        is_harmful, category_name = self._parse_safety(p_label_raw)
        
        # 2. Response 有害标签特殊处理：如果回复为空，标签设为 None
        # 否则，随 prompt_label 的判定结果
        if response_text is None or str(response_text).strip() == "" or str(response_text).lower() == "none":
            final_r_harmful = None
        else:
            final_r_harmful = is_harmful

        # 3. 提取严重程度
        severity = self._extract_severity(r_label_raw)

        return DataItem(
            query=row.get("prompt"),
            response=response_text,
            query_harmful_label=is_harmful,
            response_harmful_label=final_r_harmful,
            category=category_name,
            extra={
                "severity": severity,
                "prompt_label": p_label_raw,  # 完整存入 extra
                "label_field": r_label_raw
            }
        )

    def load(self, **load_kwargs) -> List[DataItem]:
        """
        加载并映射 BingoGuard 数据。
        
        DataItem 映射逻辑:
        - id: f"bingoguard-{idx}" (由基类生成)
        - query: row['prompt']
        - response: row['response']
        - query_harmful_label: True if 'unsafe' in prompt_label
        - response_harmful_label: 同上，但若 response 为空则为 None
        - category: 提取 prompt_label 冒号后的子类别
        - extra['severity']: 仅从 label 字段中解析出的 #level 数值
        """
        return super().load(**load_kwargs)
