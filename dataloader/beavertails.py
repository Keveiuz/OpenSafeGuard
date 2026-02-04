from dataloader.base import BaseDataLoader, DataLoaderRegistry, DataItem
from typing import Dict, Any, List

@DataLoaderRegistry.register("beavertails")
class BeaverTailsLoader(BaseDataLoader):
    """
    Hugging Face ID: PKU-Alignment/BeaverTails
    
    数据集字段及含义:
    - prompt: 用户输入的问题/提示
    - response: 模型的回复
    - category: 包含 14 个布尔值的字典，标记该样本违反的危害类别
    - is_safe: 整体安全标签，true 为安全，false 为不安全
    """

    def _extract_categories(self, category_dict: Dict[str, bool]) -> str:
        """提取所有为 True 的危害类别并用逗号分隔"""
        if not category_dict:
            return ""
        # 过滤出所有值为 True 的类别名
        violated = [cat for cat, triggered in category_dict.items() if triggered]
        return ";".join(violated)

    def mapping(self, row: Dict[str, Any]) -> DataItem:
        # 1. 安全标签转换 (注意：is_safe 是整体标签，需反转后赋给 Q 和 R)
        is_safe = row.get("is_safe", True)
        overall_harmful = not is_safe
        
        # 2. 提取具体的危害类别字符串
        category_dict = row.get("category", {})
        violated_str = self._extract_categories(category_dict)

        return DataItem(
            query=row.get("prompt"),
            response=row.get("response"),
            # 统一同步 is_safe 标签到 query 和 response
            query_harmful_label=overall_harmful,
            response_harmful_label=overall_harmful,
            category=violated_str if len(violated_str) else None,
            extra={
                "raw_category_map": category_dict
            }
        )

    def load(self, **load_kwargs) -> List[DataItem]:
        """
        加载并映射 BeaverTails 数据。
        
        DataItem 映射逻辑:
        - id: f"beavertails-{idx}" (由基类生成)
        - query: row['prompt']
        - response: row['response']
        - query_harmful_label: True if row['is_safe'] == False
        - response_harmful_label: True if row['is_safe'] == False
        - category: 提取 row['category'] 中所有为 True 的键名并用 ';' 拼接
        """
        return super().load(**load_kwargs)
