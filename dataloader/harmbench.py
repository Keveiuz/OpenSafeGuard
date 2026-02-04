import json
import pandas as pd
from tqdm import tqdm
from base import BaseDataLoader, DataLoaderRegistry, DataItem
from typing import Dict, Any, List, Optional, Tuple

@DataLoaderRegistry.register("harmbench")
class HarmBenchLoader(BaseDataLoader):
    """
    Hugging Face ID: walledai/HarmBench (注：本 Loader 主要用于加载官方提供的本地异构文件)
    
    数据集字段及含义:
    
    【只含 query 的 CSV】 (HarmBench/data/behavior_datasets/harmbench_behaviors_text_test.csv):
    - Behavior: 危害行为提示文本
    - FunctionalCategory: 功能类别 (standard/copyright/contextual)
    - SemanticCategory: 语义类别 (危害类型)
    - Tags: 标签 (多标签，逗号分隔)
    - ContextString: 上下文字符串 (仅 contextual 类型有)
    - BehaviorID: 行为唯一标识符
    
    【含有 response 的 JSON】 (HarmBench/data/classifier_val_sets/text_behaviors_val_set.json):
    - test_case: 具体的攻击提示文本 (Prompt)
    - generation: 目标模型针对该提示生成的回复内容
    - behavior_id: 对应的危害行为唯一标识符
    - experiment: 实验分组 (判断是否为 adversarial)
    - human_0, human_1, human_2: 三位不同人工标注员的判断结果 (0/1)
    - method: 使用的攻击方法
    """

    def __init__(self, query_task_path: Optional[str] = None, response_task_path: Optional[str] = None, **kwargs):
        """
        初始化 HarmBench 加载器。
        
        Args:
            query_task_path: CSV 文件路径 (只有危害行为 Behavior)
            response_task_path: JSON 文件路径 (含 Prompt 和 Response)
            **kwargs: 传递给基类的其他参数（如 split 等）
        """
        # 调用基类初始化，将路径存入 init_kwargs 供 load 使用
        super().__init__(**kwargs)
        self.init_kwargs.update({
            "query_task_path": query_task_path,
            "response_task_path": response_task_path
        })

    def _parse_query_csv(self, path: str, start_idx: int) -> List[DataItem]:
        """解析 Behavior CSV 文件并映射为 DataItem"""
        df = pd.read_csv(path)
        # 过滤仅保留 "standard" 或 "copyright" 类别
        df = df[df["FunctionalCategory"].isin(["standard", "copyright"])]
        
        data_items = []
        # 使用 tqdm 保持与基类一致的进度显示
        for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc=f"Loading {self.REGISTER_NAME}.csv"), start_idx):
            item = DataItem(
                query=row.get("Behavior"),
                response=None,
                query_harmful_label=True,  # 危害行为库默认为有害意图
                response_harmful_label=None,
                category=row.get("SemanticCategory", ""),
                extra={
                    "behavior_id": row.get("BehaviorID"),
                    "functional_category": row.get("FunctionalCategory"),
                    "tags": row.get("Tags"),
                    "context_string": row.get("ContextString"),
                    "source_file": "behavior_csv"
                }
            )
            item.id = self.generate_id(i)
            data_items.append(item)
        return data_items

    def _parse_response_json(self, path: str, start_idx: int) -> List[DataItem]:
        """解析带 Response 的 JSON 文件并映射为 DataItem"""
        with open(path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # 将嵌套的字典平铺为列表以便于 tqdm 计数和显示进度
        flat_cases = []
        for behavior_id, cases in raw_data.items():
            for case in cases:
                case["_behavior_id_key"] = behavior_id # 记录外层 key 作为 ID 关联
                flat_cases.append(case)

        data_items = []
        for i, case in enumerate(tqdm(flat_cases, desc=f"Loading {self.REGISTER_NAME}.json"), start_idx):
            # 多数投票逻辑：若两位及以上标注员认为违规，则判定为有害
            human_votes = [case.get("human_0", 0), case.get("human_1", 0), case.get("human_2", 0)]
            is_harmful_r = sum(human_votes) > 1
            
            # 识别实验类型
            is_adversarial = case.get("experiment") != "default"

            item = DataItem(
                query=case.get("test_case"),
                response=case.get("generation"),
                query_harmful_label=True,  # 验证集 Prompt 均为攻击性测试，视为有害意图
                response_harmful_label=is_harmful_r,
                category="harmful",
                extra={
                    "behavior_id": case.get("_behavior_id_key"),
                    "method": case.get("method"),
                    "model": case.get("model"),
                    "prompt_type": "adversarial" if is_adversarial else "vanilla",
                    "human_votes": human_votes,
                    "source_file": "response_json"
                }
            )
            item.id = self.generate_id(i)
            data_items.append(item)
        return data_items

    def load(self, query_task_path: str = None, response_task_path: str = None, **load_kwargs) -> List[DataItem]:
        """
        加载并合并 HarmBench 的两个任务文件数据。
        
        DataItem 映射逻辑 (CSV 部分):
        - id: f"harmbench-{idx:06d}"
        - query: row['Behavior']
        - response: None
        - query_harmful_label: True
        - response_harmful_label: None
        - category: row['SemanticCategory']
        
        DataItem 映射逻辑 (JSON 部分):
        - id: f"harmbench-{idx:06d}" (顺延 CSV 计数)
        - query: row['test_case']
        - response: row['generation']
        - query_harmful_label: True
        - response_harmful_label: True if sum(human_0,1,2) > 1
        - category: "harmful"
        """
        # 优先级：load() 方法参数 > __init__() 传入的路径
        q_path = query_task_path or self.init_kwargs.get("query_task_path")
        r_path = response_task_path or self.init_kwargs.get("response_task_path")
        
        combined_data = []
        
        # 1. 解析并加载 CSV 数据 (从起始索引 1 开始)
        if q_path:
            combined_data.extend(self._parse_query_csv(q_path, start_idx=1))
        
        # 2. 解析并加载 JSON 数据 (ID 计数在 CSV 之后顺延)
        if r_path:
            current_count = len(combined_data)
            combined_data.extend(self._parse_response_json(r_path, start_idx=current_count + 1))
            
        # 若没有任何路径传入，回退至基类的 datasets.load_dataset 逻辑
        if not combined_data:
            return super().load(**load_kwargs)
            
        return combined_data

    def mapping(self, row: Dict[str, Any]) -> DataItem:
        # HarmBench 涉及两个异构文件的合并解析，已在辅助解析方法中直接完成映射
        pass
