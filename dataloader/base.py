from typing import Dict, Any, Optional, List, Type
from abc import ABC, abstractmethod
from datasets import load_dataset
import uuid
from dataclasses import dataclass, field

CONST_HF_DATASET_IDS = {
    # Prompt Classification 数据集
    "toxicchat": "lmsys/toxic-chat",
    "openai-moderation": "mmathys/openai-moderation-api-evaluation",
    "aegis-v1": "nvidia/Aegis-AI-Content-Safety-Dataset-1.0",
    "aegis-v2": "nvidia/Aegis-AI-Content-Safety-Dataset-2.0",
    "simple-safety-tests": "Bertievidgen/SimpleSafetyTests",
    "harmbench": "walledai/HarmBench",
    "wildguard": "allenai/wildguardmix",
    "s-eval": "IS2Lab/S-Eval",

    # Response Classification 数据集
    "saferlhf": "PKU-Alignment/PKU-SafeRLHF",
    "beavertails": "PKU-Alignment/BeaverTails",
    "xstest": "walledai/XSTest",
    "xstest-response": "allenai/xstest-response",
    "bingoguard": "fanyin3639/final_safety_bingo_w_severity_v3_decontamination",

    # Response Refusal 数据集
    "or-bench": "bench-llms/or-bench",
    "sorrybench": "sorry-bench/sorry-bench-202503",
    "sorrybench-human-judgement": "sorry-bench/sorry-bench-human-judgment-202503",

    # Reasoning Trace 数据集
    "think": "Qwen/Qwen3GuardTest",

    # Jailbreak 数据集
    "jailbreakbench": "JailbreakBench/JBB-Behaviors",
    "wildjailbreak": "allenai/wildjailbreak",
    "strongreject": "walledai/StrongREJECT",
}


# --- 基础协议 ---
@dataclass
class DataItem:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    query: str = None
    response: Optional[str] = None
    query_harmful_label: Optional[bool] = None
    response_harmful_label: Optional[bool] = None
    response_refusal_label: Optional[bool] = None
    category: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)



# --- 注册器与基类 ---
class DataLoaderRegistry:
    _registry: Dict[str, Type['BaseDataLoader']] = {}
    
    # 集中管理所有路径
    HF_DATASET_IDS = CONST_HF_DATASET_IDS

    @classmethod
    def register(cls, name: str):
        def wrapper(wrapped_class: Type['BaseDataLoader']):
            # 1. 注册类
            cls._registry[name] = wrapped_class
            # 2. 记录原始注册名
            wrapped_class.REGISTER_NAME = name
            # 3. 注入默认路径：如果字典里有，就注入给类
            if name in cls.HF_DATASET_IDS:
                wrapped_class.DEFAULT_HF_PATH = cls.HF_DATASET_IDS[name]
            return wrapped_class
        return wrapper

    @classmethod
    def get_loader(cls, name: str, path: Optional[str] = None, split: str = "train", **kwargs) -> 'BaseDataLoader':
        if name not in cls._registry:
            raise ValueError(f"未找到已注册的 Loader 类: {name}")
        # 将参数解包传给构造函数
        return cls._registry[name](path=path, split=split, **kwargs)




class BaseDataLoader(ABC):
    DEFAULT_HF_PATH: str = None  
    REGISTER_NAME: str = "default"

    def __init__(self, path: Optional[str] = None, split: str = "train", **kwargs):

        # 优先级：显式传入的 path > 类绑定的默认路径
        self.path = path or self.DEFAULT_HF_PATH
        
        # 将 split 存入 init_kwargs，方便 load 方法统一处理
        self.init_kwargs = {"split": split, **kwargs}
        
        if not self.path:
            raise ValueError(f"Loader {self.__class__.__name__} 未绑定路径，且未传入路径")

    @abstractmethod
    def mapping(self, row: Dict[str, Any]) -> DataItem:
        pass

    def generate_id(self, idx: int) -> str:
        return f"{self.REGISTER_NAME}-{idx:06d}"

    def load(self, split: str = "train", **load_kwargs) -> List[DataItem]:
        from tqdm import tqdm
        # 1. 合并参数
        final_kwargs = {**self.init_kwargs, "split": split, **load_kwargs}
        current_split = final_kwargs.get("split", "train")
        
        # 2. 加载原始数据集
        dataset = load_dataset(self.path, **final_kwargs)

        # 3. 提取 split
        if hasattr(dataset, "keys") and current_split in dataset:
            dataset = dataset[current_split]

        results = []
        
        # 4. 在循环中加入 tqdm 进度条
        # desc 设置为注册名，这样多个数据集加载时能分清楚
        for i, row in enumerate(tqdm(dataset, desc=f"Loading {self.REGISTER_NAME}.{current_split}"), 1):
            item = self.mapping(row)
            item.id = self.generate_id(i)
            results.append(item)
            
        return results
