import uuid
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass
class DataItem:
    """
    Safeguard 框架基础数据交换协议
    """
    # 唯一标识符，默认生成一个随机 UUID，也可手动指定
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # 核心文本内容
    query: str = None
    response: Optional[str] = None
    
    # 标注信息：True-是, False-否, None-未标注
    query_harmful_label: Optional[bool] = None
    response_harmful_label: Optional[bool] = None
    response_refusal_label: Optional[bool] = None
    
    # 分类与元数据
    category: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)
