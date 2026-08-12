"""T-G01-2 条件生成 Prompt 拼装（§3.1 / §3.2）."""

from __future__ import annotations

PROPAGANDA_PROMPTS: dict[str, str] = {
    "诉诸权威": "援引专家、机构或权威人士的言论或判断来支撑你的观点，使读者因信任权威来源而接受你的结论，而非依赖独立的事实推理。",
    "诉诸恐惧": "描绘威胁、灾难或严重损失，使读者感到若不采纳你的观点将面临危险，从而被恐惧驱动接受你的判断。",
    "诉诸质疑": "质疑对方动机、诚意或说法的可信度，使读者对其论断产生怀疑，而非正面反驳其论据本身。",
    "诉诸潮流": "强调越来越多人正在这样做或持相同看法，暗示不跟进将落伍、孤立或脱离主流。",
    "贴标签": "用简短有力的称谓概括并定性讨论对象，使读者在深入了解之前便对其形成鲜明（通常是负面）的整体印象。",
    "非黑即白": "将复杂问题简化为只有两个对立选项，迫使读者在其中做出非此即彼的选择，排除中间立场与其他可能性。",
    "预设立场": "在叙述或提问中隐含尚未证实的假设，使读者在不知不觉中默认该前提成立，再在此基础上接受你的结论。",
    "喊口号": "用简短、有力、易记的口号式语句传递核心态度，以情绪冲击和节奏感替代细致论证。",
    "挥舞旗帜": "诉诸集体认同、国家荣誉或共同价值，激发读者的归属感与使命感，使观点与身份认同绑定。",
    "加载语言": "选用带有强烈褒贬色彩的情感词汇描述讨论对象，引导读者产生相应的好恶情绪，而非中性陈述。",
    "光辉普照": "用美好、正面但空泛的褒义词描绘目标，回避具体事实与论证细节，使读者因好感而接受观点。",
    "夸张": "放大风险、收益或后果的程度，使其比实际情况更为极端醒目，以强化说服力。",
    "过度简化": "将复杂因果归结为单一原因或单一解决方案，忽略其他重要因素，使问题看起来更简单、答案更明确。",
    "断章取义": "只引用对你有利的部分事实或数据，略去不利背景与限定条件，使证据显得比实际情况更有支持力。",
    "红鲱鱼": "引入与核心议题相关但会分散注意力的旁支信息，将讨论焦点从关键问题移开。",
    "重复": "用不同措辞反复陈述同一核心观点或结论，加深读者印象，使该信息显得更为确定和普遍。",
}

LENGTH_HINTS: dict[str, str] = {
    "short": "≤50 字",
    "medium": "51–140 字",
    "long": "≥141 字",
}


def build_generation_prompt(guidance: dict) -> str:
    method = guidance["propaganda_method"]
    if method not in PROPAGANDA_PROMPTS:
        raise KeyError(f"未知宣传手段: {method!r}，可选: {list(PROPAGANDA_PROMPTS)}")

    length_key = guidance["length_limit"]
    if length_key not in LENGTH_HINTS:
        raise KeyError(f"未知字数档位: {length_key!r}，可选: {list(LENGTH_HINTS)}")

    stance = guidance["stance"]
    rhetoric = PROPAGANDA_PROMPTS[method]
    length_hint = LENGTH_HINTS[length_key]

    return f"""围绕以下议题写一段中文评论：

【议题】{guidance['issue']}

【修辞要求】{rhetoric}

【立场】{stance['label']}「{stance['target']}」
【字数】{length_hint}

只输出正文，不要解释。"""
