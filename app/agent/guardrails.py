"""输出护栏：把「模型说的话」与「工具返回的事实」做交叉校验。

检查项：
- 金额越权：承诺金额超过规则引擎授权上限
- 虚假动作：声称已退款/已提交/已转接，但没有对应工具成功调用
- 政策幻觉：引用了本轮未检索到的 policy_id
- 过度承诺 / 站外退款 / 索要敏感信息
- 未脱敏的个人信息
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

SAFE_INPUT_REPLY = (
    "抱歉，这个话题我无法处理。我是售后客服助手，可以帮您查询订单、办理退换货、"
    "查询物流和申请发票。请告诉我订单号或具体诉求。"
)

INJECTION_PATTERNS = [
    re.compile(r"(忽略|无视|跳过).{0,8}(以上|之前|前面|系统).{0,6}(指令|规则|提示|设定)"),
    re.compile(r"(输出|打印|告诉我).{0,6}(系统提示|system prompt|你的设定)"),
    re.compile(r"(扮演|现在你是|从现在开始你是).{0,10}(管理员|开发者|黑客|另一个)"),
    re.compile(r"(开发者模式|越狱|DAN模式|ignore (all )?previous)", re.IGNORECASE),
]

PROMISE_RE = re.compile(
    r"为您|给您|退您|赔您|补偿您|可以退|可退|将退|会退|退款金额|赔付金额|补偿金额|返还|到账|最多可退"
)
AMOUNT_RE = re.compile(r"(?:¥|￥)\s*(\d+(?:\.\d+)?)|(\d+(?:\.\d+)?)\s*(?:元|块)")

_SENTENCE_START = r"(?:^|[。！？!?\n；;])\s*"

ACTION_RULES = [
    (
        re.compile(_SENTENCE_START + r"已(?:为您|给您)?(?:提交|发起|创建|登记|安排|开具|补发)"),
        "create_after_sale_ticket",
        "声称已提交/办理",
    ),
    (
        re.compile(r"已(?:为您|给您)(?:提交|发起|创建|登记|安排|开具|补发)"),
        "create_after_sale_ticket",
        "声称已提交/办理",
    ),
    (
        re.compile(_SENTENCE_START + r"已(?:为您|给您|经)?(?:退款|赔付|补偿|退回|退还)"),
        "create_after_sale_ticket",
        "声称已退款/赔付",
    ),
    (
        re.compile(r"已(?:为您|给您)(?:退款|赔付|补偿|退回|退还)"),
        "create_after_sale_ticket",
        "声称已退款/赔付",
    ),
    (
        re.compile(_SENTENCE_START + r"已(?:为您|经)?(?:转接|转人工|接入人工)"),
        "transfer_to_human",
        "声称已转人工",
    ),
    (
        re.compile(r"已(?:为您)?(?:转接|转人工|接入人工)"),
        "transfer_to_human",
        "声称已转人工",
    ),
]

FORBIDDEN_PATTERNS = [
    (re.compile(r"(保证|百分百|100%|绝对|一定)(?:能|可以|会)?(?:退款|退货|赔付|赔偿|通过)"), "过度承诺"),
    (re.compile(r"无条件(?:退款|退货)"), "过度承诺"),
    (re.compile(r"(?:加微信|扫码|私下|站外).{0,8}(?:退款|转账|赔付)"), "引导站外退款"),
    (re.compile(r"(?:验证码|支付密码|银行卡密码)"), "索要敏感信息"),
    (re.compile(r"(?:删除|撤回).{0,4}(?:投诉|差评)"), "不当引导"),
]

POLICY_ID_RE = re.compile(r"P-[A-Z]{3}-\d{3}")
CITATION_BRACKET_RE = re.compile(r"【[^】]*P-[A-Z]{3}-\d{3}[^】]*】|\[[^\]]*P-[A-Z]{3}-\d{3}[^\]]*\]")
MOBILE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
ID_CARD_RE = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")
SENTENCE_SPLIT_RE = re.compile(r"[。！？!?\n；;]")


@dataclass
class Violation:
    code: str
    message: str
    severity: str = "block"  # block | warn

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message, "severity": self.severity}


@dataclass
class GuardContext:
    authorized_max_refund: float | None = None
    retrieved_policy_ids: set[str] = field(default_factory=set)
    successful_tools: set[str] = field(default_factory=set)
    requires_human: bool = False
    tool_calls: int = 0


def check_input(text: str) -> list[Violation]:
    for pattern in INJECTION_PATTERNS:
        if pattern.search(text):
            return [
                Violation(
                    "INPUT_INJECTION",
                    "用户输入包含提示词注入/越权指令，已拦截并按普通售后咨询处理",
                )
            ]
    return []


def _amounts(text: str) -> list[float]:
    values = []
    for match in AMOUNT_RE.finditer(text):
        raw = match.group(1) or match.group(2)
        try:
            values.append(float(raw))
        except ValueError:
            continue
    return values


def check_output(answer: str, ctx: GuardContext) -> list[Violation]:
    violations: list[Violation] = []
    limit = ctx.authorized_max_refund or 0.0

    # 1. 金额越权
    for sentence in SENTENCE_SPLIT_RE.split(answer):
        if not PROMISE_RE.search(sentence):
            continue
        # 引用政策原文中的「最高/上限/不超过」属于信息说明，不是对用户的承诺。
        # 引用编号与金额可能被句号拆到相邻句，这里用整段回复是否带引用来判断。
        if CITATION_BRACKET_RE.search(answer) and re.search(
            r"最高|上限|不超过|标准|额度", sentence
        ):
            continue
        for amount in _amounts(sentence):
            if amount > limit + 0.01:
                violations.append(
                    Violation(
                        "AMOUNT_OVER_LIMIT",
                        f"回复承诺金额 ¥{amount:.0f} 超出本轮授权上限 ¥{limit:.0f}",
                    )
                )

    # 2. 虚假动作（没有对应工具成功调用）
    for pattern, tool, label in ACTION_RULES:
        if pattern.search(answer) and tool not in ctx.successful_tools:
            violations.append(
                Violation("UNSUPPORTED_ACTION", f"{label}，但本轮没有对应的成功工具调用")
            )

    # 3. 需人工审核时，不得声称已退款/已到账
    if ctx.requires_human and re.search(r"已(?:为您|经)?(?:退款|赔付|补偿|到账)", answer):
        violations.append(
            Violation("HUMAN_REVIEW_PROMISE", "工单需人工审核，不得声称已退款或已到账")
        )

    # 4. 政策编号幻觉
    cited = set(POLICY_ID_RE.findall(answer))
    unknown = cited - ctx.retrieved_policy_ids
    if unknown:
        violations.append(
            Violation(
                "UNKNOWN_POLICY",
                "引用了本轮未检索到的政策编号：" + "、".join(sorted(unknown)),
            )
        )
    if not cited and re.search(r"根据.{0,10}(规则|政策|条款)", answer):
        violations.append(
            Violation("MISSING_CITATION", "提到“根据规则/政策”但未给出政策编号", "warn")
        )

    # 5. 绝对化表述 / 站外退款 / 索要敏感信息
    for pattern, label in FORBIDDEN_PATTERNS:
        if pattern.search(answer):
            violations.append(Violation("FORBIDDEN_PHRASE", f"{label}：{pattern.pattern}"))

    # 6. 个人信息未脱敏
    if MOBILE_RE.search(answer):
        violations.append(Violation("PII_UNMASKED", "回复中出现完整手机号，应掩码处理"))
    if ID_CARD_RE.search(answer):
        violations.append(Violation("PII_UNMASKED", "回复中出现完整身份证号，应掩码处理"))

    # 7. 应提示人工审核
    if ctx.requires_human and not re.search(r"人工|审核|核实", answer):
        violations.append(
            Violation("MISSING_HANDOFF_NOTICE", "该单需人工审核，回复未提示", "warn")
        )

    # 去重：同一 code + message 只保留一条
    unique: list[Violation] = []
    seen: set[tuple[str, str]] = set()
    for violation in violations:
        key = (violation.code, violation.message)
        if key in seen:
            continue
        seen.add(key)
        unique.append(violation)
    return unique


def blocking(violations: list[Violation]) -> list[Violation]:
    return [v for v in violations if v.severity == "block"]


def format_violations(violations: list[Violation]) -> str:
    return "\n".join(f"- [{v.code}] {v.message}" for v in violations)


def preview(payload: dict, limit: int = 700) -> str:
    text = json.dumps(payload, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + "…"
