"""售后规则引擎（确定性代码）。

设计原则：金额、时效、类目、风控门禁全部由代码判定，模型只负责理解诉求、
选择工具和组织语言。写操作（建单）会重新跑一遍规则，不信任模型传入的金额。
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from ..config import parse_dt
from .ids import next_id
from .registry import Tool, ToolContext

NO_REASON_CATEGORIES = {"生鲜", "虚拟商品", "定制商品", "食品"}
OPENED_EXCLUDE_CATEGORIES = {"美妆", "个护", "内衣", "母婴", "食品"}
ACTIVATION_SENSITIVE = {"手机", "电脑", "数码", "相机"}
OPEN_AFTER_SALE_STATUSES = {"pending_human", "processing", "auto_approved"}

REQUEST_TYPES = ["退货退款", "仅退款", "换货", "维修", "价保", "发票", "物流", "投诉"]


def _add(res: dict[str, Any], code: str, passed: bool, detail: str) -> None:
    res["rules"].append({"code": code, "passed": passed, "detail": detail})


def _days(now: datetime, then: datetime) -> int:
    return (now.date() - then.date()).days


def _hours(now: datetime, then: datetime) -> float:
    return max(0.0, (now - then).total_seconds() / 3600)


def _money(value: float) -> float:
    return round(float(value or 0), 2)


def canonical_request_type(value: str) -> str:
    value = (value or "").strip()
    if value in REQUEST_TYPES:
        return value
    if "仅退款" in value or "只退款" in value:
        return "仅退款"
    if "换" in value:
        return "换货"
    if "维修" in value or "保修" in value:
        return "维修"
    if "价保" in value or "保价" in value or "价格" in value:
        return "价保"
    if "发票" in value:
        return "发票"
    if "物流" in value or "快递" in value:
        return "物流"
    if "投诉" in value:
        return "投诉"
    return "退货退款"


def canonical_reason(value: str) -> str:
    value = (value or "").strip()
    if any(k in value for k in ("质量", "坏", "破损", "故障", "开胶", "划痕", "瑕疵", "变质", "死", "少件", "漏发", "过敏")):
        return "质量问题"
    if any(k in value for k in ("不符", "货不对板", "假货", "描述")):
        return "描述不符"
    if any(k in value for k in ("不想要", "不喜欢", "不合适", "无理由", "七天", "7天")):
        return "七天无理由"
    if any(k in value for k in ("没收到", "未收到", "没拿到")):
        return "未收到货"
    if any(k in value for k in ("降价", "便宜", "价保", "保价")):
        return "价格保护"
    if any(k in value for k in ("延迟", "慢", "停滞", "不更新")):
        return "物流延迟"
    if any(k in value for k in ("投诉", "12315", "曝光")):
        return "投诉"
    if "发票" in value:
        return "发票"
    return "其他"


def _base(order_id: str, request_type: str, reason: str) -> dict[str, Any]:
    return {
        "order_id": order_id,
        "request_type": request_type,
        "reason": reason,
        "eligible": False,
        "decision": "暂不可受理",
        "max_refund": 0.0,
        "requires_human": False,
        "auto_approve": False,
        "rules": [],
        "evidence_required": [],
        "policy_ids": [],
        "next_step": "",
        "suggestions": [],
    }


def _finalize(res: dict[str, Any], user: dict[str, Any], auto_limit: float) -> dict[str, Any]:
    amount = _money(res["max_refund"])
    res["max_refund"] = amount

    if res["eligible"] and amount > auto_limit:
        res["requires_human"] = True
        _add(
            res,
            "R-AMOUNT-LIMIT",
            False,
            f"退款金额 ¥{amount:.0f} 超过自动审批额度 ¥{auto_limit:.0f}，需人工审核",
        )
    if res["eligible"] and (user or {}).get("risk_level") == "high":
        res["requires_human"] = True
        _add(res, "R-RISK", False, "该笔申请需要人工进一步核实")

    if res["eligible"]:
        res["decision"] = "可以受理，需人工审核" if res["requires_human"] else "可以受理"
        res["auto_approve"] = not res["requires_human"]
    elif res["requires_human"]:
        res["decision"] = "需人工核实"
    if not res["next_step"]:
        res["next_step"] = "可提交售后申请，等待审核" if res["eligible"] else "可转人工进一步核实"
    if not res["suggestions"]:
        res["suggestions"] = ["转人工进一步核实"]
    return res


def evaluate_eligibility(
    order: dict[str, Any],
    user: dict[str, Any],
    request_type: str,
    reason: str,
    now: datetime,
    auto_limit: float = 500.0,
) -> dict[str, Any]:
    rt = canonical_request_type(request_type)
    rs = canonical_reason(reason)
    res = _base(order["order_id"], rt, rs)
    status = order.get("status")
    items = order.get("items") or []
    categories = [item.get("category", "") for item in items]
    condition = order.get("condition") or {}
    paid = _money(order.get("paid_amount", 0.0))

    # ---------------------------------------------------------- 订单状态门禁
    if status == "已取消":
        _add(res, "R-ORDER-CANCELLED", False, "订单已取消，售后流程已关闭")
        res["suggestions"] = ["转人工核实退款是否已到账"]
        res["policy_ids"] = ["P-REF-001"]
        return _finalize(res, user, auto_limit)

    existing = order.get("after_sale")
    if (
        existing
        and existing.get("status") in OPEN_AFTER_SALE_STATUSES
        and existing.get("type") == rt
    ):
        _add(res, "R-EXISTING-TICKET", False, "该订单已有同类型售后单正在处理中")
        res["suggestions"] = [f"可在订单详情查看工单 {existing.get('ticket_id')} 的处理进度"]
        res["policy_ids"] = ["P-REF-003"]
        return _finalize(res, user, auto_limit)

    # -------------------------------------------------------------- 投诉
    if rt == "投诉":
        _add(res, "R-COMPLAINT", True, "投诉类诉求需要人工客服跟进")
        res["eligible"] = True
        res["requires_human"] = True
        res["policy_ids"] = ["P-COM-001"]
        res["next_step"] = "已转接人工客服，优先处理"
        return _finalize(res, user, auto_limit)

    # -------------------------------------------------------------- 发票
    if rt == "发票":
        invoice = order.get("invoice") or {}
        if invoice.get("status") == "未开":
            _add(res, "R-INVOICE", True, "订单未开票，可申请开具")
            res["eligible"] = True
            res["evidence_required"] = ["发票抬头（个人姓名，或单位名称 + 纳税人识别号）"]
            res["next_step"] = "可提交开票申请，电子发票 24 小时内发送至预留邮箱"
        else:
            _add(res, "R-INVOICE-ISSUED", False, "发票已开具，换开或冲红需要人工处理")
            res["requires_human"] = True
            res["suggestions"] = ["转人工办理发票换开或冲红"]
        res["policy_ids"] = ["P-INV-001"]
        return _finalize(res, user, auto_limit)

    # -------------------------------------------------------------- 价保
    if rt == "价保":
        delivered = parse_dt(order.get("delivered_at"))
        pp_days = int(order.get("price_protection_days", 0) or 0)
        current = order.get("current_price")
        if delivered is None:
            _add(res, "R-PP-DELIVERED", False, "订单尚未签收，无法申请价保")
        else:
            days = _days(now, delivered)
            within = pp_days > 0 and 0 <= days <= pp_days
            _add(res, "R-PP-WINDOW", within, f"签收后第 {days} 天，价保期 {pp_days} 天")
            diff = _money(paid - (current if current is not None else paid))
            has_drop = diff > 0
            _add(
                res,
                "R-PP-DROP",
                has_drop,
                f"实付 ¥{paid:.0f}，当前售价 ¥{float(current or paid):.0f}，差价 ¥{diff:.0f}"
                if has_drop
                else "当前售价未低于实付金额",
            )
            if within and has_drop:
                res["eligible"] = True
                res["max_refund"] = diff
                res["next_step"] = "可提交价保申请，审核通过后差价原路退回"
        res["policy_ids"] = ["P-PRI-001"]
        if not res["eligible"]:
            res["suggestions"] = ["核对是否在价保期内，或转人工核实活动规则"]
        return _finalize(res, user, auto_limit)

    # -------------------------------------------------------------- 物流
    if rt == "物流":
        if status == "待发货":
            paid_at = parse_dt(order.get("paid_at") or order.get("created_at"))
            hours = _hours(now, paid_at) if paid_at else 0
            overdue = hours >= 48
            _add(res, "R-LOG-SHIP", overdue, f"付款后 {hours:.0f} 小时，平台承诺 48 小时内发货")
            if overdue:
                res["eligible"] = True
                res["next_step"] = "可提交催发货工单；若仍未发货，可申请仅退款"
            else:
                res["suggestions"] = ["物流尚未超时，可继续等待或申请仅退款"]
            res["policy_ids"] = ["P-LOG-001", "P-REF-001"]
        elif status == "运输中":
            events = order.get("logistics_events") or []
            last_time = parse_dt(events[-1]["time"]) if events else parse_dt(order.get("shipped_at"))
            stale = _hours(now, last_time) if last_time else 0
            shipped = parse_dt(order.get("shipped_at"))
            days_since_ship = _days(now, shipped) if shipped else 0
            lost = days_since_ship >= 7
            _add(res, "R-LOG-STALE", stale >= 48, f"物流轨迹已停滞 {stale:.0f} 小时（异常阈值 48 小时）")
            _add(res, "R-LOG-LOST", lost, f"发货后 {days_since_ship} 天仍无签收记录")
            if lost:
                res["eligible"] = True
                res["max_refund"] = paid
                res["next_step"] = "可选择补发或全额退款，无需您联系承运商"
                res["evidence_required"] = ["承运商丢件确认（客服代为核实）"]
            elif stale >= 48:
                res["eligible"] = True
                res["max_refund"] = 10.0
                res["next_step"] = "可发放 10 元无门槛延迟补偿券，24 小时内到账"
            else:
                res["suggestions"] = ["物流仍在正常时效内，建议继续关注"]
            res["policy_ids"] = ["P-LOG-001"] + (["P-REF-001"] if lost else [])
        elif status == "已签收":
            _add(res, "R-LOG-SIGNED", False, "系统显示已签收，不能按物流异常直接处理")
            res["requires_human"] = True
            res["suggestions"] = ["转人工核实签收凭证，确认是否为虚假签收"]
            res["policy_ids"] = ["P-LOG-001"]
        else:
            _add(res, "R-LOG-STATE", False, f"当前订单状态为「{status}」，无法按物流异常处理")
            res["suggestions"] = ["转人工核实订单状态"]
            res["policy_ids"] = ["P-LOG-001"]
        return _finalize(res, user, auto_limit)

    # ------------------------------------------------------- 未发货 / 运输中
    if status == "待发货":
        if rt == "仅退款":
            _add(res, "R-REF-UNSHIPPED", True, "订单尚未发货，可直接申请仅退款")
            res["eligible"] = True
            res["max_refund"] = paid
            res["next_step"] = "可立即提交仅退款，系统 2 小时内处理"
            res["policy_ids"] = ["P-REF-001"]
        else:
            _add(res, "R-NOT-SHIPPED", False, "订单尚未发货，不需要退货")
            res["suggestions"] = ["可直接申请仅退款"]
            res["policy_ids"] = ["P-REF-001"]
        return _finalize(res, user, auto_limit)

    if status == "运输中":
        if rt == "仅退款":
            _add(res, "R-REF-INTRANSIT", True, "订单运输中，可申请仅退款或拒收")
            res["eligible"] = True
            res["max_refund"] = paid
            res["next_step"] = "可申请仅退款，商家会尝试拦截包裹"
            res["policy_ids"] = ["P-REF-001"]
        else:
            _add(res, "R-INTRANSIT-NEED-REFUND", False, "运输中的订单不支持直接退货退款")
            res["suggestions"] = ["可申请仅退款，或拒收后再申请退货"]
            res["policy_ids"] = ["P-REF-001"]
        return _finalize(res, user, auto_limit)

    if status != "已签收":
        _add(res, "R-ORDER-STATE", False, f"当前订单状态为「{status}」，暂不支持该售后类型")
        res["suggestions"] = ["转人工核实订单状态"]
        return _finalize(res, user, auto_limit)

    delivered = parse_dt(order.get("delivered_at"))
    if delivered is None:
        _add(res, "R-DELIVERED-TIME", False, "订单缺少签收时间，无法计算售后时效")
        res["requires_human"] = True
        res["suggestions"] = ["转人工核实签收时间"]
        return _finalize(res, user, auto_limit)

    days = _days(now, delivered)

    # -------------------------------------------------------------- 维修
    if rt == "维修":
        warranty_months = int(order.get("warranty_months", 0) or 0)
        warranty_days = warranty_months * 30.44
        within = warranty_months > 0 and days <= warranty_days
        _add(
            res,
            "R-WARRANTY",
            within,
            f"签收后第 {days} 天，保修期 {warranty_months} 个月"
            if warranty_months
            else "该商品无保修期",
        )
        if within:
            res["eligible"] = True
            res["evidence_required"] = ["故障描述", "订单号或购买凭证"]
            if any(cat in ACTIVATION_SENSITIVE for cat in categories):
                res["evidence_required"].append("品牌授权网点检测报告")
                res["requires_human"] = True
            res["next_step"] = "可预约免费维修，维修周期超过 15 个工作日可要求换新"
        else:
            res["suggestions"] = ["已超出保修期，可转人工咨询付费维修"]
        res["policy_ids"] = ["P-REF-002"]
        return _finalize(res, user, auto_limit)

    # ------------------------------------------------------- 七天无理由
    if rs == "七天无理由":
        within = days <= 7
        _add(res, "R-RET-7D", within, f"签收后第 {days} 天，{'在' if within else '已超过'} 7 天无理由期")

        blocked: list[str] = []
        for cat in categories:
            if cat in NO_REASON_CATEGORIES:
                blocked.append(f"{cat}属于不适用七天无理由的类目")
            elif cat in OPENED_EXCLUDE_CATEGORIES and (
                condition.get("opened") or condition.get("sealed_broken")
            ):
                blocked.append(f"{cat}已拆封，不支持无理由退货")
            elif cat in ACTIVATION_SENSITIVE and condition.get("activated"):
                blocked.append(f"{cat}已激活，不支持无理由退货")
        if condition.get("used") and not blocked:
            blocked.append("商品已使用，影响二次销售")
        _add(
            res,
            "R-RET-CONDITION",
            not blocked,
            "；".join(blocked) if blocked else "商品类目与状态符合无理由退货要求",
        )

        if rt == "仅退款":
            _add(res, "R-REF-NEED-RETURN", False, "已签收商品的无理由退货需退回商品，不能仅退款")
            res["suggestions"] = ["改为申请退货退款，退回运费按规则承担"]
        elif within and not blocked:
            res["eligible"] = True
            res["max_refund"] = paid if rt == "退货退款" else 0.0
            if rt == "退货退款":
                insurance_note = (
                    "，订单含运费险，退回运费可按规则理赔"
                    if order.get("shipping_insurance")
                    else "，退回运费需由您承担"
                )
                res["next_step"] = "可提交退货退款申请，需在 7 天内寄回商品" + insurance_note
            else:
                res["next_step"] = "可提交换货申请，换货优先于退货"
        else:
            res["suggestions"] = (
                ["若商品存在质量问题，可按质量问题申请退换", "或转人工说明特殊情况"]
                if blocked or not within
                else ["转人工进一步核实"]
            )

        res["policy_ids"] = ["P-RET-001", "P-RET-002"]
        if order.get("shipping_insurance"):
            res["policy_ids"].append("P-LOG-002")
        return _finalize(res, user, auto_limit)

    # ----------------------------------------------- 质量问题 / 描述不符
    if rs in ("质量问题", "描述不符"):
        is_fresh = "生鲜" in categories
        if is_fresh:
            hours = _hours(now, delivered)
            within = hours <= 24
            _add(res, "R-FRESH-24H", within, f"签收后 {hours:.0f} 小时，生鲜需 24 小时内反馈")
        else:
            within = days <= 15
            _add(res, "R-QC-15D", within, f"签收后第 {days} 天，质量问题退换时效 15 天")

        is_3c = any(cat in ACTIVATION_SENSITIVE for cat in categories)
        evidence = ["商品问题照片或视频（不少于 2 张 / 10 秒）"]
        if is_3c:
            evidence.append("品牌授权网点检测报告")
        if is_fresh:
            evidence.append("变质或死亡部位照片")
        res["evidence_required"] = evidence

        if rt == "仅退款" and paid > 200 and not is_fresh:
            _add(res, "R-REF-LIMIT", False, f"已签收订单仅退款上限 200 元，本单 ¥{paid:.0f}")
            res["suggestions"] = ["改为申请退货退款，质量问题往返运费由商家承担"]
        elif within:
            res["eligible"] = True
            res["max_refund"] = paid if rt in ("退货退款", "仅退款") else 0.0
            if is_fresh:
                res["next_step"] = "生鲜质量问题无需寄回商品，可直接申请全额退款"
            else:
                res["next_step"] = (
                    "可提交退货退款申请，往返运费由商家承担"
                    if rt == "退货退款"
                    else "可提交换货申请，往返运费由商家承担"
                    if rt == "换货"
                    else "可提交仅退款申请，无需寄回商品"
                )
        else:
            res["suggestions"] = ["已超过退换时效，保修期内可申请维修", "或转人工说明特殊情况"]

        res["policy_ids"] = ["P-REF-002"]
        if is_fresh:
            res["policy_ids"].append("P-RET-002")
        return _finalize(res, user, auto_limit)

    # ---------------------------------------------------------- 未收到货
    if rs == "未收到货":
        _add(res, "R-LOG-CONFLICT", False, "系统显示已签收，与「未收到货」的描述不一致")
        res["requires_human"] = True
        res["suggestions"] = ["转人工核实签收凭证，确认是否虚假签收"]
        res["policy_ids"] = ["P-LOG-001"]
        return _finalize(res, user, auto_limit)

    _add(res, "R-MANUAL-REVIEW", False, "该诉求需要人工进一步判断")
    res["requires_human"] = True
    res["suggestions"] = ["转人工说明具体情况"]
    res["policy_ids"] = ["P-COM-001"]
    return _finalize(res, user, auto_limit)


# --------------------------------------------------------------------- 工具
def _check_handler(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    order = ctx.orders.get(args.get("order_id"))
    if not order:
        return {
            "ok": False,
            "error": "ORDER_NOT_FOUND",
            "message": f"未查询到订单 {args.get('order_id')}，请核对后重试",
            "data": {"order_id": args.get("order_id")},
        }
    user = ctx.orders.user(order.get("user_id"))
    result = evaluate_eligibility(
        order,
        user,
        args.get("request_type", ""),
        args.get("reason", ""),
        ctx.now,
        auto_limit=float(ctx.meta.get("auto_limit", 500.0)),
    )
    return {
        "ok": True,
        "data": result,
        "policy_ids": result["policy_ids"],
        "authorized_max_refund": result["max_refund"],
    }


def _sla_for(request_type: str, auto: bool) -> tuple[str, int]:
    if not auto:
        return "30 分钟内完成审核", 30
    mapping = {
        "退货退款": ("预计 1-3 个工作日原路退回", 2880),
        "仅退款": ("预计 2 小时内到账", 120),
        "换货": ("48 小时内完成换货审核", 2880),
        "维修": ("2 小时内联系您预约维修", 120),
        "价保": ("差价预计 1-3 个工作日原路退回", 2880),
        "物流": ("补偿券预计 24 小时内到账", 1440),
        "发票": ("电子发票 24 小时内开具", 1440),
        "投诉": ("5 分钟内人工响应", 5),
    }
    return mapping.get(request_type, ("预计 1-3 个工作日处理完成", 2880))


def _create_handler(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    order = ctx.orders.get(args.get("order_id"))
    if not order:
        return {
            "ok": False,
            "error": "ORDER_NOT_FOUND",
            "message": f"未查询到订单 {args.get('order_id')}",
            "data": {"order_id": args.get("order_id")},
        }

    rt = canonical_request_type(args.get("request_type", ""))
    reason = canonical_reason(args.get("reason", ""))
    note = str(args.get("note", ""))[:300]
    user = ctx.orders.user(order.get("user_id"))

    existing = order.get("after_sale")
    if (
        existing
        and existing.get("status") in OPEN_AFTER_SALE_STATUSES
        and existing.get("type") == rt
    ):
        return {
            "ok": True,
            "data": {
                **existing,
                "duplicated": True,
                "message": "该订单已有处理中的同类售后单，本次未重复创建",
            },
            "policy_ids": ["P-REF-003"],
        }

    # 关键：写操作重新执行规则，不信任模型参数
    result = evaluate_eligibility(
        order,
        user,
        rt,
        reason,
        ctx.now,
        auto_limit=float(ctx.meta.get("auto_limit", 500.0)),
    )
    if not result["eligible"]:
        return {
            "ok": False,
            "error": "NOT_ELIGIBLE",
            "message": f"按规则无法创建 {rt} 工单：" + "；".join(
                rule["detail"] for rule in result["rules"] if not rule["passed"]
            ),
            "data": result,
            "policy_ids": result["policy_ids"],
        }

    requires_human = result["requires_human"]
    status = "pending_human" if requires_human else "auto_approved"
    approved = 0.0 if requires_human else result["max_refund"]
    sla, sla_minutes = _sla_for(rt, not requires_human)
    ticket_id = next_id("AS", ctx.now)

    ticket = {
        "ticket_id": ticket_id,
        "order_id": order["order_id"],
        "user_id": order.get("user_id"),
        "type": rt,
        "request_type": rt,
        "reason": reason,
        "status": status,
        "status_label": "待人工审核" if requires_human else "已自动受理",
        "approved_amount": approved,
        "max_refund": result["max_refund"],
        "requires_human": requires_human,
        "created_at": ctx.now.isoformat(),
        "sla": sla,
        "sla_minutes": sla_minutes,
        "note": note,
        "policy_ids": result["policy_ids"],
        "duplicated": False,
    }
    order["after_sale"] = ticket
    order.setdefault("audit", []).append(
        {
            "at": ctx.now.isoformat(),
            "action": "create_after_sale_ticket",
            "ticket_id": ticket_id,
            "status": status,
        }
    )
    return {"ok": True, "data": ticket, "policy_ids": result["policy_ids"]}


CHECK_ELIGIBILITY = Tool(
    name="check_after_sale_eligibility",
    description=(
        "判定订单是否满足某类售后条件，并返回规则判定明细、最多可退金额、是否需要人工审批、"
        "所需凭证和政策编号。涉及退货、退款、换货、维修、价保、发票、物流补偿时，必须先调用本工具，"
        "回复中的金额只能引用本工具返回的 max_refund。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "order_id": {"type": "string", "description": "订单号，例如 TB2026102200101"},
            "request_type": {
                "type": "string",
                "enum": REQUEST_TYPES,
                "description": "售后类型",
            },
            "reason": {
                "type": "string",
                "description": "消费者的具体原因，例如：质量问题、七天无理由、未收到货、降价",
            },
        },
        "required": ["order_id", "request_type", "reason"],
    },
    handler=_check_handler,
)


CREATE_TICKET = Tool(
    name="create_after_sale_ticket",
    description=(
        "创建售后工单。仅当消费者明确要求提交/申请/办理时调用。写操作会重新校验规则，"
        "不满足条件会返回 NOT_ELIGIBLE；金额超过 500 元或风控命中时自动转为待人工审核。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "order_id": {"type": "string", "description": "订单号"},
            "request_type": {"type": "string", "enum": REQUEST_TYPES, "description": "售后类型"},
            "reason": {"type": "string", "description": "消费者的原因"},
            "note": {"type": "string", "description": "需要记录给人工客服的补充说明，可为空"},
        },
        "required": ["order_id", "request_type", "reason"],
    },
    handler=_create_handler,
)
