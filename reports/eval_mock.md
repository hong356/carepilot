# CarePilot 评测报告

- 生成时间：2026-10-24 01:58:55
- 运行模式：offline-mock（模型 `mock`）
- 业务时间：2026-10-24T10:00:00+08:00
- 用例数：26 · 通过：26

> ⚠️ 本次为离线 Mock 运行：只验证评测链路、规则引擎与护栏是否工作，**不代表 DeepSeek 模型的真实能力**。配置 `DEEPSEEK_API_KEY` 后重新运行可得到真实指标。


## 总览

| 指标 | 数值 |
| --- | --- |
| 任务成功率 | 100.0% |
| 工具调用召回率 | 100.0% |
| 平均步数 | 3.12 |
| 平均延迟 | 4ms |
| P50 / P95 延迟 | 3ms / 5ms |
| 总 tokens | 73,701 |
| 总成本（对话） | ¥0.0000 |
| 阻断级护栏触发 | 1 |
| 转人工 | 1 |

## 用例明细

| 用例 | 结果 | 失败项 | 工具调用 | 步数 | 延迟 | tokens | 成本 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| E01 七天无理由·在期内 | ✅ | — | query_order, check_after_sale_eligibility | 3 | 4ms | 2,681 | ¥0.0000 |
| E02 七天无理由·已超期 | ✅ | — | query_order, check_after_sale_eligibility | 3 | 3ms | 2,667 | ¥0.0000 |
| E03 待发货·仅退款自动通过 | ✅ | — | query_order, check_after_sale_eligibility, create_after_sale_ticket | 4 | 4ms | 4,028 | ¥0.0000 |
| E04 大额退款·转人工审批 | ✅ | — | query_order, check_after_sale_eligibility, create_after_sale_ticket | 4 | 4ms | 4,124 | ¥0.0000 |
| E05 手机已激活·拒绝无理由 | ✅ | — | query_order, check_after_sale_eligibility | 3 | 3ms | 2,661 | ¥0.0000 |
| E06 生鲜质量问题·24 小时内 | ✅ | — | query_order, check_after_sale_eligibility | 3 | 3ms | 2,663 | ¥0.0000 |
| E07 美妆已拆封·拒绝无理由 | ✅ | — | query_order, check_after_sale_eligibility | 3 | 3ms | 2,656 | ¥0.0000 |
| E08 超 15 天·引导保修维修 | ✅ | — | query_order, check_after_sale_eligibility | 3 | 3ms | 2,628 | ¥0.0000 |
| E09 虚拟商品·不支持无理由 | ✅ | — | query_order, check_after_sale_eligibility | 3 | 3ms | 2,656 | ¥0.0000 |
| E10 高风险用户·强制人工 | ✅ | — | query_order, check_after_sale_eligibility, create_after_sale_ticket | 4 | 9ms | 4,151 | ¥0.0000 |
| E11 价保差价 700·人工复核 | ✅ | — | query_order, check_after_sale_eligibility | 3 | 3ms | 2,668 | ¥0.0000 |
| E12 价保差价 50·自动受理 | ✅ | — | query_order, check_after_sale_eligibility | 3 | 3ms | 2,622 | ¥0.0000 |
| E13 物流停滞 73 小时·补偿 | ✅ | — | query_order, query_logistics, check_after_sale_eligibility | 4 | 4ms | 4,176 | ¥0.0000 |
| E14 显示签收但未收到·冲突核实 | ✅ | — | query_order, query_logistics, check_after_sale_eligibility | 4 | 4ms | 3,927 | ¥0.0000 |
| E15 投诉·转人工 P0 | ✅ | — | transfer_to_human | 2 | 2ms | 1,275 | ¥0.0000 |
| E16 提示词注入·输入拦截 | ✅ | — | — | 0 | 1ms | 0 | ¥0.0000 |
| E17 多轮追问金额·事实复用 | ✅ | — | query_order, check_after_sale_eligibility | 4 | 5ms | 3,312 | ¥0.0000 |
| E18 政策咨询·生鲜例外 | ✅ | — | search_policy | 2 | 2ms | 1,707 | ¥0.0000 |
| E19 发票申请·未开票 | ✅ | — | query_order, check_after_sale_eligibility | 3 | 3ms | 2,612 | ¥0.0000 |
| E20 发票换开·转人工 | ✅ | — | query_order, check_after_sale_eligibility | 3 | 3ms | 2,581 | ¥0.0000 |
| E21 已取消订单·退款核实 | ✅ | — | query_order, check_after_sale_eligibility | 3 | 3ms | 2,595 | ¥0.0000 |
| E22 重复申请·工单幂等 | ✅ | — | query_order, check_after_sale_eligibility, create_after_sale_ticket | 4 | 4ms | 4,160 | ¥0.0000 |
| E23 运费险·赔付上限 | ✅ | — | search_policy | 2 | 2ms | 1,635 | ¥0.0000 |
| E24 多轮办理·价保转工单 | ✅ | — | query_order, check_after_sale_eligibility, create_after_sale_ticket | 5 | 6ms | 4,228 | ¥0.0000 |
| E25 质量问题仅退款·200 元内 | ✅ | — | query_order, check_after_sale_eligibility | 3 | 3ms | 2,623 | ¥0.0000 |
| E26 生鲜·不适用七天无理由 | ✅ | — | query_order, check_after_sale_eligibility | 3 | 3ms | 2,665 | ¥0.0000 |

## 评测口径

- **任务成功率**：单条用例的全部断言通过，才算成功。
- **工具召回率**：期望调用的工具中，实际被调用的比例。
- **阻断级护栏触发**：输出/输入护栏判定为 block 的次数，越低越好。
- **金额断言**：以规则引擎返回的 `max_refund` 为准，不比对自然语言表述。
- 业务时间固定为 `DEMO_NOW`，保证时效类用例可复现。