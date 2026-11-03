# DiscoveryRSI 对 CMR/LGE 控制面的参考评估

核查日期：2026-10-27。本文区分论文事实、公开代码观察、项目适配建议；不表示这些建议已上线。

## 资料与复现身份

- [论文v1：DiscoveryRSI: Recursively Learning to Discover Valuable Hypotheses](https://arxiv.org/html/2610.23839v1)。截图对应arXiv:2610.23839。
- [作者项目](https://github.com/Gen-Verse/DiscoveryRSI)，本次只读检出提交`7389e0b58ae50486ab35aa3dcbfa2fa15fea1a91`，未安装到服务器或运行第三方代码。
- [模型发布](https://huggingface.co/Gen-Verse/DiscoveryRSI-27B)。模型卡定位为研究假设与计划生成。
- [DFM-Bench](https://github.com/Gen-Verse/DFM-Bench)，把研究进展和执行交付分开报告。

## 应借鉴的机制

论文§3描述三条经验利用路径：贡献归因产生训练监督；Harness Replay诊断并验证工作流改动；Rediscovery把接受的更新用于新问题。研究对象保留版本、父子关系、证据、计划与判据；§4把假设、修订、路径、全轮和贡献分层评估。
[论文§3–4](https://arxiv.org/html/2610.23839v1#S3)。

作者展示的自动假设评分提升和文献关联比例，不能转换为本项目的Mask准确率、自动验收率或待筛清零保证。作者附录D明确自动评审的相关性、证据缺失和交付与研究质量的区别。API调用也不会自动更新服务端模型权重。
[论文附录D](https://arxiv.org/html/2610.23839v1#A4)。

## 公开代码核查：不要把演示harness当完整RSI部署件

以下观察限定于本次检出版本，而非否认论文或模型发布：

1. `core.py`的Candidate保留parents/evidence/critique；ReplayBuffer提供JSONL追加/读取，适合作为可追溯对象的简洁示例。DiscoveryLoop每轮选择本轮最高评分候选；该选择本身没有父版本无退化比较。
2. `harness.py`的候选提示词传入父ID，但未传入该父节点的完整正文；新run构建新DiscoveryLoop，入口没有先读取历史ReplayBuffer恢复经验。单有JSONL和父ID不能证明跨模型继承。
3. 同一harness通过adapter生成和评审；工具在此入口作为describe信息进入提示词。这不是本项目候选NIfTI生成、独立评分、应用审计与故障恢复的替代实现。
4. 此仓库是小型公开agentic core。阅读到的入口未实现论文中的完整归因加权训练、工作流候选受控选择与权重更新。不能把README的整体论文介绍当成每项训练流程已随此入口开源。
5. pyproject要求Python>=3.10，本项目服务器myenv为3.9；直接混装还会带来运行时差异。

源码：[core.py](https://github.com/Gen-Verse/DiscoveryRSI/blob/7389e0b58ae50486ab35aa3dcbfa2fa15fea1a91/src/discovery_rsi/core.py)，[harness.py](https://github.com/Gen-Verse/DiscoveryRSI/blob/7389e0b58ae50486ab35aa3dcbfa2fa15fea1a91/src/discovery_rsi/harness.py)，[pyproject](https://github.com/Gen-Verse/DiscoveryRSI/blob/7389e0b58ae50486ab35aa3dcbfa2fa15fea1a91/pyproject.toml)。

## 本项目适配建议（维护者提出，尚非已验收能力）

| 对应机制 | 控制面应记录或执行 | 验收方式 |
| --- | --- | --- |
| 版本化假设/修订 | 问题簇、父版本、规则/工具变更、实际输入输出合同、有效证据、作用范围 | 对照真实父版本，不能只比自身声明或最后一次分数 |
| 贡献归因 | 把实际收益关联到Owner动作、角色/模型、工具版本及审计回执 | 分别记录执行成功、标签质量增量、额外解决数量；重复发现去重 |
| Harness Replay | 保存失败快照源码/哈希、环境、调用合同、异常和完整测试反馈；先重放故障 | 固定模型和预算比较控制器改动；不能声称未执行的替代路径成功 |
| Rediscovery | 接受的新工具/提示策略进入新任务，失败经验可被后任模型检索 | 用未参与该修复的新样本/病例簇验证；保留冻结测试答案 |
| 分层评估 | 将代码可运行、当前标签改善、后续泛化、全轮成本分开 | 看板不能用PROMOTED、Token消耗或测试数量代替临床/标签进展 |

优先顺序：先把已部署角色→真实候选→独立评分→真实回执跑通；再将成功和失败汇入可回放经验；最后做固定预算的父子版本/新簇评估。保持现有医生配对评分器与冻结基准为验收来源，不用论文中的科研假设评分代替Mask评分。

当前使用冻结API模型，可改进的是工作流、工具、上下文和策略选择。若今后要做论文式参数学习，需要另立训练数据/算力/基准和部署计划；不能把保存skill称为已更新模型权重。

## 与run899实测的关系

本轮发现了输入字段、输出字段、几何维度、测试反馈截断、未晋级源码丢失等控制/交接问题。这些适合作为Harness Replay的故障样本，而不是要求医生补标。

用户已明确授权Qwen接任同一Owner的一次修复与独立评审；模型实际接收上一个失败快照的生成器/测试源码和哈希，且没有新建分析轮次。Qwen已完成Owner验证、完整回归和独立实现评审。生产标签结果必须另查真实应用审计，不能据此预先宣布标签已修复。

本轮同时改变了模型、继承上下文与测试诊断，不能据此因果归因“Qwen一定优于DeepSeek”或“某个控制补丁单独带来全部收益”。后续需要固定输入、版本和预算的对照。

Codex继续只负责框架与验证接线；不依据本文自行选择医学修法、不改领域Owner源码、不绕过独立评分或强制分桶。


本轮最终验证：输出文件字段兼容后，授权服务处理10例，1例实际标签更新并CHECKED，
当前独立评分器65→100、备份与源哈希均匹配；6例无评分提升、3例配对MI退化而拒绝。
这是工具链真正产生过标签进度的证据；不是新样本泛化优势、医生准确率或全自主RSI的证明。
常驻服务按有界授权结束后保持暂停。完整集成回归1837通过（skipped=1）。
