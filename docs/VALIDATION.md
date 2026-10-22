# 第一版验证范围

状态：CPU定向回归与短程原生smoke通过；没有启动长训练，没有在本独立项目重新运行完整1024任务评测。此结果证明已验证范围内的模型/数学等价与接线，不是导航性能或实机安全认证。

| 检查 | 实际证据 |
|---|---|
| 正常CPU import/model | 63tensor；没有加载Isaac；无GRU/LSTM/RNN实例 |
| U153导入 | 原文件SHA严格匹配；完整63tensor、两Adam/VN/GAE恢复；Actor68608、critic78592步 |
| 原snapshot teacher forward | 原Planar类直接加载与原factory函数源码；同64个真实观测，loc/std/normalizedV/rawV最大误差均0 |
| 独立U153配置核对 | 435个共有叶字段递归比较；7处仅身份/phase/ordinal/路径/schema命名差异，无意外共有数值参数变化；不声称删除的旧campaign元数据相同 |
| CPU测试 | 7项通过；配置拒绝、真实scene/requests SHA、3200噪声oracle、目标系/GAE有限终端、提前全终态prefix、数值golden |
| 数值golden | 直接由权威snapshot原函数生成；5个导航/首次到达/到达后deadline/静动碰重叠/首次到达同时deadline场景，奖励和终止向量exact；名义限幅、原top1 CBF、无刷accel2RPYT各字段exact（含显式NaN相等） |
| relocation/noarchive | 复制到独立临时路径，audit hook禁止读取旧archive；正常model/config/resources加载通过，旧archive访问尝试0 |
| 原生smoke | 32×128步采集=4096训练帧、1次完整PPO更新，再32步sampled；共160控制步/环境 |
| 原生健康 | 所有drone state finite、每外层5子步；Actor与critic权重均实际变化；实际LR5e-6/5e-5 |
| reset/save-restore | lane0 reset时其它31lane缓冲/位姿/速度与全局clock不变；保存恢复policy digest相同；短eval无policy/VN变化 |
| 实际物理 | 原生构造/末审执行mass42g、惯量/COM及adapter类型getter断言；USD单体0joint、线/角阻尼0.2与gyro真实检查通过 |
| source保留 | 原939条源码/资源及symlink内容SHA与源HEAD未变；原版本worktree干净 |

原PPO、GAE、structured encoder、smooth std、goal frame、reward、top1 unbounded CBF、backend与转换器的11个数学模块保持与snapshot字节一致。实际生产不导入原campaign contract/父runtime，也不靠stub模块通过CPU import。

smoke前两次初始化失败已保留于外部验证目录：一次是两种模块名重复注册，一次是旧TensorDict debug `.shapes`属性。两次均未进入collector控制rollout/更新。分别统一模块导入名、改用真实shape字典后，smoke_v3完成。没有放宽任务、物理或学习条件。

smoke后只有正式eval的提前全终态oracle校验、actual ticks记录、固定eval seed检查，以及新增CPU黄金fixture；详见 `NATIVE_SMOKE.json` 的精确SHA差异。原生smoke运行的物理/PPO/控制路径未改，不为这个CPU边界重复GPU。

源码/资源manifest位于 `provenance/PRODUCT_MANIFEST.json`，源映射见 `SOURCE_MAP.json`；CPU teacher证据见 `CPU_FORWARD_PARITY.json`，原生摘要见 `NATIVE_SMOKE.json`，隔离验证见 `RELOCATION.json`。详细旧stdout/firstbatch、初始失败和生成脚本保留在本次外部交付artifact，不作为production必需输入。

注意：三个动作seed仅代表执行采样变化，不是独立训练seed。新独立仓库未重测旧U153的628/1024结果，不能将旧结果当作新项目的原生完整迁移认证。
