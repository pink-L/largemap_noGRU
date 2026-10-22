# LargeMap_noGRU

独立的 Crazyflie 无刷无人机大地图导航训练与评测项目。代码源自 OmniDrones 快照 `24c2ce80d26c0b0eb72a14756c3e34f61c94a1f5`，默认模型是 U153 回退基线。这里没有原仓库 Git 历史、外层 drones 子模块、其他任务注册、RNN/GRU 算法或夜间实验调度器。

这是模拟研究实现，不是实机安全证明。完整历史实验与原始结果未随本仓库复制。默认参数的 CPU 等价性和短程原生接线验证见 `docs/VALIDATION.md`；未在新项目重新运行完整1024任务评测或长训练，不声称迁移后的导航性能已认证。

## 环境与运行方式

使用 **Python 3.11、Isaac Sim 5.1** 的现有环境，在仓库根目录运行 `python -m largemap_no_gru ...`。代码和资源以源码目录布局交付；可用于 editable 开发，不提供经过认证的 wheel/非editable安装。不要移动 `largemap_no_gru/` 而丢失同级 `scripts/`、`resources/`、`config/`、`omni_drones/`。

已验证依赖版本：PyTorch 2.7.0+cu128、TorchRL 0.11.1、TensorDict 0.11.0、NumPy 1.26.0、OmegaConf 2.3.1、Hydra 1.3.6、einops 0.8.2。测试使用 pytest 9.1.1；可选在线日志使用 wandb 0.29.0。具体安装版本另见 `requirements-validated.txt`。不要为运行项目盲目升级现有 Isaac/PyTorch 环境。

CPU 模型、checkpoint导入和测试不加载 Isaac。原生仿真需要可用 NVIDIA GPU、兼容驱动/CUDA、Isaac扩展及其许可证；仓库不分发 Isaac/NVIDIA runtime。沿用旧工程的 `omni.isaac` 兼容API，在5.1上有弃用提示，不声称兼容未测试的新版本。

```bash
# 在已具备依赖的环境中，无GPU模型构造及纯CPU回归
CUDA_VISIBLE_DEVICES='' python -m largemap_no_gru model-check
CUDA_VISIBLE_DEVICES='' python -m pytest tests/test_cpu.py -q
```

## 默认配置与数据流

`config/baseline.yaml` 是直接配置，不依赖旧 Hydra 配置树或本机归档目录。默认训练场景 S280D50、32秒、1024环境；训练按每环境四边随机重置，短距分支比例0。完整评测使用随包固定1024个四边全距离任务。两种任务采样不是同一协议。

- Actor观测：state21、static36×4、dynamic4×10；固定episode目标坐标系，真实yaw扫描；不恢复已取消的速度/3与gyro/10归一化。
- 模型：静态CNN128 + 动态MLP64 + state21；独立Actor/critic，三层MLP，`state_smooth_v1`标准差，无GRU/LSTM。
- latent Gaussian样本保留为 `agents.z_policy` 用于PPO概率；`tanh(z)` 映射到目标坐标系名义加速度，逐轴尺度(1.5,1.5,0.5)，XY范数1.5，总范数1.6 m/s²。
- 世界系top1 unbounded CBF → acceleration2RPYT → firmware PID → paper电机模型 → PhysX；保留原fallback和限幅语义，不包含新CBF修复。
- 单刚体42g；惯量(3.3,3.6,5.9)×10⁻⁵ kg·m²；paper电机曲线、50ms电机动态；线/角阻尼均0.2。2ms×5子步，即100Hz Actor/外层控制。
- 到达半径0.5m、首次到达奖励80、hold rate1、velocity rate5、执行平滑rate0.02、CBF修正惩罚系数3.0；原终止/有限时域bootstrap、GAE、ValueNorm与PPO目标保持。
- 默认实际Actor LR=5e-6，critic LR=5e-5；4 epochs/64 minibatches/128步rollout。配置中的遗留 `algo.lr` 不是两个优化器实际学习率，实际值在run和checkpoint记录。

第一版仅验证默认profile。修改场景、时限、奖励、观测或物理需要显式配置与相应验证，不能把未测配置当作已经认证。共享基础类仍保留原模块名，以降低行为和checkpoint键名变化；它们不导入原实验目录。

## 外置U153 checkpoint与一次性导入

U153原文件不随Git发布。由用户在可信存储中取得：

- 文件名：`U_u0153_020054016.pt`，7,837,967 bytes
- SHA256：`76ceb7d6f85c46a8694d7fe638b38f1da3bb1337e3e0d41a6daaebdaaf7df527`
- 来源：旧R76→U153的CBF系数3.0、S280D50/32s、无刷阻尼0.2实验；其历史identity/ledger保留在迁移receipt。它不是V77。

只对可信checkpoint使用 `torch.load`。导入器只接受上述精确SHA，严格加载完整63tensor，检查两Adam按参数名的moments/step、ValueNorm/GAE及实际LR。它不运行旧L/N/R validator，不改旧identity，也不声称复现了原历史训练。

```bash
python -m largemap_no_gru import-checkpoint \
  --source /path/to/U_u0153_020054016.pt \
  --output /path/to/models/u153_portable.pt
python -m largemap_no_gru model-check --checkpoint /path/to/models/u153_portable.pt
```

生成新schema checkpoint及 `.import.json`。默认继承Actor Adam68608步、critic Adam78592步；完整学习状态保留，物理/RNG/全局时钟/episode/cohort重新开始。迁移时不需要旧archive存在；原始checkpoint路径由命令行指定。

checkpoint绑定实际源码/配置/资源allowlist。生产只在启动与结束检查，不每步扫描整个Git树。兼容源码改变后必须明确迁移，不可改存档SHA假冒原版本。当前第一版不提供任意旧checkpoint的宽松加载。

## 训练

必须明确选择从零或resume，缺checkpoint不会静默重新初始化。所有输出使用尚不存在的新目录；学习预算为完整 `num_envs×128` batch，不隐含补尾批。

```bash
# 示例，不由本仓库自动执行
python -m largemap_no_gru train --from-scratch --updates 38 \
  --output /path/to/runs/scratch_5m --wandb-mode off
python -m largemap_no_gru train --resume /path/to/models/u153_portable.pt \
  --updates 38 --save-every 38 --seed 3725 \
  --output /path/to/runs/u153_continue --wandb-mode off
```

38个完整更新=4,980,736新帧。resume恢复新schema的完整policy、两Adam、GAE与ValueNorm；新local ledger延续既有学习计数。默认保留checkpoint实际LR；只有显式 `--actor-lr` / `--critic-lr` 才覆盖并记录。

当前项目不继承历史夜间campaign的性能晋级、分支预算和固定4096观测局部KL停止控制器；它们属于外部实验调度，不是PPO目标。本项目仍硬拒非有限模型、损失和状态、错误子步、资源/source漂移及不匹配checkpoint。默认稳定参数不等于性能不会退化，长训练需用户自行定义评测/停止计划。

结束时未完成episode记录为phase boundary中断，不计作成功或失败。保存、resume均是学习状态恢复，不能称连续物理轨迹恢复。

## 完整评测

```bash
python -m largemap_no_gru eval --checkpoint /path/to/models/u153_portable.pt \
  --suite s280d50-full --action-mode sampled --action-seed 93728 \
  --output /path/to/eval/u153_seed93728 --wandb-mode off
```

1024固定任务、环境seed3725、最多3200控制步。sampled动作使用独立NumPy `SeedSequence([action_seed])` normal流，每tick固定消费1024×1×3（float64生成→float32），不因lane结束改变其他lane噪声。三种已有oracle种子93728/93729/93730可显式选择。原 first-terminal锁定、终态parking、统计重叠与primary reason保留；正式结果要求1024完成且0 censored。`--action-mode mean`可显式比较，但不默认额外跑。

`evaluation.json` 保存原完整请求级指标、静碰/动碰/crash/越界/超时及保持。到达、末态安全和最后5秒低速保持是不同指标。评测不更新policy、ValueNorm或优化器；不把短smoke结果写成导航成功率。

## 最小原生smoke

```bash
python -m largemap_no_gru smoke --resume /path/to/models/u153_portable.pt \
  --envs 32 --steps 128 --updates 1 \
  --output /path/to/smoke/check --wandb-mode off
```

独立diagnostic目的：S280D50实际场景前32个合法任务，128步采集、一次PPO更新，再32步sampled接线（总160控制步/环境）。另检查lane0部分reset不改变其他lane或全局clock、保存后完整学习状态恢复和物理末审。它不能替代完整3200×1024评测，不用于推断安全率。

可用 `NAVVEL_KIT_EXTRA_ARGS='--/rtx/raytracingEnabled=false'` 关闭不需要的raytracing，沿用原headless运行设置。不要把它误当作物理参数变化。

## 日志、退出与文件布局

默认W&B关闭；可显式 `--wandb-mode online --wandb-project ... --run-name ...`。网络故障写本地 `WANDB_PENDING.json`，不为补日志重跑科学任务。原生worker之外的纯Python父进程检查COMPLETION/FAILURE和真实return code，避免Kit在异常shutdown时exit0被误认为成功。

- `largemap_no_gru/`：独立配置、checkpoint、CLI与原生接线。
- `scripts/`：只保留LargeMap实际数学/环境/helper；无历史campaign入口。
- `omni_drones/`：必要模型/学习/物理/控制兼容层，已解除无关eager registry。
- `resources/`：真实train/eval场景、任务和CPU parity观测fixture、动作noise oracle。
- `config/`：默认resolved配置；`provenance/`：源SHA映射与迁移范围。
- `tests/`、`docs/`：定向CPU回归、验收证据和限制。

原始weights、checkpoint、训练输出、W&B、缓存和密钥不入Git。来源与MIT版权见 `LICENSE`；无刷电机/反扭矩来自 *How to Model Your Crazyflie Brushless* 论文/开源参数。保留USD资产来自原OmniDrones工程，具体源文件SHA在provenance；未把NVIDIA安装或上游全部机器人资产打包。
