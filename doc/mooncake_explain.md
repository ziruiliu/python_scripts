# Mooncake 总体架构与 Mooncake Store 角色说明（面向 SGLang 集成）

本文从系统视角总结 Mooncake 的整体架构，并重点说明 Mooncake Store 在其中的位置。最后结合 SGLang，解释什么场景下会**同时**用到 Transfer Engine 和 Mooncake Store。

---

## 1. 一句话理解 Mooncake

Mooncake 可以理解为一个“面向 LLM 推理 KVCache 的分布式缓存基础设施”，包含：

- **高速数据传输能力**（Transfer Engine）
- **分布式 KV 存储与元数据编排能力**（Mooncake Store）

其中，Transfer Engine 主要解决“怎么快传”；Mooncake Store 主要解决“存在哪、怎么管、何时淘汰/下沉”。

---

## 2. Mooncake 总体架构（逻辑分层）

从职责上，可以分为三层：

### 2.1 控制面（Control Plane）

- 负责对象元数据、副本信息、节点状态、空间分配策略等。
- 在 Mooncake Store 中主要对应 `master service` 相关能力（对象到副本/节点的映射管理、生命周期管理等）。

### 2.2 数据面（Data Plane）

- 负责真实 payload 传输。
- 关键能力是零拷贝、RDMA、多网卡聚合等，这部分主要由 **Transfer Engine** 提供。
- 上层（vLLM/SGLang/Store Client）提交传输请求后，真正的数据流通常不经过 master，而是节点间直传。

### 2.3 存储面（Storage Plane）

- 负责 KV 的驻留介质与层级管理（内存、文件、分层缓存后端）。
- 在 Mooncake Store 中体现为对象读写、副本管理、以及 offload 到本地文件后端（如 bucket/file-per-key/offset allocator，及可扩展插件如 HF3FS）。

---

## 3. Mooncake Store 在整体中的位置与角色

Mooncake Store 不是“仅仅一个 KV API 封装”，而是把**分布式缓存管理**与**分层存储落盘**接在一起的核心模块：

1. **北向接口角色**
   - 对上承接推理引擎（如 SGLang/vLLM）的 KV 生命周期请求（Put/Get/BatchQuery/Remove 等）。
2. **南向编排角色**
   - 对下调用 Transfer Engine 完成跨节点数据搬运。
3. **中心协调角色**
   - 通过 master 维护元数据一致性、对象定位与副本状态。
4. **容量治理角色**
   - 通过 FileStorage + StorageBackendInterface 实现 offload（热数据在内存、冷数据下沉到磁盘/文件系统插件）。

换句话说：

- **Transfer Engine 更像“高性能网络 DMA 引擎”**；
- **Mooncake Store 更像“带分布式控制面的 KV 缓存系统 + 分层存储管理器”**。

---

## 4. SGLang 集成时，什么时候只用 Transfer Engine？

当你做的是 **PD/EPD 形态的解耦推理链路**（例如 prefill→decode、encoder→prefill 的跨进程/跨节点数据转发），且关注点是“阶段间高速传输”，通常以 Transfer Engine 作为传输后端：

- 目标：降低 TTFT、减少 CPU 参与、提升跨节点带宽利用。
- 典型行为：传输中间结果（KV、embedding、激活等）而非做完整分层持久化。

这类场景下，Store 能力可能并非主路径，或者仅作为可选组件存在。

---

## 5. SGLang 集成时，什么时候只用 Mooncake Store？

当你主要使用的是 **HiCache 分层缓存能力**，并且重点是 L1/L2/L3 分层与缓存命中管理（而非跨阶段解耦传输），就会显式依赖 Mooncake Store 作为 L3 后端：

- L1：GPU
- L2：CPU
- L3：Mooncake（分布式缓存池）

此时系统核心价值是“容量扩展 + 缓存共享 + 冷热分层治理”。

---

## 6. SGLang 什么时候会同时用到 Transfer Engine + Mooncake Store？（重点）

在下面这些场景中，两者通常会同时出现：

### 场景 A：SGLang HiCache 使用 Mooncake 作为 L3，且跨节点拉取远端缓存

- SGLang 通过 Mooncake Store 定位 KV 所在节点与对象元数据；
- 真正把远端 KV 拉到本地时，Store 内部通过 Transfer Engine 发起批量读取/拷贝；
- 结果是：
  - **Store 负责“找数据、管数据”**
  - **Transfer Engine 负责“快传数据”**

### 场景 B：内存不够触发 offload，随后远端回源读取

- 节点将部分 KV 从内存下沉到 Mooncake Store 的持久层（FileStorage + Backend）；
- 之后其他节点或后续请求需要这些 KV 时：
  1) 先由 Store 完成对象定位与 offload 对象读取编排；
  2) 再由 Transfer Engine 执行高吞吐传输到目标 buffer。

### 场景 C：既做推理解耦（PD/EPD），又启用分层缓存

- 一部分链路是“阶段间传输”（偏 Transfer Engine 能力）；
- 另一部分链路是“层级缓存命中/回源”（偏 Store 能力）；
- 实际生产中二者叠加，可以同时改善**时延**与**容量**。

---

## 7. 用调用链看“二者协同”

以 Mooncake Store 的 offload 回读路径为例，可抽象为：

1. 请求方向远端 store 服务发起 `batch_get_offload_object`（拿到远端 buffer 指针等信息）；
2. 本地根据返回的 `transfer_engine_addr + pointers` 调用 `BatchGetOffloadObject`；
3. `BatchGetOffloadObject` 再提交到 `transfer_submitter`；
4. `transfer_submitter` 打开远端 segment，生成 `READ` 请求并批量提交给 Transfer Engine。

这说明：**Mooncake Store 的“对象级语义”与 Transfer Engine 的“字节级高速搬运”是分层协同关系，而不是替代关系。**

---

## 8. 选型建议（SGLang 视角）

- 你的主要瓶颈是“阶段间跨机传输时延/带宽” → 优先用 Transfer Engine 路径。
- 你的主要瓶颈是“KV 容量与跨实例共享” → 优先用 Mooncake Store / HiCache 路径。
- 你既要低时延又要大容量，且存在冷热数据流动 → 同时启用 Transfer Engine + Mooncake Store。

实践上，很多线上系统会从“只开传输后端”起步，再逐步引入 Store 的分层与 offload 能力。

---

## 9. 总结

- Mooncake 是一个“控制面 + 数据面 + 存储面”协同的 KVCache 基础设施。
- Mooncake Store 在整体中承担对象管理、元数据协调、分层存储治理与对接上层框架的核心职责。
- 在 SGLang 中，**只做解耦传输**时可偏 Transfer Engine；**做分层缓存**时偏 Mooncake Store；而在跨节点回源、offload 回读、或 PD/EPD + HiCache 叠加场景下，二者会自然同时使用。
