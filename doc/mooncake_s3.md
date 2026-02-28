# Mooncake Store 存储引擎与 HF3FS 插件分析报告

本文基于 `mooncake-store/src/` 与 `mooncake-store/src/hf3fs/` 的实现，梳理 Mooncake 存储引擎工作原理，并给出“实现一个类似 HF3FS 的 S3 定制存储插件”所需工作项。

---

## 1. 总体架构：FileStorage + StorageBackendInterface

Mooncake 的存储路径可以理解为两层：

1. **控制与调度层：`FileStorage`**
   - 负责初始化、注册本地内存、周期性 heartbeat、向 master 汇报 offload 成功、接收待 offload key 列表并触发实际落盘。
2. **数据落盘层：`StorageBackendInterface` 的具体后端实现**
   - 当前有三类：
     - `BucketStorageBackend`（默认）
     - `StorageBackendAdaptor`（file-per-key）
     - `OffsetAllocatorStorageBackend`（偏移分配器）

入口是 `CreateStorageBackend`，根据 `MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR` 决定实例化哪种后端。

---

## 2. FileStorage 的核心工作流

`FileStorage::Init()` 主要做四件事：

1. 注册本地内存给数据平面（`RegisterLocalMemory`）
2. 初始化后端（`storage_backend_->Init()`）
3. 扫描本地已有元数据并重新通知 master（`ScanMeta + NotifyOffloadSuccess`）
4. 启动后台线程：heartbeat 与 client buffer GC

运行时，关键链路如下：

- **Heartbeat**：`client_->OffloadObjectHeartbeat` 拉取待 offload 对象。
- **OffloadObjects**：
  - 若后端是 bucket，会先按 bucket 约束分组（`AllocateOffloadingBuckets`）。
  - 再通过 `BatchQuerySegmentSlices` 从本机内存副本取到切片，调用 `storage_backend_->BatchOffload` 写入后端。
  - 成功后通过 `NotifyOffloadSuccess` 更新 master 元数据。
- **BatchGet**：客户端取数据时，先从本地 buffer allocator 分配目标内存，再调用 `storage_backend_->BatchLoad` 回填数据。

这个设计让 `FileStorage` 只关心“何时 offload / 如何与 master 协调”，而把“具体如何持久化”下沉给后端插件接口。

---

## 3. StorageBackendInterface 抽象与插件边界

`StorageBackendInterface` 是插件化最关键的稳定边界，核心方法包括：

- `Init()`：初始化后端、恢复状态
- `BatchOffload()`：批量写入，成功后通过回调返回每个 key 的 `StorageObjectMetadata`
- `BatchLoad()`：按 metadata 回读数据
- `IsExist()`：判断 key 是否已存在
- `IsEnableOffloading()`：容量与策略判定
- `ScanMeta()`：重启恢复时迭代所有 metadata 回放到 master

**如果要做 S3 插件，最直接路径就是新增一个实现该接口的 `S3StorageBackend`。**

---

## 4. 当前三种后端实现的差异（对 S3 设计有借鉴）

### 4.1 BucketStorageBackend（默认）

特征：

- 把多个 key 打包进一个 bucket 文件（`*.bucket`）并有独立 metadata 文件（`*.metadata`）。
- `BuildBucket` 阶段把每个对象编码为：`key bytes + value bytes`，并记录 `(offset, key_size, data_size)`。
- `BatchLoad` 会按 bucket 分组读取，减少文件打开次数。
- 通过 `BucketReadGuard` 的 `inflight_reads_` 计数实现“删除 bucket 前等待并发读完成”的安全删除。
- `Init` 会扫描磁盘恢复 bucket 元数据，同时清理“仅有数据文件无 metadata”的 orphan bucket。

适用场景：大量小对象 offload，写入和读取都希望批量化。

### 4.2 FilePerKey（StorageBackendAdaptor + StorageBackend）

特征：

- 每个 key 对应一个文件路径。
- `StorageBackend` 封装了通用文件 I/O 与本地配额/驱逐（FIFO）逻辑。
- 支持 POSIX / io_uring /（编译开启时）3FS 三种 `StorageFile` 实现。

适用场景：实现简单，便于调试，但元数据与 inode 压力较大。

### 4.3 OffsetAllocatorStorageBackend

特征：

- 使用单数据文件（`kv_cache.data`）+ 偏移分配器管理空间。
- 每条记录有 header（key_len/value_len）+ key + value，支持覆盖与回收。
- 用分片锁 + 原子计数维护高并发 map 与配额统计。

适用场景：减少文件数量，面向高吞吐本地块设备。

---

## 5. HF3FS 插件是如何接入的

HF3FS 的实现本质是“新增一种 `StorageFile` 实现（`ThreeFSFile`）并在文件打开处按条件切换”。

### 5.1 接入点

- `StorageFile` 是统一 I/O 抽象（write/read/vector_write/vector_read）。
- `StorageBackend::create_file()` 在 `USE_3FS` 且路径识别为 3FS mount 时：
  - 调用 `hf3fs_reg_fd` 注册 fd
  - 返回 `ThreeFSFile`；否则回退 POSIX / uring

即：上层后端逻辑无需改，只替换了底层文件 I/O 实现。

### 5.2 线程资源模型

HF3FS 使用 `USRBIOResourceManager` 维护**线程级资源**：

- 每线程一套 `hf3fs_iov` + 读写 `hf3fs_ior`
- 首次访问线程时懒初始化，析构时统一清理

这样避免跨线程共享 ring/buffer 的复杂同步，符合高性能 I/O 常见模式。

### 5.3 ThreeFSFile 读写路径

`ThreeFSFile` 的 read/write/vector_* 的共同点：

- 根据 `iov_size` 分块
- 数据在“用户 buffer ↔ hf3fs shared iov buffer”间拷贝
- 调 `hf3fs_prep_io -> hf3fs_submit_ios -> hf3fs_wait_for_ios` 完成一次批次 I/O

写失败时析构会尝试删除损坏文件，保证后续恢复不读到半写数据。

---

## 6. 如果要实现“类似 HF3FS 的 S3 定制插件”，应怎么做

这里给出两条实现路径，建议优先 **路径 A（后端级插件）**。

### 路径 A（推荐）：新增 `S3StorageBackend`，实现 StorageBackendInterface

这是更自然、可维护的对象存储接入方式。

#### A.1 目录与构建

- 新建目录：`mooncake-store/src/s3/`、`mooncake-store/include/s3/`
- 新增 `S3StorageBackend` 类（继承 `StorageBackendInterface`）
- 在 `CreateStorageBackend()` 增加 `StorageBackendType::kS3` 分支
- 扩展环境变量：
  - `MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR=s3_storage_backend`
  - `MOONCAKE_S3_ENDPOINT / ACCESS_KEY / SECRET_KEY / REGION / BUCKET / PREFIX`
- CMake 增加可选编译开关（如 `USE_S3=ON`）与 SDK 链接

#### A.2 元数据模型（必须先定）

需要定义如何把 `StorageObjectMetadata` 映射到 S3 对象：

- **方案 1：一 key 一 object（简单）**
  - `metadata.bucket_id` 可复用为逻辑分组 ID 或置 0
  - `offset/key_size/data_size` 可退化（如 offset=0）
- **方案 2：bucket 打包后上传单 object（更接近当前 bucket backend）**
  - 上传 `bucket data object` + `bucket metadata object`
  - 读取时按 offset range GET

建议先做方案 1 跑通，再演进方案 2 追求吞吐。

#### A.3 实现接口方法

1. `Init()`
   - 初始化 S3 client
   - 校验 bucket 可访问
   - 从本地 checkpoint 或对象存储索引恢复 key->metadata 映射
2. `BatchOffload()`
   - 批量 PUT（可并发）
   - 成功后构造 `StorageObjectMetadata` 回调 `complete_handler`
   - 失败策略：全失败返回错误，或部分成功（需定义幂等与回滚）
3. `BatchLoad()`
   - 基于 metadata 发起 GET / Range GET
   - 回填到传入 Slice
4. `IsExist()`
   - 优先查本地内存索引（避免频繁 HEAD）
5. `IsEnableOffloading()`
   - 本地维护 total_keys/total_size 配额计数
6. `ScanMeta()`
   - 从本地持久化索引迭代回放给 master

#### A.4 一致性与恢复（重点）

对象存储插件常见问题不是“写 API”，而是“元数据一致性”：

- 建议引入**两阶段语义**：
  1) 先写对象（含校验信息，如 CRC/etag）
  2) 再原子更新本地索引（WAL/manifest）
- 重启恢复时：
  - 重放 WAL
  - 清理“对象已写但索引未提交”的悬挂记录（按策略补提交或回收）
- 必须定义幂等 key（如 object key 带版本号或 content hash），避免重试写脏。

#### A.5 性能优化建议

- 并发上传下载：线程池 + 请求聚合
- 大对象 multi-part upload；小对象批量化
- 本地热缓存（可复用 `LocalHotCache` 思路）
- 限速与重试退避
- 可观测性：成功率、P99 延迟、重试次数、吞吐

---

### 路径 B：模仿 HF3FS，仅替换 StorageFile 为 `S3File`（不推荐主路径）

理论上可做，但 S3 不是 POSIX 语义：

- 无原生随机覆盖写（通常要重写对象或 multipart compose）
- `vector_write`/`vector_read` 语义很难一一映射
- open/close/fd 模型与对象存储模型不匹配

因此“像 HF3FS 那样只换 File 层”在 S3 上会引入较多语义适配复杂度，最终仍会逼近后端级改造。

---

## 7. 推荐的最小可行实现（MVP）清单

1. 新增 `kS3` backend 类型 + `CreateStorageBackend` 分支
2. `S3StorageBackend::Init/BatchOffload/BatchLoad/IsExist/IsEnableOffloading/ScanMeta` 六大接口实现
3. 先做“一 key 一 object”
4. 本地 manifest（例如 RocksDB/SQLite/append-only log）记录 key->metadata
5. 重启时 `ScanMeta` 回放到 master
6. 失败注入与回归测试：
   - 网络抖动、超时、部分成功
   - 重试幂等
   - 进程崩溃恢复一致性

---

## 8. 结论

- Mooncake 当前的插件化核心不在 `FileStorage`，而在 `StorageBackendInterface`。
- HF3FS 是“文件 I/O 层替换”的范例；S3 更适合“后端实现层替换”。
- 若按上述路径 A 推进，可以在不破坏现有 master/client 协议的前提下，增量引入对象存储能力，并逐步优化性能与一致性保障。
