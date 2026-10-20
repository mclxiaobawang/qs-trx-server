# QS TRX Server

TRX 策略专用后端，包含 Bitget 现货实盘执行、独立批次账本、研究回放、账户与理财到账核对、只读监控、Telegram 通知和在线备份。

仓库仅保留 TRX 功能及其依赖。Python 模块继续使用 `freqtrade.strategy.trx_phase1` 和 `freqtrade.rpc.api_server` 路径，以兼容已有部署脚本和保存任务；不再包含 Freqtrade 通用机器人、其他交易所、FreqAI、Hyperopt 或旧原生策略适配器。

## 本地运行

需要 Python 3.12 或更新版本：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
mkdir -p user_data
cp config_examples/server.example.json user_data/config.trx.json
```

在本地配置文件的 `api_server` 中填写 `username`、`password` 和至少 32 字符的随机 `jwt_secret_key`，也可通过 `TRX_API_USERNAME`、`TRX_API_PASSWORD`、`TRX_API_JWT_SECRET_KEY` 覆盖。配置文件、账户凭证、SQLite 账本和运行数据均不进入 Git。

```bash
.venv/bin/trx-server --config user_data/config.trx.json
```

配置示例监听 `0.0.0.0:8080`；本机专用时可改为 `127.0.0.1`。生产环境使用 HTTPS 反向代理，并在 `CORS_origins` 中填写实际前端来源。

`GET /api/v1/ping` 是公开健康检查；`POST /api/v1/token/login` 使用 HTTP Basic 返回 access/refresh token，`POST /api/v1/token/refresh` 刷新访问令牌。TRX 接口统一位于 `/api/v1/trx`，需要登录凭证或 access token。接口文档默认关闭，可设置 `enable_openapi: true` 后查看 `/docs`。

默认 `existing_live_only: true`、`auto_resume_live: false`。首次启动不会自动创建或启动实盘账户。实盘准备、复核和启动继续使用原有 TRX 接口与 review hash 流程。交易进程只运行一个 worker；不要并行启动多个实例操作同一账户。

## Docker

先准备上述配置和 `user_data` 目录，并将其写入权限授予容器 UID 10001。设置三个 `TRX_API_*` 环境变量后运行：

```bash
docker compose up --build -d
```

Compose 只在宿主机 `127.0.0.1:8080` 发布端口，容器内存及 memory+swap 上限均为 512 MiB。已有线上实例不会因本地代码清理自动更新；实际部署需使用新的 `trx-server` 入口。

## 研究与运维

本地历史回放使用 Bitget TRX/USDT 一分钟 Feather 行情：

```bash
.venv/bin/trx-replay --data user_data/data/bitget/TRX_USDT-1m.feather
```

独立监控、历史读取、备份和恢复工具仍位于 `freqtrade/strategy/trx_phase1`。监控可继续使用 `PYTHONPATH=freqtrade/strategy python -m trx_phase1.live_ops_app` 启动，不加载交易执行客户端。详细边界见 [TRX 策略与运维说明](docs/trx-phase1.md)。

## 验证

```bash
.venv/bin/python -m pytest
.venv/bin/ruff check freqtrade tests
```

测试使用隔离的临时账户及模拟交易所，不提交真实订单。两个历史归档研究测试依赖未纳入版本控制的原始实验包，缺失时明确跳过。

本项目从 Freqtrade 的 TRX 定制实现提取，保留原 [GPLv3 许可证](LICENSE)。
