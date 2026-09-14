"""本地 PostgreSQL 生命周期管理（本机没有 Docker，用它替代 docker compose）。

必须用 conda 环境里的 python 运行，脚本会从 sys.executable 反推出 PostgreSQL 的
可执行文件在哪（同一个环境里装了 python 和 postgresql）：

    E:\\conda-envs\\paperqa\\python.exe scripts\\db.py init       # 首次：初始化数据目录
    E:\\conda-envs\\paperqa\\python.exe scripts\\db.py start      # 启动
    E:\\conda-envs\\paperqa\\python.exe scripts\\db.py create-db  # 建 paperqa 数据库
    E:\\conda-envs\\paperqa\\python.exe scripts\\db.py status     # 看状态
    E:\\conda-envs\\paperqa\\python.exe scripts\\db.py stop       # 停止
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

PG_USER = "postgres"
PG_PASSWORD = "postgres"
PG_PORT = "5432"
DB_NAME = "paperqa"

# 数据目录默认放 E 盘（C 盘空间紧张），可用环境变量覆盖
DATA_DIR = Path(os.environ.get("PAPERQA_PGDATA", r"E:\conda-envs\pgdata"))
LOG_FILE = DATA_DIR.parent / "pgdata.log"


def pg_bin_dir() -> Path:
    """从当前 python 解释器推出同环境内 PostgreSQL 的 bin 目录。"""
    bin_dir = Path(sys.executable).parent / "Library" / "bin"
    if not (bin_dir / "pg_ctl.exe").exists():
        raise SystemExit(
            f"找不到 pg_ctl.exe（{bin_dir}）。请用 conda 环境的 python 运行本脚本：\n"
            r"  E:\conda-envs\paperqa\python.exe scripts\db.py <命令>"
        )
    return bin_dir


def run(
    exe: str,
    *args: str,
    env: dict[str, str] | None = None,
    detach: bool = False,
) -> int:
    """调用 PostgreSQL 原生工具。

    detach=True 用于 `pg_ctl start`：它会派生一个长期存活的 postgres.exe，而子进程会
    继承本进程的标准输出。如果不切断，任何 `python scripts/db.py start | tail` 之类的
    管道都会永久阻塞（tail 等到所有写端关闭才输出，但 postgres 一直活着）。
    启动诊断信息本来就会写进 -l 指定的日志文件，所以这里直接丢弃即可。
    """
    command = [str(pg_bin_dir() / exe), *args]
    if detach:
        return subprocess.run(
            command, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        ).returncode
    return subprocess.run(command, env=env).returncode


def cmd_init() -> None:
    """初始化数据目录：只在第一次需要，相当于建一个全新的空数据库集群。"""
    if (DATA_DIR / "PG_VERSION").exists():
        print(f"数据目录已初始化，跳过：{DATA_DIR}")
        return

    DATA_DIR.parent.mkdir(parents=True, exist_ok=True)
    # initdb 需要一个只含密码的文件来设置超级用户密码
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(PG_PASSWORD)
        password_file = f.name

    try:
        code = run(
            "initdb.exe",
            "-D", str(DATA_DIR),
            "-U", PG_USER,
            f"--pwfile={password_file}",
            "-E", "UTF8",
            "--locale=C",
        )
        if code != 0:
            raise SystemExit("initdb 失败")
        print(f"\n初始化完成：{DATA_DIR}")
    finally:
        os.unlink(password_file)


def cmd_start() -> None:
    if not (DATA_DIR / "PG_VERSION").exists():
        raise SystemExit(f"数据目录还没初始化：{DATA_DIR}\n先运行：python scripts/db.py init")

    code = run(
        "pg_ctl.exe",
        "-D", str(DATA_DIR),
        "-l", str(LOG_FILE),
        "-o", f"-p {PG_PORT}",
        "-w",              # 等到数据库真正可接受连接再返回
        "-t", "30",        # 最多等 30 秒，避免启动失败时无限等待
        "start",
        detach=True,
    )
    if code != 0:
        raise SystemExit("启动失败，查看日志：" + str(LOG_FILE))
    print(f"PostgreSQL 已启动，端口 {PG_PORT}，数据目录 {DATA_DIR}")


def cmd_stop() -> None:
    run("pg_ctl.exe", "-D", str(DATA_DIR), "-m", "fast", "stop")
    print("已停止")


def cmd_status() -> None:
    run("pg_ctl.exe", "-D", str(DATA_DIR), "status")


def cmd_create_db() -> None:
    """创建 paperqa 数据库。postgres 这个默认库是给管理用的，业务库要单独建。"""
    env = {**os.environ, "PGPASSWORD": PG_PASSWORD}
    code = run(
        "createdb.exe",
        "-h", "localhost", "-p", PG_PORT, "-U", PG_USER,
        DB_NAME,
        env=env,
    )
    if code != 0:
        # 已存在时 createdb 返回非 0，这里不当成错误
        print(f"数据库 {DB_NAME} 可能已存在，继续")
        return
    print(f"数据库 {DB_NAME} 已创建")


def cmd_psql() -> None:
    """开一个 psql 交互终端，方便手写 SQL 观察数据。"""
    env = {**os.environ, "PGPASSWORD": PG_PASSWORD}
    run("psql.exe", "-h", "localhost", "-p", PG_PORT, "-U", PG_USER, "-d", DB_NAME, env=env)


COMMANDS = {
    "init": cmd_init,
    "start": cmd_start,
    "stop": cmd_stop,
    "status": cmd_status,
    "create-db": cmd_create_db,
    "psql": cmd_psql,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=sorted(COMMANDS))
    args = parser.parse_args()
    COMMANDS[args.command]()


if __name__ == "__main__":
    main()
