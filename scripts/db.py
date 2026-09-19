"""Local PostgreSQL lifecycle management, standing in for docker compose.

Must run under the conda environment's python: PostgreSQL's binaries are located
from sys.executable, so python and postgresql have to come from the same
environment. Commands: init, start, stop, status, create-db, psql.

    python scripts\\db.py <command>
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

# Kept outside the repo on purpose; override with PAPERQA_PGDATA.
DATA_DIR = Path(os.environ.get("PAPERQA_PGDATA", r"E:\conda-envs\pgdata"))
LOG_FILE = DATA_DIR.parent / "pgdata.log"


def pg_bin_dir() -> Path:
    """Return the bin dir of the PostgreSQL installed next to the running interpreter.

    Exits if pg_ctl.exe is absent -- that means the wrong interpreter was used.
    """
    bin_dir = Path(sys.executable).parent / "Library" / "bin"
    if not (bin_dir / "pg_ctl.exe").exists():
        raise SystemExit(
            f"pg_ctl.exe not found ({bin_dir}). Run with the conda environment's python:\n"
            r"  E:\conda-envs\paperqa\python.exe scripts\db.py <command>"
        )
    return bin_dir


def run(
    exe: str,
    *args: str,
    env: dict[str, str] | None = None,
    detach: bool = False,
) -> int:
    """Run one of PostgreSQL's own tools and return its exit code.

    detach=True is for `pg_ctl start`, which spawns a long-lived postgres.exe that
    inherits our stdout/stderr; without discarding them, a pipe such as
    `python scripts/db.py start | tail` would block forever. Startup diagnostics
    go to the -l log file regardless.
    """
    command = [str(pg_bin_dir() / exe), *args]
    if detach:
        return subprocess.run(
            command, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        ).returncode
    return subprocess.run(command, env=env).returncode


def cmd_init() -> None:
    """Create the data directory (a fresh, empty database cluster). Needed once."""
    if (DATA_DIR / "PG_VERSION").exists():
        print(f"Data directory already initialized, skipping: {DATA_DIR}")
        return

    DATA_DIR.parent.mkdir(parents=True, exist_ok=True)
    # initdb reads the superuser password from a file, not from stdin or the command line.
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
            raise SystemExit("initdb failed")
        print(f"\nInitialization complete: {DATA_DIR}")
    finally:
        os.unlink(password_file)


def cmd_start() -> None:
    if not (DATA_DIR / "PG_VERSION").exists():
        raise SystemExit(f"Data directory not initialized: {DATA_DIR}\nRun this first: python scripts/db.py init")

    code = run(
        "pg_ctl.exe",
        "-D", str(DATA_DIR),
        "-l", str(LOG_FILE),
        "-o", f"-p {PG_PORT}",
        "-w",              # return only once the server accepts connections
        "-t", "30",        # cap the wait so a failed start cannot hang forever
        "start",
        detach=True,
    )
    if code != 0:
        raise SystemExit("Start failed, check the log: " + str(LOG_FILE))
    print(f"PostgreSQL started, port {PG_PORT}, data directory {DATA_DIR}")


def cmd_stop() -> None:
    run("pg_ctl.exe", "-D", str(DATA_DIR), "-m", "fast", "stop")
    print("Stopped")


def cmd_status() -> None:
    run("pg_ctl.exe", "-D", str(DATA_DIR), "status")


def cmd_create_db() -> None:
    """Create the paperqa database; the default `postgres` db is administrative."""
    env = {**os.environ, "PGPASSWORD": PG_PASSWORD}
    code = run(
        "createdb.exe",
        "-h", "localhost", "-p", PG_PORT, "-U", PG_USER,
        DB_NAME,
        env=env,
    )
    if code != 0:
        # createdb exits non-zero when the database already exists; not an error here.
        print(f"Database {DB_NAME} may already exist, continuing")
        return
    print(f"Database {DB_NAME} created")


def cmd_psql() -> None:
    """Open an interactive psql session against the paperqa database."""
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
