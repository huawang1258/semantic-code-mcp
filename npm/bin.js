#!/usr/bin/env node
/**
 * semantic-code-mcp npm 薄壳：把 MCP server 启动透传给 Python 包。
 *
 * 解析顺序：
 *   1. uvx semantic-code-mcp      （推荐：uv 自动管理隔离环境）
 *   2. semantic-code-mcp          （已 pip install 的 console script）
 * 都不可用时打印安装指引退出。
 *
 * stdio 直通（MCP 协议走 stdin/stdout），信号与退出码原样转发。
 */
"use strict";

const { spawnSync, spawn } = require("node:child_process");

const PYPI_PACKAGE = "semantic-code-mcp";

function exists(cmd) {
  const probe = process.platform === "win32" ? "where" : "which";
  return spawnSync(probe, [cmd], { stdio: "ignore" }).status === 0;
}

function main() {
  const args = process.argv.slice(2);
  let file = null;
  let fileArgs = [];

  if (exists("uvx")) {
    file = "uvx";
    fileArgs = [PYPI_PACKAGE, ...args];
  } else if (exists(PYPI_PACKAGE)) {
    file = PYPI_PACKAGE;
    fileArgs = args;
  } else {
    process.stderr.write(
      [
        `semantic-code-mcp: no Python runtime launcher found.`,
        ``,
        `Install one of:`,
        `  1. uv (recommended): https://docs.astral.sh/uv/getting-started/installation/`,
        `     then this wrapper runs: uvx ${PYPI_PACKAGE}`,
        `  2. pip install ${PYPI_PACKAGE}`,
        ``,
      ].join("\n")
    );
    process.exit(1);
  }

  const child = spawn(file, fileArgs, { stdio: "inherit" });
  for (const sig of ["SIGINT", "SIGTERM"]) {
    process.on(sig, () => child.kill(sig));
  }
  child.on("exit", (code, signal) => {
    process.exit(signal ? 1 : code ?? 0);
  });
  child.on("error", (err) => {
    process.stderr.write(`semantic-code-mcp: failed to launch ${file}: ${err.message}\n`);
    process.exit(1);
  });
}

main();
