#!/usr/bin/env node
/**
 * semcode-mcp npm 薄壳：把 MCP server 启动透传给 Python 包。
 *
 * 解析顺序：
 *   1. uvx semcode-mcp==<本包版本>  （推荐：uv 隔离环境；钉版本防随 PyPI latest 漂移）
 *   2. python -m semantic_code_mcp.server（已 pip install 场景；不探 semcode-mcp
 *      可执行名 —— 在 npm 全局 bin 目录里会命中本 shim 自身导致无限递归）
 * 都不可用时打印安装指引退出。
 *
 * stdio 直通（MCP 协议走 stdin/stdout），信号与退出码原样转发。
 */
"use strict";

const { spawnSync, spawn } = require("node:child_process");

const PYPI_PACKAGE = "semcode-mcp";
const PKG_VERSION = require("./package.json").version;
const PY_MODULE = "semantic_code_mcp.server";

function exists(cmd) {
  const probe = process.platform === "win32" ? "where" : "which";
  return spawnSync(probe, [cmd], { stdio: "ignore" }).status === 0;
}

function pythonWithModule() {
  // 找一个已 pip install 本包的 Python（python → python3）
  for (const py of ["python", "python3"]) {
    if (!exists(py)) continue;
    const r = spawnSync(py, ["-c", `import ${PY_MODULE.split(".")[0]}`], { stdio: "ignore" });
    if (r.status === 0) return py;
  }
  return null;
}

function main() {
  const args = process.argv.slice(2);
  let file = null;
  let fileArgs = [];

  const py = exists("uvx") ? null : pythonWithModule();
  if (exists("uvx")) {
    file = "uvx";
    // 钉到与 npm 包同版本（release 流程校验两端版本一致），
    // 旧 npm 包不会随 PyPI latest 漂移
    fileArgs = [`${PYPI_PACKAGE}==${PKG_VERSION}`, ...args];
  } else if (py) {
    file = py;
    fileArgs = ["-m", PY_MODULE, ...args];
  } else {
    process.stderr.write(
      [
        `semcode-mcp: no Python runtime launcher found.`,
        ``,
        `Install one of:`,
        `  1. uv (recommended): https://docs.astral.sh/uv/getting-started/installation/`,
        `     then this wrapper runs: uvx ${PYPI_PACKAGE}==${PKG_VERSION}`,
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
    process.stderr.write(`semcode-mcp: failed to launch ${file}: ${err.message}\n`);
    process.exit(1);
  });
}

main();
