"""Tree-sitter AST 切分模块。

把源码文件按函数/类/方法等语义单元切成代码块，保持语义完整。
不支持的语言或解析失败时，自动回退到按行切分。
"""
from __future__ import annotations

import hashlib
import importlib
from dataclasses import dataclass, field
from pathlib import Path

from tree_sitter import Language, Parser


# 文件扩展名 → tree-sitter 语言名
EXT_TO_LANG = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".kt": "kotlin",
    ".scala": "scala",
    ".swift": "swift",
    ".sql": "sql",
    ".sh": "bash",
    ".lua": "lua",
    # 非代码但承载架构/配置语义的文件（无 parser，走按行切分）
    ".md": "markdown",
    ".markdown": "markdown",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".json": "json",
    ".toml": "toml",
    ".properties": "properties",
    ".proto": "proto",
}

# 各语言中代表"函数/类/方法"的 AST 节点类型
SPLIT_NODE_TYPES = {
    "python": {"function_definition", "class_definition", "decorated_definition"},
    "javascript": {"function_declaration", "class_declaration", "method_definition"},
    "typescript": {"function_declaration", "class_declaration", "method_definition", "interface_declaration", "enum_declaration"},
    "tsx": {"function_declaration", "class_declaration", "method_definition", "interface_declaration", "enum_declaration"},
    "java": {"method_declaration", "class_declaration", "interface_declaration", "constructor_declaration", "enum_declaration"},
    "go": {"function_declaration", "method_declaration", "type_declaration"},
    "rust": {"function_item", "impl_item", "struct_item", "trait_item", "enum_item"},
    "c": {"function_definition", "struct_specifier"},
    "cpp": {"function_definition", "class_specifier", "struct_specifier"},
    "csharp": {"method_declaration", "class_declaration", "interface_declaration", "struct_declaration"},
    "ruby": {"method", "class", "module", "singleton_method"},
    "php": {"function_definition", "method_declaration", "class_declaration", "interface_declaration"},
    "kotlin": {"function_declaration", "class_declaration", "object_declaration"},
    "scala": {"function_definition", "class_definition", "object_definition", "trait_definition"},
    "swift": {"function_declaration", "class_declaration", "protocol_declaration"},
    "lua": {"function_declaration", "function_definition"},
}

# 各语言中代表“函数调用”的 AST 节点类型（用于构建 call graph）
CALL_NODE_TYPES = {
    "python": {"call"},
    "javascript": {"call_expression", "new_expression"},
    "typescript": {"call_expression", "new_expression"},
    "tsx": {"call_expression", "new_expression"},
    "java": {"method_invocation", "object_creation_expression"},
    "go": {"call_expression"},
    "rust": {"call_expression", "macro_invocation"},
    "c": {"call_expression"},
    "cpp": {"call_expression"},
    "csharp": {"invocation_expression", "object_creation_expression"},
    "ruby": {"call", "method_call"},
    "php": {"function_call_expression", "member_call_expression", "scoped_call_expression"},
    "kotlin": {"call_expression"},
    "scala": {"call_expression"},
    "swift": {"call_expression"},
    "lua": {"function_call"},
}

# call 节点中“被调用名”可能所在的字段（type 覆盖 Java/C# 的 object_creation_expression）
_CALL_NAME_FIELDS = ("function", "name", "method", "macro", "constructor", "type")
# 成员调用的接收者字段（如 Java method_invocation 的 object）：
# 记录接收者标识符，支持“谁调用了 PriceRuleEvaluator”类意图查询
# （Java 写法 priceRuleEvaluator.evaluate() 里类名不出现在方法名中）
_CALL_RECEIVER_FIELDS = ("object", "receiver")
# 标识符类叶子节点（被调用名的末端）
_NAME_LEAF_TYPES = {
    "identifier", "field_identifier", "type_identifier", "name",
    "property_identifier", "constant", "simple_identifier",
}

# 常见内置/容器方法，作为调用名无区分度，构图时过滤
_CALL_STOPLIST = {
    "print", "len", "range", "enumerate", "zip", "map", "filter", "sorted",
    "list", "dict", "set", "tuple", "str", "int", "float", "bool", "bytes",
    "open", "isinstance", "type", "super", "getattr", "setattr", "hasattr",
    "min", "max", "sum", "any", "all", "abs", "repr", "format", "iter", "next",
    "append", "extend", "insert", "remove", "pop", "get", "items", "keys",
    "values", "update", "add", "join", "split", "strip", "replace", "decode",
    "encode", "startswith", "endswith", "lower", "upper", "find", "index",
    "getenv", "setdefault", "copy", "clear", "sort", "reverse", "count",
    # 接收者噪声（日志/自引用等无区分度标识符）
    "this", "self", "log", "logger", "console", "System",
}

# 超过此行数的容器节点（如大类）会递归切内部定义
MAX_CHUNK_LINES = 200
# 按行切分的窗口与重叠
LINE_WINDOW = 60
LINE_OVERLAP = 10
# 过小的块（行数与字符数都不足）会被丢弃
MIN_CHUNK_LINES = 3
MIN_CHUNK_CHARS = 40

_PARSER_CACHE: dict = {}


@dataclass
class CodeChunk:
    """一个语义代码块。"""

    file_path: str
    language: str
    symbol: str          # 函数/类名，行切块为 lines_<起始行>
    start_line: int      # 1-indexed
    end_line: int        # 1-indexed
    code: str
    blob_hash: str = ""  # SHA256(file_path + code)，用于内容寻址增量
    calls: list[str] = field(default_factory=list)  # 本块内调用的函数名（call graph 边）

    def __post_init__(self) -> None:
        if not self.blob_hash:
            h = hashlib.sha256()
            h.update(self.file_path.encode("utf-8", "ignore"))
            h.update(b"\x00")
            h.update(self.code.encode("utf-8", "ignore"))
            self.blob_hash = h.hexdigest()


# 语言名 -> (pip 模块名, 取 Language 的函数名)
# 使用独立语言包（含预编译 wheel），避免运行时联网下载 grammar
_LANG_MODULES = {
    "python": ("tree_sitter_python", "language"),
    "javascript": ("tree_sitter_javascript", "language"),
    "typescript": ("tree_sitter_typescript", "language_typescript"),
    "tsx": ("tree_sitter_typescript", "language_tsx"),
    "java": ("tree_sitter_java", "language"),
    "go": ("tree_sitter_go", "language"),
    "rust": ("tree_sitter_rust", "language"),
    "c": ("tree_sitter_c", "language"),
    "cpp": ("tree_sitter_cpp", "language"),
    "csharp": ("tree_sitter_c_sharp", "language"),
    "ruby": ("tree_sitter_ruby", "language"),
    "php": ("tree_sitter_php", "language_php"),
    "bash": ("tree_sitter_bash", "language"),
}


def _get_parser(lang: str):
    """获取并缓存 parser，失败返回 None（该语言回退按行切）。"""
    if lang in _PARSER_CACHE:
        return _PARSER_CACHE[lang]
    parser = None
    spec = _LANG_MODULES.get(lang)
    if spec is not None:
        mod_name, func_name = spec
        try:
            mod = importlib.import_module(mod_name)
            language = Language(getattr(mod, func_name)())
            parser = Parser(language)
        except Exception:
            parser = None
    _PARSER_CACHE[lang] = parser
    return parser


def _get_symbol(node) -> str:
    """提取节点的符号名（函数名/类名）。"""
    # 装饰器节点（如 Python @dataclass class）：递归取内部真正定义的名字
    if node.type == "decorated_definition":
        for child in node.children:
            if child.type.endswith(("definition", "declaration")):
                return _get_symbol(child)
    name_node = node.child_by_field_name("name")
    if name_node is not None:
        return name_node.text.decode("utf-8", "ignore")
    # 回退：找第一个标识符类子节点
    for child in node.children:
        if child.type in ("identifier", "type_identifier", "field_identifier", "name"):
            return child.text.decode("utf-8", "ignore")
    return node.type


def _has_split_descendant(node, split_types: set) -> bool:
    """node 内部（不含自身）是否还有可切分节点。"""
    for child in node.children:
        if child.type in split_types:
            return True
        if _has_split_descendant(child, split_types):
            return True
    return False


def _last_identifier(node) -> str | None:
    """取节点末端的标识符文本（如 a.b.c() 取 c）。"""
    if node.type in _NAME_LEAF_TYPES and node.child_count == 0:
        return node.text.decode("utf-8", "ignore")
    last = None
    for child in node.children:
        r = _last_identifier(child)
        if r:
            last = r
    return last


# 泛型实参子树（Java type_arguments / C# type_argument_list），取类型名时跳过
_TYPE_ARG_NODE_TYPES = {"type_arguments", "type_argument_list"}


def _type_name(node) -> str | None:
    """取类型节点的主类型名：跳过泛型实参后取末端标识符。

    new ArrayList<String>() -> ArrayList（而非 String）；
    new com.foo.Bar()       -> Bar。
    """
    if node.type in _NAME_LEAF_TYPES and node.child_count == 0:
        return node.text.decode("utf-8", "ignore")
    last = None
    for child in node.children:
        if child.type in _TYPE_ARG_NODE_TYPES:
            continue
        r = _type_name(child)
        if r:
            last = r
    return last


def _callee_name(call_node) -> str | None:
    """从 call 节点提取被调用函数名（取末端标识符；type 字段走类型名提取）。"""
    target = None
    matched_field = None
    for f in _CALL_NAME_FIELDS:
        target = call_node.child_by_field_name(f)
        if target is not None:
            matched_field = f
            break
    if target is None and call_node.child_count:
        target = call_node.children[0]
    if target is None:
        return None
    if matched_field == "type":
        return _type_name(target)
    return _last_identifier(target)


def _extract_calls(node, lang: str) -> list[str]:
    """提取节点子树内所有函数调用的被调用名（去重，保序）。"""
    call_types = CALL_NODE_TYPES.get(lang)
    if not call_types:
        return []
    names: list[str] = []
    seen: set[str] = set()

    def _visit(n):
        if n.type in call_types:
            name = _callee_name(n)
            if name and name not in seen and name.isidentifier() and name not in _CALL_STOPLIST:
                seen.add(name)
                names.append(name)
            # 成员调用的接收者（obj.method() 的 obj）也记一条边，
            # 使“谁调用了某类”能通过实例字段名命中调用方
            for f in _CALL_RECEIVER_FIELDS:
                r = n.child_by_field_name(f)
                if r is None:
                    continue
                recv = _last_identifier(r)
                if recv and recv not in seen and recv.isidentifier() and recv not in _CALL_STOPLIST:
                    seen.add(recv)
                    names.append(recv)
                break
        for c in n.children:
            _visit(c)

    _visit(node)
    return names


def _make_chunk(node, lang: str, file_path: str) -> CodeChunk:
    code = node.text.decode("utf-8", "ignore")
    return CodeChunk(
        file_path=file_path,
        language=lang,
        symbol=_get_symbol(node),
        start_line=node.start_point[0] + 1,
        end_line=node.end_point[0] + 1,
        code=code,
        calls=_extract_calls(node, lang),
    )


def _walk(node, chunks: list, lang: str, file_path: str) -> None:
    """递归遍历 AST，收集可切分节点。

    小定义整体成块；大容器（如大类）递归切内部，避免单块过大。
    """
    split_types = SPLIT_NODE_TYPES.get(lang, set())
    if not split_types:
        return
    for child in node.children:
        if child.type in split_types:
            n_lines = child.end_point[0] - child.start_point[0] + 1
            has_sub = _has_split_descendant(child, split_types)
            if n_lines <= MAX_CHUNK_LINES or not has_sub:
                chunks.append(_make_chunk(child, lang, file_path))
            else:
                _walk(child, chunks, lang, file_path)
        else:
            _walk(child, chunks, lang, file_path)


def _chunk_by_lines(file_path: str, code: str, lang: str) -> list[CodeChunk]:
    """回退策略：按固定行窗口 + 重叠切分。"""
    lines = code.split("\n")
    chunks: list[CodeChunk] = []
    step = max(1, LINE_WINDOW - LINE_OVERLAP)
    for i in range(0, len(lines), step):
        block = lines[i : i + LINE_WINDOW]
        text = "\n".join(block)
        if not text.strip():
            continue
        chunks.append(
            CodeChunk(
                file_path=file_path,
                language=lang,
                symbol=f"lines_{i + 1}",
                start_line=i + 1,
                end_line=i + len(block),
                code=text,
            )
        )
        if i + LINE_WINDOW >= len(lines):
            break
    return chunks


def _is_meaningful(chunk: CodeChunk) -> bool:
    n_lines = chunk.end_line - chunk.start_line + 1
    return n_lines >= MIN_CHUNK_LINES or len(chunk.code.strip()) >= MIN_CHUNK_CHARS


def chunk_file(file_path: str, code: str | None = None) -> list[CodeChunk]:
    """把一个文件切成代码块。

    参数:
        file_path: 文件路径（用于推断语言与生成 blob_hash）
        code: 文件内容，None 时自动读取
    返回:
        CodeChunk 列表
    """
    path = Path(file_path)
    if code is None:
        try:
            code = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return []
    if not code.strip():
        return []

    lang = EXT_TO_LANG.get(path.suffix.lower())
    if not lang:
        return _chunk_by_lines(file_path, code, "text")

    parser = _get_parser(lang)
    if parser is None:
        return _chunk_by_lines(file_path, code, lang)

    try:
        tree = parser.parse(bytes(code, "utf-8"))
        chunks: list[CodeChunk] = []
        _walk(tree.root_node, chunks, lang, file_path)
        chunks = [c for c in chunks if _is_meaningful(c)]
        if not chunks:
            return _chunk_by_lines(file_path, code, lang)
        return chunks
    except Exception:
        return _chunk_by_lines(file_path, code, lang)
