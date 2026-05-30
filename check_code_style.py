import ast
import sys
import tokenize
from pathlib import Path


def has_docstring(node):
    return bool(node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str))


def check_file(path):
    errors = []
    text = path.read_text()
    with path.open("rb") as handle:
        for tok in tokenize.tokenize(handle.readline):
            if tok.type == tokenize.COMMENT:
                errors.append(f"{path}:{tok.start[0]} comment")
    tree = ast.parse(text)
    nodes = [tree]
    nodes.extend(node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)))
    for node in nodes:
        if has_docstring(node):
            line = node.body[0].lineno
            errors.append(f"{path}:{line} docstring")
    return errors


def main():
    root = Path(__file__).resolve().parent
    errors = []
    for path in sorted(root.glob("*.py")):
        errors.extend(check_file(path))
    if errors:
        print("\n".join(errors))
        sys.exit(1)
    print("ok")


if __name__ == "__main__":
    main()
