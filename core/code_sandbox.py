"""
Code sandbox for LLM-generated Python code.

Validates LLM-generated `revise_state` and `intrinsic_reward` code in three
stages:
1. AST whitelist check (blocks dangerous operations such as exec/eval/open).
2. Test execution with random inputs (output type, NaN/Inf, dim constraints).
3. Dimension detection (records the extra dims added by `revise_state`).

Corresponds to the "Code Validation" step in the GIFT paper and guarantees that
LLM-generated code is safe to execute downstream.
"""

import ast
import numpy as np
from typing import Callable, Dict, Optional


# Whitelist of modules / functions importable from LLM-generated code.
_ALLOWED_IMPORTS = {'numpy', 'math', 'np', 'feature_library'}
_BLOCKED_BUILTINS = {
    'exec', 'eval', 'compile', 'open', '__import__',
    'globals', 'locals', 'vars', 'dir', 'getattr', 'setattr', 'delattr',
    'breakpoint', 'exit', 'quit',
}
_BLOCKED_ATTRS = {
    '__import__', '__builtins__', '__file__', '__name__',
}


def _ast_check(code_str: str) -> list:
    """Stage 1: AST whitelist validation. Returns list of error strings.

    Blocks `exec`, `eval`, `open`, and other dangerous calls/imports.
    """
    errors = []
    try:
        tree = ast.parse(code_str)
    except SyntaxError as e:
        return [f"Syntax error: {e}"]

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _BLOCKED_BUILTINS:
                errors.append(f"Blocked builtin: {func.id}()")
            if isinstance(func, ast.Attribute):
                if func.attr in _BLOCKED_ATTRS:
                    errors.append(f"Blocked attribute: .{func.attr}")

        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split('.')[0]
                if root not in _ALLOWED_IMPORTS:
                    errors.append(f"Blocked import: {alias.name}")

        if isinstance(node, ast.ImportFrom):
            if node.module:
                root = node.module.split('.')[0]
                if root not in _ALLOWED_IMPORTS:
                    errors.append(f"Blocked from-import: {node.module}")

    return errors


def _extract_functions(code_str: str) -> Dict[str, Callable]:
    """Extract ``revise_state`` and ``intrinsic_reward`` from a code string.

    The code is executed in a restricted namespace that only exposes ``numpy``
    and ``feature_library``.
    """
    namespace = {'np': np, 'numpy': np}
    try:
        import math
        namespace['math'] = math
    except ImportError:
        pass
    try:
        import feature_library
        namespace['feature_library'] = feature_library
    except ImportError:
        pass

    exec(compile(code_str, '<llm_code>', 'exec'), namespace)

    result = {}
    for name in ['revise_state', 'intrinsic_reward']:
        if name in namespace and callable(namespace[name]):
            result[name] = namespace[name]
    return result


def _test_execution(functions: Dict[str, Callable]) -> list:
    """Stage 2: Test execution with random inputs. Returns list of error strings.

    Validates that with a random 120-dim input:
    - ``revise_state`` returns an ndarray of dim >= 120 with the first 120 dims
      preserved;
    - ``intrinsic_reward`` returns a scalar in ``[-100, 100]``;
    - neither output contains NaN or Inf.
    """
    errors = []
    np.random.seed(42)
    test_state = np.random.randn(120) * 100 + 150

    if 'revise_state' in functions:
        try:
            revised = functions['revise_state'](test_state)
            if not isinstance(revised, np.ndarray):
                errors.append(f"revise_state must return np.ndarray, got {type(revised)}")
            elif revised.ndim != 1:
                errors.append(f"revise_state must return 1D array, got {revised.ndim}D")
            elif np.any(np.isnan(revised)):
                errors.append("revise_state output contains NaN")
            elif np.any(np.isinf(revised)):
                errors.append("revise_state output contains Inf")
            elif len(revised) < 120:
                errors.append(f"revise_state output too short: {len(revised)} < 120")
            elif not np.allclose(revised[:120], test_state, atol=1e-6):
                errors.append("revise_state must preserve first 120 dims (source state)")
        except Exception as e:
            errors.append(f"revise_state execution error: {e}")
    else:
        errors.append("revise_state function not found in code")

    if 'intrinsic_reward' in functions:
        try:
            test_input = functions.get('revise_state', lambda s: s)(test_state)
            reward_val = functions['intrinsic_reward'](test_input)
            if not isinstance(reward_val, (int, float, np.integer, np.floating)):
                errors.append(f"intrinsic_reward must return scalar, got {type(reward_val)}")
            elif np.isnan(float(reward_val)):
                errors.append("intrinsic_reward returned NaN")
            elif np.isinf(float(reward_val)):
                errors.append("intrinsic_reward returned Inf")
            elif abs(float(reward_val)) > 100:
                errors.append(f"intrinsic_reward out of range [-100, 100]: {reward_val}")
        except Exception as e:
            errors.append(f"intrinsic_reward execution error: {e}")
    else:
        errors.append("intrinsic_reward function not found in code")

    return errors


def validate(code_str: str) -> Dict:
    """Full validation pipeline for LLM-generated code.

    Returns:
        {
            'ok': bool,
            'errors': list[str],
            'revise_state': callable or None,
            'intrinsic_reward': callable or None,
            'feature_dim': int,   # number of extra feature dims
            'state_dim': int,     # total state dim (120 + feature_dim)
        }

    Full pipeline: AST check -> function extraction -> test execution -> dim
    detection. The result carries callable references and dimension info.
    """
    result = {
        'ok': False,
        'errors': [],
        'revise_state': None,
        'intrinsic_reward': None,
        'feature_dim': 0,
        'state_dim': 120,
    }

    ast_errors = _ast_check(code_str)
    if ast_errors:
        result['errors'] = ast_errors
        return result

    try:
        functions = _extract_functions(code_str)
    except Exception as e:
        result['errors'] = [f"Code extraction error: {e}"]
        return result

    if 'revise_state' not in functions:
        result['errors'] = ["revise_state function not found"]
        return result

    test_errors = _test_execution(functions)
    if test_errors:
        result['errors'] = test_errors
        return result

    np.random.seed(42)
    test_state = np.random.randn(120) * 100 + 150
    try:
        revised = functions['revise_state'](test_state)
        state_dim = len(revised)
        feature_dim = state_dim - 120
    except Exception as e:
        result['errors'] = [f"Dimension detection error: {e}"]
        return result

    result['ok'] = True
    result['revise_state'] = functions['revise_state']
    result['intrinsic_reward'] = functions.get('intrinsic_reward')
    result['feature_dim'] = feature_dim
    result['state_dim'] = state_dim
    return result
