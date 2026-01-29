import numpy as np
import time
import ast, json, os, sys, subprocess, tempfile, textwrap
from pathlib import Path
import numpy as np
from core.evolve import evolve

def run_code(
    code: str,
    func_name: str,
    args=None,
    kwargs=None,
    timeout: int = 30,
):
    """
    Execute arbitrary LLM-generated code in isolation, calling a specific function.
    
    Parameters
    ----------
    code : str
        Python source code defining at least the function `func_name`.

    func_name : str
        Name of the function inside `code` to call.

    args : list or tuple
        Positional arguments to pass to the function.

    kwargs : dict
        Keyword arguments to pass to the function.

    timeout : int
        Execution timeout in seconds.

    Returns
    -------
    Python object 
        Whatever the function returns (scalar / list / dict / array / etc).
    """

    args = [] if args is None else list(args)
    kwargs = {} if kwargs is None else dict(kwargs)

    # Basic static syntax check
    try:
        ast.parse(code)
    except SyntaxError as e:
        raise RuntimeError(f"syntax error: {e}")

    # Serialize args/kwargs to JSON for the helper process
    encoded_args = json.dumps(args)
    encoded_kwargs = json.dumps(kwargs)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        cand_path = tmp / "cand.py"
        helper_path = tmp / "runner.py"

        # Prepend minimal safe imports
        header = "import numpy as np\nimport pandas as pd\n"
        cand_path.write_text(header + code)

        # Helper script
        helper = textwrap.dedent(f"""
        import json
        import numpy as np
        import pandas as pd
        import importlib.util
        from pathlib import Path
        import time, copy, random

        # Load candidate module
        CAND = Path(r"{cand_path}")
        spec = importlib.util.spec_from_file_location("cand", CAND)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # Load arguments
        args = json.loads({json.dumps(encoded_args)})
        kwargs = json.loads({json.dumps(encoded_kwargs)})

        # Call the target function
        func = getattr(mod, "{func_name}", None)
        if func is None:
            raise RuntimeError("function '{func_name}' not found")

        def f(x):
            return np.exp(np.sin(x) + np.square(np.cos(x)))

        WARMUP_RUNS=3
        TIMED_RUNS=10    
        
        size = 5_000_000
        x = np.random.random(size).astype(np.float64)

        times = []
        for i in range(WARMUP_RUNS + TIMED_RUNS):
            x_in = x.copy()

            # Don't time the warm-up runs
            if i < WARMUP_RUNS:
                result = func(x_in)
            # Start timing after warm-up
            else:
                start = time.perf_counter()
                result = func(x_in)
                times.append(time.perf_counter() - start)
            if not isinstance(result, np.ndarray) or result.shape != x.shape or not np.allclose(result, f(x)):
                result = np.nan
                break

        # Convert numpy arrays to lists for JSON transport
        if isinstance(result, np.ndarray):
            result = result.tolist()

        out = result if np.isnan(result).any() else -np.median(times)

        print(json.dumps(out))
        """)
        helper_path.write_text(helper)

        # Execute helper
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in {"HTTP_PROXY","HTTPS_PROXY","ALL_PROXY","NO_PROXY"}
        }

        try:
            res = subprocess.run(
                [sys.executable, str(helper_path)],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                env=env,
            )

        except subprocess.TimeoutExpired:
            raise RuntimeError("timeout")

        if res.returncode != 0:
            err = res.stderr.strip() or "unknown error"
            raise RuntimeError(err)

        # Parse returned JSON
        try:
            out = json.loads(res.stdout)
        except Exception as e:
            raise RuntimeError(f"failed to decode output: {e}")

        # Convert list back to numpy array if appropriate
        try:
            arr = np.array(out, dtype=float)
            # Heuristic: if conversion works and shape != (), return array
            if arr.shape != ():
                return arr
        except Exception:
            pass

        return out


def f(x):
    return np.exp(np.sin(x) + np.square(np.cos(x)))

def objective(code):
    try:
        time = run_code(code, "build_func")
        return time
    except Exception as e:
        print(e)
        print(code)
        return np.nan


if __name__ == "__main__":

    query = """
        You are improving a Python function that computes the expression:

            f(x) = exp(sin(x) + cos(x)**2)

        The input x will always be a 1-dimensional NumPy array of dtype float64. 
        Your function must return a NumPy array of the same shape, with the exact same values 
        as the naive implementation:

            np.exp(np.sin(x) + np.square(np.cos(x)))

        Your goal is to generate a version of this function that produces exactly identical 
        outputs (up to floating point precision of float64) while being significantly faster 
        on large arrays (length ~10,000,000).

        Constraints and requirements:

        1. The function must be named build_func and must take a single argument x.
        Example signature:
            def build_func(x):
                ...

        2. Your implementation must compute the mathematically correct value of 
        exp(sin(x) + cos(x)**2). The evaluator will test your output against the 
        naive implementation on multiple randomized arrays and will reject any 
        implementation with incorrect outputs.

        3. Your function must avoid unnecessary temporary arrays and should try to minimize:
        - number of passes over the data
        - number of calls to transcendental functions
        - intermediate allocations
        - Python loops or list comprehensions
        - shape broadcasts that allocate new arrays

        4. You may use NumPy operations only. No other external libraries.
        If you choose to use NumExpr, Numba, or similar packages, your code will be rejected.

        5. Your output must be valid Python code only. 
        Do not include any commentary, text, or explanation — only the function.

        You will be provided with an existing build_func implementation. 
        Your task is to return an improved function with the same signature and semantics, 
        but faster runtime.
        """

    seeds = ["def build_func(x):    return np.exp(np.sin(x) + np.square(np.cos(x)))"] * 20

    # run evolution
    results = evolve(
        query,
        objective,
        seeds,
        K=10,
        C=4,
        GENS=30,
        log_path="examples/func_opt/log.jsonl"
    )