import tempfile, subprocess, textwrap, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd
import ast

# TODO: generalize to any function (not just `build_feature`)
# TODO: run in a safer (docker?) environment?
def run_llm_code(
        code: str,
        input_df: pd.DataFrame,
        timeout: int = 30
    ) -> np.ndarray:
    """
    Execute LLM-generated code that defines:
        def build_feature(df) -> array-like
    Returns: np.ndarray (float64)
    Raises RuntimeError on failure/timeout.
    """
    # 0) quick static sanity check
    try:
        ast.parse(code)
    except SyntaxError as e:
        raise RuntimeError(f"syntax error: {e}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        cand_path   = tmp / "cand.py"
        data_path   = tmp / "data.csv"
        helper_path = tmp / "runner.py"

        # 1) write candidate code (always prepend safe imports)
        cand_code = "import pandas as pd\nimport numpy as np\n" + code
        cand_path.write_text(cand_code)

        # 2) write input df (CSV to avoid extra deps)
        input_df.reset_index(drop=True).to_csv(data_path,index=False)

        # 3) helper script uses absolute paths
        helper = textwrap.dedent(f"""
        import pandas as pd, numpy as np, importlib.util, json
        from pathlib import Path

        CAND = Path(r"{str(cand_path)}")
        DATA = Path(r"{str(data_path)}")

        df = pd.read_csv(DATA)
        spec = importlib.util.spec_from_file_location("cand", CAND)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        feat = mod.build_feature(df)
        arr  = np.asarray(feat, dtype=float)
        print(json.dumps(arr.tolist()))
        """)
        helper_path.write_text(helper)

        # 4) run in a clean working dir (tmp) with timeout
        try:
            res = subprocess.run(
                [sys.executable, str(helper_path)],
                cwd=tmp,                # <<< important
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                env={k: v for k, v in os.environ.items() if k not in {"HTTP_PROXY","HTTPS_PROXY","ALL_PROXY","NO_PROXY"}}
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("timeout")

        if res.returncode != 0:
            err = res.stderr.strip() or "unknown error"
            raise RuntimeError(err)

        try:
            return np.array(json.loads(res.stdout), dtype=float)
        except Exception as e:
            raise RuntimeError(f"bad runner output: {e}")
