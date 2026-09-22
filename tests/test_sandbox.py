from isaac_eval.sandbox import run_code, run_with_retry

CFG = {"sandbox": {"mode": "subprocess", "timeout_s": 5, "memory_mb": 512, "max_attempts": 3, "network": False}}
DATASET = "data/datasets/jeju_phytoplankton.xlsx"
SHEET = "식.플 정량분석"


def test_last_line_json_parsed():
    code = (
        "import json\n"
        "result = {'ok': 1}\n"
        "print(json.dumps(result, ensure_ascii=False))\n"
    )
    r = run_code(code, DATASET, SHEET, CFG)
    assert r.ok is True
    assert r.value == {"ok": 1}


def test_dataset_loaded_directly_by_child():
    code = (
        "import json\n"
        "result = {'n_rows': len(df)}\n"
        "print(json.dumps(result, ensure_ascii=False))\n"
    )
    r = run_code(code, DATASET, SHEET, CFG)
    assert r.ok is True
    assert r.value["n_rows"] == 8


def test_timeout_enforced():
    code = "import time\ntime.sleep(30)\n"
    cfg = {"sandbox": {**CFG["sandbox"], "timeout_s": 1}}
    r = run_code(code, DATASET, SHEET, cfg)
    assert r.ok is False
    assert "timeout" in (r.error or "").lower()


def test_network_blocked():
    code = (
        "import socket, json\n"
        "try:\n"
        "    socket.create_connection(('8.8.8.8', 53), timeout=2)\n"
        "    result = {'network': True}\n"
        "except Exception:\n"
        "    result = {'network': False}\n"
        "print(json.dumps(result))\n"
    )
    r = run_code(code, DATASET, SHEET, CFG)
    assert r.ok is True
    assert r.value["network"] is False


def test_retry_feeds_traceback():
    calls = []

    def gen(prev_error):
        calls.append(prev_error)
        if prev_error is None:
            return "raise KeyError('boom')\n"
        return "import json\nprint(json.dumps({'fixed': True}))\n"

    results = run_with_retry(gen, DATASET, SHEET, CFG)
    assert results[-1].ok is True
    assert results[0].ok is False
    assert "boom" in results[0].traceback
    assert calls[1] is not None and "boom" in calls[1]
