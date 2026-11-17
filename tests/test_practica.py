"""The snippets in practica/ are the ones typed in the practice video: each must run and print
what the video shows (03_softmax_roto.py must print the nan that the video explains)."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

PRACTICA = Path(__file__).resolve().parent.parent / "practica"


def run(name: str) -> str:
    cmd = [sys.executable, str(PRACTICA / name)]
    r = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
    return r.stdout


def nums(line: str) -> list[float]:
    return [float(m) for m in re.findall(r"-?\d+\.?\d*(?:e[-+]\d+)?", line)]


def test_01_ajustar_finds_the_hidden_formula() -> None:
    lines = run("01_ajustar.py").splitlines()
    assert nums(lines[0])[1] == pytest.approx(17668.14, rel=1e-3)
    assert nums(lines[-2])[1] == pytest.approx(1.07, abs=0.02)  # the noise has variance 1
    w_cpu, w_ram, b = nums(lines[-1])
    assert w_cpu == pytest.approx(30.0, abs=0.2)
    assert w_ram == pytest.approx(-3.9, abs=0.2)  # more RAM, less latency: the sign is learned
    assert b == pytest.approx(119.9, abs=0.2)


def test_02_optimizadores_adam_wins_in_the_canyon() -> None:
    lines = run("02_optimizadores.py").splitlines()
    err = {line.split()[0]: float(line.split()[2]) for line in lines}
    assert list(err) == ["sgd", "momentum", "adam"]
    assert err["sgd"] > err["momentum"] > err["adam"]
    assert nums(lines[0])[-1] == pytest.approx(0.1469, abs=1e-3)  # SGD still far along w[1]


def test_03_softmax_roto_prints_nan() -> None:
    out = run("03_softmax_roto.py")
    assert "inf]" in out.splitlines()[0]
    assert "nuestra: [ 0.  0. nan]" in out
    assert "de JAX:  [0. 0. 1.]" in out


def test_03_softmax_matches_jax() -> None:
    lines = run("03_softmax.py").splitlines()
    assert lines[0] == "nuestra: [-110. -100.    0.]"
    assert lines[1] == "de JAX:  [-110. -100.    0.]"
    assert lines[2] == "probabilidades: [0. 0. 1.]"


def test_04_red_separates_the_rings() -> None:
    assert nums(run("04_red.py"))[0] >= 99.5


def test_05_weight_decay_closes_the_gap() -> None:
    lines = run("05_weight_decay.py").splitlines()
    assert [line.split()[0] for line in lines] == ["wd=0.0", "wd=1.0"]
    _, train0, test0, norm0 = nums(lines[0])
    _, train1, test1, norm1 = nums(lines[1])
    assert train0 < 0.1 and test0 == pytest.approx(16.28, rel=0.02)  # memorized
    assert test1 == pytest.approx(2.13, rel=0.02) and norm1 < norm0 / 2
