#!/usr/bin/env python3
"""DiffMIC yml config 의 traindata/testdata 를 갱신해 다시 저장.

DiffMIC main.py 는 모드에 따라 다른 YAML loader 를 쓰므로 출력 형식이 중요:
  - train 모드 (safe_load + dict2namespace) → plain dict YAML 이 맞음
  - test/sample 모드 (unsafe_load 후 attribute 접근) → Namespace 덤프 YAML 이 맞음

--format auto (기본): 입력 파일 형식을 따라감
--format dict     : 항상 plain dict 로 저장 (train 용)
--format namespace: 항상 Namespace 덤프 로 저장 (test 용; 학습 후 log config 갱신)
"""
import argparse, yaml, os

def to_plain(obj):
    if isinstance(obj, argparse.Namespace):
        return {k: to_plain(v) for k, v in vars(obj).items()}
    if isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(x) for x in obj]
    return obj

def to_namespace(obj):
    if isinstance(obj, dict):
        ns = argparse.Namespace()
        for k, v in obj.items():
            setattr(ns, k, to_namespace(v))
        return ns
    if isinstance(obj, list):
        return [to_namespace(x) for x in obj]
    return obj

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="src", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--traindata", default=None)
ap.add_argument("--testdata", default=None)
ap.add_argument("--format", choices=["auto", "dict", "namespace"], default="auto",
                help="출력 형식 (기본 auto: 입력 형식 따라감)")
a = ap.parse_args()

with open(a.src) as f:
    raw = f.read()

is_namespace_input = "python/object:argparse.Namespace" in raw
loader = yaml.unsafe_load if is_namespace_input else yaml.safe_load
cfg_plain = to_plain(loader(raw))

if not isinstance(cfg_plain, dict):
    raise SystemExit(f"unexpected config root type: {type(cfg_plain)}")

cfg_plain.setdefault("data", {})
if a.traindata: cfg_plain["data"]["traindata"] = a.traindata
if a.testdata:  cfg_plain["data"]["testdata"]  = a.testdata
cfg_plain.pop("tb_logger", None)
cfg_plain.pop("device", None)

if a.format == "auto":
    fmt = "namespace" if is_namespace_input else "dict"
else:
    fmt = a.format

os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
with open(a.out, "w") as f:
    if fmt == "namespace":
        yaml.dump(to_namespace(cfg_plain), f, default_flow_style=False)
    else:
        yaml.safe_dump(cfg_plain, f, default_flow_style=False)
print(f"wrote: {a.out}  (format: {fmt})")
