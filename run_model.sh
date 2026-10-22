#!/usr/bin/env bash
#
# run_model.sh —— 评测一个新模型的统一入口。
#
# 用法：
#   ./run_model.sh check  <模型名>              # 预检：模型存在？收图？流式？C 档数据？
#   ./run_model.sh a      <模型名> [bench...]   # 跑 A 档（默认 7 个）
#   ./run_model.sh b      <模型名> [bench...]   # 跑 B 档（默认 4 个）+ 补跑
#   ./run_model.sh c      <模型名> [bench...]   # 跑 C 档（默认 8 个）
#   ./run_model.sh all    <模型名>              # A + B
#   ./run_model.sh allc   <模型名>              # A + B + C
#   ./run_model.sh retry  <模型名> [bench...]   # 补跑失败题并合并
#   ./run_model.sh emptyfix <模型名> [bench...] # 补跑空回复。⚠️ 先看 token_counts：多数空回复是推理撞上限，
#                                              重新请求等于只给这个模型重掷骰子，不可比（见 run_a_tier_api.py）
#   ./run_model.sh judge  <模型名>              # RoboBench 规划题 LLM judge（b 跑完后必须执行）
#   ./run_model.sh status <模型名>              # 看进度
#   ./run_model.sh prefetch [bench...]          # 预取 C 档数据（不需模型名）
#   ./run_model.sh attr   <模型名>              # 只补 #4 客体视觉属性（visual_attribute 31 题）
#
# 示例：
#   ./run_model.sh all qwen/qwen3.8-flash
#   ./run_model.sh a   qwen/qwen3.8-27b causalphys physbench
#   ./run_model.sh retry qwen/qwen3.8-flash
#   ./run_model.sh prefetch epic phyx           # 只预取两个
#   LIMIT=20 FOREGROUND=1 ./run_model.sh c qwen/qwen3.8-flash omnispatial
#
# 参数都会传给底层的 run_a_tier_api.py；下面这几个属于本脚本：
#   CONC=16          并发数（默认 16）
#   MAX_MB=6         请求体超过多少 MB 改用 JPEG（默认 6，0 = 关闭）
#   LIMIT=20         只跑前 N 题（调试用，默认不限制）
#   FOREGROUND=1     前台运行（默认后台 + 输出到 /tmp/run_<后缀>.log）
#   BASE_URL=...     换服务商时指定端点（默认用 .env 的 EVAL_ROUTER_BASE_URL）
#   API_KEY=...      换服务商时指定 key（默认用 .env 的 EVAL_ROUTER_API_KEY）
#   SUFFIX=...       手动指定结果目录后缀（默认由模型名生成）
#   SKIP_DATA=1      check 时跳过第 4 项（C 档数据离线校验，约 25 秒）
#
# 换服务商（三个一起给，否则会和 .env 混用）：
#   BASE_URL=https://api.openai.com/v1 API_KEY=sk-xxx \
#     ./run_model.sh all gpt-4o
#   同一模型名在不同服务商各跑一份时，用 SUFFIX 区分结果目录：
#   SUFFIX=gpt4o_official ./run_model.sh all gpt-4o
#
# 两条必须记住的：
#   1) 流式（stream=true）是必需的：网关空闲 60s 就 504，thinking 模型会被截断
#   2) 失败题不会自动重来，跑完要 retry

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null || PY="python3"

A_BENCH=(causalphys physbench vsibench momagraph negvqa refspatial egotaskqa eriq)
B_BENCH=(robobench robobench_plan mvrobobench pio asimov eai mmupd aqua openeqa intphys2 corecognition tvbench_cap13 vlm4d_cap37 spatialuncertain)
# C 档：这 8 个名字与 run_a_tier_api.py 的 BENCHMARKS 键一致（跨档唯一）。
# 注意与 lmms-eval 的 --tasks 取值拼写不同：
#   omnispatial       <-> omni_spatial
#   robospatial       <-> robo_spatial
#   crosspoint        <-> crosspoint_bench
#   phyx              <-> phyx_mc
#   epic              <-> epic_bench
# 结果目录两边是一样的（eval_C_<bench>`_sampled` 对 `_<模型后缀>`），所以能直接并排比。
C_BENCH=(crosspoint multihopspatial omnispatial robospatial sparbench embspatial phyx epic epic_orient)

CONC="${CONC:-16}"
MAX_MB="${MAX_MB:-6}"
LIMIT="${LIMIT:-}"
FOREGROUND="${FOREGROUND:-0}"
BASE_URL="${BASE_URL:-}"
API_KEY="${API_KEY:-}"
SUFFIX_OVERRIDE="${SUFFIX:-}"

die() { echo "错误：$*" >&2; exit 1; }

suffix_of() { "$PY" -c "import re,sys;print(re.sub(r'[^a-zA-Z0-9]','_',sys.argv[1].split('/')[-1]).strip('_'))" "$1"; }

# 从 .env 读一个键（解析方式和 run_a_tier_api.py 一致）
env_get() {
  "$PY" - "$1" <<'PYEOF'
import sys
key = sys.argv[1]
try:
    for line in open(".env"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        if k.strip() == key:
            print(v.strip().strip('"').strip("'"))
except FileNotFoundError:
    pass
PYEOF
}

# ---------------------------------------------------------------- 参数解析

CMD="${1:-}"; shift || true

# prefetch 跟模型无关（只拉 C 档数据），在参数解析前就分发掉，免得被下面的
# 「没给模型名」拦下，也免得为一个不需要模型的动作去校验 .env 里的凭据。
if [ "$CMD" = "prefetch" ]; then
  exec "$PY" "$HERE/prefetch_c_tier.py" "$@"
fi

[ -f .env ] || die "找不到 .env（需要 EVAL_ROUTER_BASE_URL / EVAL_ROUTER_API_KEY）"

# 位置参数：模型名可省略（PROFILE 提供了 <PROFILE>_MODEL 时）。
# 首个位置参数若正好是已知 benchmark 名，就按 bench 解释，不当模型名 —— 否则
# `a causalphys` 会把 causalphys 当成模型，并且回退去跑全部 benchmark。
is_bench() { local b; for b in "${A_BENCH[@]}" "${B_BENCH[@]}" "${C_BENCH[@]}"; do [ "$1" = "$b" ] && return 0; done; return 1; }

MODEL=""
if [ -n "${1:-}" ] && ! is_bench "$1"; then
  MODEL="$1"; shift || true
fi
BENCHES=("$@")

# 服务商凭据的来源，优先级：命令行 BASE_URL/API_KEY > PROFILE（.env 里的 <前缀>_*）> .env 默认
#   PROFILE=OPENAI ./run_model.sh all gpt-4o     # 读 .env 的 OPENAI_BASE_URL / OPENAI_API_KEY[/OPENAI_MODEL]
CRED_ARGS=()
CRED_SOURCE=""
if [ -n "${BASE_URL:-}" ] || [ -n "${API_KEY:-}" ]; then
  [ -n "${BASE_URL:-}" ] && [ -n "${API_KEY:-}" ] || die "BASE_URL 和 API_KEY 要一起给（只给一个会和 .env 里的配成一对）"
  CRED_ARGS=(--base-url "$BASE_URL" --api-key "$API_KEY")
  CRED_SOURCE="命令行"
elif [ -n "${PROFILE:-}" ]; then
  _pb="$(env_get "${PROFILE}_BASE_URL")"
  _pk="$(env_get "${PROFILE}_API_KEY")"
  _pm="$(env_get "${PROFILE}_MODEL")"
  [ -n "$_pb" ] && [ -n "$_pk" ] || die "PROFILE=$PROFILE，但 .env 里缺少 ${PROFILE}_BASE_URL 或 ${PROFILE}_API_KEY"
  CRED_ARGS=(--base-url "$_pb" --api-key "$_pk")
  CRED_SOURCE="profile $PROFILE"
  [ -n "$MODEL" ] || MODEL="$_pm"
fi
# 统一成 BASE_URL / API_KEY，供 do_check 里的 export 使用
if [ ${#CRED_ARGS[@]} -gt 0 ]; then
  BASE_URL="${CRED_ARGS[1]}"
  API_KEY="${CRED_ARGS[3]}"
fi

[ -n "$CMD" ] || die "用法：$0 {check|a|b|c|all|allc|retry|judge|status} [模型名] [bench...]
       $0 prefetch [bench...]          # 与模型无关，单独用法
     模型名可省略（PROFILE 提供 <PROFILE>_MODEL 时），例如：
       PROFILE=SEED $0 check
       PROFILE=SEED LIMIT=20 $0 a causalphys
     换服务商也可直接给：BASE_URL=... API_KEY=... $0 all gpt-4o"

# 未知命令在这里就拦下。否则它会被当成「模型名」吞掉，最后报成「没给模型名」，
# 排查方向完全被带偏（prefetch 已在上面的 CMD 分支单独处理）。
case "$CMD" in
  check|a|b|c|all|allc|retry|emptyfix|judge|status|attr) ;;
  *) die "未知命令：$CMD（可用：check a b c all allc retry emptyfix judge status attr prefetch）" ;;
esac

[ -n "$MODEL" ] || die "没给模型名，且 .env 里没有 ${PROFILE:-<PROFILE>}_MODEL。请补上模型名：
       $0 $CMD <模型名> ${BENCHES[*]:-}"

# 固定版本：浮动名今后可能被服务商改指向新版本（2026-10-17 OpenRouter 上线了
# deepseek-v4.1-flash-1017，浮动名当时仍指 0910）。换成规范名（canonical slug）钉死；
# 后缀沿用浮动名推出的那个，结果继续落在已有目录里。新增条目前先实测规范名能调用。
case "$MODEL" in
  deepseek/deepseek-v4.1-flash)
    MODEL="deepseek/deepseek-v4.1-flash-20260910"
    SUFFIX="${SUFFIX:-deepseek_v4_1_flash}" ;;
esac

SUFFIX="${SUFFIX:-$(suffix_of "$MODEL")}"
LOG="/tmp/run_${SUFFIX}.log"

# 模型名里的 / 会让 logs/dirs 变成子目录，提前拦下
case "$MODEL" in *..*) die "模型名含 ..，拒绝执行" ;; esac

# ---------------------------------------------------------------- 预检

do_check() {
  echo "=== 预检 $MODEL ==="
  echo "--- 1) 模型是否可用"
  set -a; . ./.env; set +a
  # 换服务商时用命令行给的端点/凭据，并导出给下面两段 Python 用
  if [ ${#CRED_ARGS[@]} -gt 0 ]; then
    export EVAL_ROUTER_BASE_URL="$BASE_URL"
    export EVAL_ROUTER_API_KEY="$API_KEY"
    echo "  （使用命令行指定的端点 $BASE_URL）"
  fi
  local code
  : > /tmp/_chk.json   # 清空，避免 curl 失败时读到上一次的残留内容
  code=$(curl -s -o /tmp/_chk.json -w "%{http_code}" --max-time 60 \
    -X POST "${EVAL_ROUTER_BASE_URL%/}/chat/completions" \
    -H "Authorization: Bearer $EVAL_ROUTER_API_KEY" -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":4}")
  if [ "$code" = "000" ]; then
    echo "  ✗ 连不上 ${EVAL_ROUTER_BASE_URL}（HTTP 000：DNS/网络不通，或端点写错）"
    return 1
  fi
  if [ "$code" != "200" ]; then
    echo "  ✗ HTTP $code  $(head -c 160 /tmp/_chk.json)"
    if [ ${#CRED_ARGS[@]} -gt 0 ]; then
      echo "  提示：换服务商后模型名要按那家的命名写，别照搬路由上的 id"
    else
      echo "  提示：用下面这条看路由上有哪些模型"
      echo "    curl -s \"\$EVAL_ROUTER_BASE_URL/models\" -H \"Authorization: Bearer \$EVAL_ROUTER_API_KEY\" | python3 -m json.tool | grep '\"id\"'"
    fi
    return 1
  fi
  echo "  ✓ 可用"

  echo "--- 2) 是否收图"
  MODEL="$MODEL" "$PY" - <<'PYEOF'
import os, io, json, base64, urllib.request, urllib.error
from PIL import Image, ImageDraw
base=os.environ["EVAL_ROUTER_BASE_URL"].rstrip("/"); key=os.environ["EVAL_ROUTER_API_KEY"]
m=os.environ["MODEL"]
im=Image.new("RGB",(224,224),(255,255,255)); d=ImageDraw.Draw(im)
d.ellipse((40,40,184,184), fill=(220,30,30))
b=io.BytesIO(); im.save(b, format="PNG")
body={"model":m,"max_tokens":512,"temperature":0,"messages":[{"role":"user","content":[
  {"type":"image_url","image_url":{"url":"data:image/png;base64,"+base64.b64encode(b.getvalue()).decode()}},
  {"type":"text","text":"What color is the shape? One word."}]}]}
req=urllib.request.Request(base+"/chat/completions", data=json.dumps(body).encode(),
    headers={"Authorization":f"Bearer {key}","Content-Type":"application/json"})
try:
    with urllib.request.urlopen(req, timeout=300) as r: dd=json.loads(r.read())
    ans=(dd["choices"][0]["message"].get("content") or "").strip()
    print(f"  {'✓' if 'red' in ans.lower() else '✗'} 答: {ans[:40]!r}")
except urllib.error.HTTPError as e:
    print(f"  ✗ HTTP{e.code} {e.read()[:120].decode(errors='replace')}")
except Exception as e:
    print(f"  ✗ {type(e).__name__}: {str(e)[:100]}")
PYEOF

  echo "--- 3) 流式兼容（客户端固定带 stream + include_usage）"
  MODEL="$MODEL" "$PY" - <<'PYEOF'
import os
from openai import OpenAI
c=OpenAI(api_key=os.environ["EVAL_ROUTER_API_KEY"], base_url=os.environ["EVAL_ROUTER_BASE_URL"], timeout=600, max_retries=0)
m=os.environ["MODEL"]
for opts in (True, False):
    try:
        kw={"stream_options":{"include_usage":True}} if opts else {}
        s=c.chat.completions.create(model=m, messages=[{"role":"user","content":"Say OK."}], max_tokens=64, stream=True, **kw)
        txt=[]; u=None
        for ch in s:
            if ch.choices and ch.choices[0].delta and ch.choices[0].delta.content: txt.append(ch.choices[0].delta.content)
            if ch.usage: u=ch.usage
        print(f"  {'✓' if opts else ' '} stream_options={str(opts):5s} OK  usage={'有' if u else '无'}")
    except Exception as e:
        print(f"  ✗ stream_options={str(opts):5s} {type(e).__name__}: {str(e)[:80]}")
print("  注：stream_options 那条失败就不能直接用，需先给 models/chat/openai.py 加开关")
PYEOF

  # 4) C 档数据。A/B 档的媒体已本地化，但 C 档是运行时从 HF 拉的，而本机连不上
  # huggingface.co。没预取的话不会报错，会在 HEAD 重试上磨很久，最后 exit 0，
  # 分数被静默污染 —— 所以值得在长跑前花这 25 秒。
  if [ "${SKIP_DATA:-0}" = "1" ]; then
    echo "--- 4) C 档数据（已按 SKIP_DATA=1 跳过）"
    return 0
  fi
  echo "--- 4) C 档数据是否离线可用（约 25 秒，SKIP_DATA=1 可跳过）"
  if "$PY" prefetch_c_tier.py --verify-only > /tmp/_cdata.txt 2>&1; then
    echo "  ✓ 8 个数据集 + CrossPoint 图片树均可离线加载"
  else
    echo "  ✗ 有 benchmark 离线不可用，跑 C 档前先执行：./run_model.sh prefetch"
    sed -n 's/^ *FAIL/    /p' /tmp/_cdata.txt | head -8
    return 1
  fi
}

# ---------------------------------------------------------------- 评测

run_bench() {  # run_bench <bench...>
  local extra=()
  [ -n "$LIMIT" ] && extra+=(--limit "$LIMIT")
  # 必须显式传后缀：run_a_tier_api.py 会从模型名自己推导后缀，SUFFIX=xxx 不传下去
  # 就静默失效，结果会和默认后缀的旧目录混在一起（2026-10-02 踩到过）。
  extra+=(--output-dir-suffix "$SUFFIX")
  echo
  echo "############ $MODEL"
  echo "######## 后缀=$SUFFIX  并发=$CONC  上限=${MAX_MB}MB  任务=$*${LIMIT:+  每题限制=$LIMIT 题}"
  echo "######## 日志=$LOG"
  echo
  if [ "$FOREGROUND" = "1" ]; then
    LMMS_MAX_REQUEST_MB="$MAX_MB" "$PY" run_a_tier_api.py \
      --model "$MODEL" "${CRED_ARGS[@]}" --num-concurrent "$CONC" "${extra[@]}" --tasks "$@"
  else
    LMMS_MAX_REQUEST_MB="$MAX_MB" nohup "$PY" run_a_tier_api.py \
      --model "$MODEL" "${CRED_ARGS[@]}" --num-concurrent "$CONC" "${extra[@]}" --tasks "$@" > "$LOG" 2>&1 &
    echo "已在后台启动 (pid $!)。跟踪进度："
    echo "  $0 status $MODEL"
    echo "  tail -f $LOG"
  fi
}

do_retry() {  # do_retry <bench...>
  echo "=== 补跑 $MODEL 的失败题 (bench: ${*:-全部}) ==="
  local benches=("$@")
  [ ${#benches[@]} -eq 0 ] && benches=("${A_BENCH[@]}" "${B_BENCH[@]}" "${C_BENCH[@]}")
  # 必须显式传后缀：run_a_tier_api.py 会从模型名自己推导，SUFFIX=xxx 不传下去
  # 就静默失效，导致补跑去读 eval_*_<推导后缀>/ 而找不到本次结果（2026-10-04 踩到过）。
  LMMS_MAX_REQUEST_MB="$MAX_MB" "$PY" run_a_tier_api.py \
    --model "$MODEL" "${CRED_ARGS[@]}" --num-concurrent "$CONC" --retry-failed \
    --output-dir-suffix "$SUFFIX" \
    --tasks "${benches[@]}" || true

  echo
  echo "=== 合并回原结果 ==="
  for b in "${benches[@]}"; do
    local d
    d="$(ls -d "eval_A_${b}_${SUFFIX}" "eval_B_${b}_${SUFFIX}" "eval_C_${b}_${SUFFIX}" 2>/dev/null | head -1)"
    [ -n "$d" ] || continue
    [ -d "${d}_retry" ] || { echo "  $d: 无补跑结果，跳过"; continue; }
    "$PY" merge_retry.py "$d"
  done
  echo
  echo "合并后的结果在 eval_*_${SUFFIX}_merged/，原目录未改动。"
}

do_judge() {
  # 有补跑就判合并后的目录（merge_retry 产出的是未判分的原始样本，合并后要重判）
  # 额外参数是**任务名**（不是 bench 名），只判这些任务：
  #   ./run_model.sh judge <模型> robobench_implicit_goal_q1
  # 不给就判目录里全部开放式任务。judge 默认 OpenRouter 的 gpt-4o，见 judge 脚本头部。
  local d="eval_B_robobench_plan_${SUFFIX}_merged"
  [ -d "$d" ] || d="eval_B_robobench_plan_${SUFFIX}"
  [ -d "$d" ] || die "找不到 RoboBench 规划题结果目录（先跑：$0 b $MODEL robobench_plan）"
  echo "=== LLM judge: $d ${BENCHES[*]:-（全部任务）} ==="
  local task_args=()
  [ ${#BENCHES[@]} -gt 0 ] && task_args=(--tasks "${BENCHES[@]}")
  "$PY" judge_robobench_openended.py --eval-dir "$d" --out "${d}_judge.json" "${task_args[@]}"
}

# ---------------------------------------------------------------- 进度

do_status() {
  echo "=== $MODEL (后缀 $SUFFIX) ==="
  local any=0
  for b in "${A_BENCH[@]}" "${B_BENCH[@]}" "${C_BENCH[@]}"; do
    for d in "eval_A_${b}_${SUFFIX}" "eval_B_${b}_${SUFFIX}" "eval_C_${b}_${SUFFIX}"; do
      [ -d "$d" ] || continue
      any=1
      local n
      n="$(find "$d" -name '*_samples_*.jsonl' 2>/dev/null | wc -l)"
      local f
      f="$(grep -rl 'LMMS_EVAL_REQUEST_FAILED' "$d" --include='*_samples_*.jsonl' 2>/dev/null | wc -l)"
      local res
      res="$(find "$d" -name '*results.json' 2>/dev/null | wc -l)"
      printf '  %-34s 样本文件=%-3s 含失败=%-3s results=%s\n' "$(basename "$d")" "$n" "$f" "$res"
    done
  done
  [ "$any" = "1" ] || { echo "  还没有结果目录"; return; }
  echo
  echo "运行中的进程："
  pgrep -af "lmms_eval eval" | grep -F "$MODEL" | sed 's/^/  /' || echo "  （无）"
  [ -f "$LOG" ] && { echo; echo "驱动日志尾部 ($LOG)："; tail -3 "$LOG" | sed 's/^/  /'; }
}

# ---------------------------------------------------------------- 分发

# 注意：这里**不能**写成 `[ ${#BENCHES[@]} -gt 0 ] && run_bench "$@" || run_bench "全部"`。
# bash 的 `A && B || C` 里，B 一旦返回非 0，就会转去执行 C —— 于是一个 benchmark
# 失败（或被 Ctrl-C / kill 掉）会**静默触发整档重跑**。2026-10-11 踩到过：epic 的
# 判分器卡死被 TERM 后，驱动立刻把 C 档 8 项全跑了一遍，白烧算力，且再撞同一个 epic。
# 用 if 明确表达「给定了就只跑给定的」。
run_tier() {  # run_tier <档位> <默认 benchmark...>
  local tier="$1"; shift
  if [ ${#BENCHES[@]} -gt 0 ]; then
    run_bench "${BENCHES[@]}"
  else
    run_bench "$@"
  fi
}

case "$CMD" in
  check)  do_check ;;
  attr)
    # 不带 bench 参数时默认只跑 robobench_attr（run_tier 的第二个参数是默认值）
    if [ ${#BENCHES[@]} -gt 0 ]; then run_bench "${BENCHES[@]}"; else run_bench robobench_attr; fi ;;
   a)      run_tier a "${A_BENCH[@]}" ;;
  b)      run_tier b "${B_BENCH[@]}" ;;
  c)      run_tier c "${C_BENCH[@]}" ;;
  all)    run_bench "${A_BENCH[@]}" "${B_BENCH[@]}" ;;
  allc)   run_bench "${A_BENCH[@]}" "${B_BENCH[@]}" "${C_BENCH[@]}" ;;
  retry)  do_retry "${BENCHES[@]}" ;;
  emptyfix)
    # 只请求报表所读目录里的空回复/失败题，合并回 _merged（旧 _merged 先备份）。前台执行。
    LMMS_MAX_REQUEST_MB="$MAX_MB" "$PY" run_a_tier_api.py \
      --model "$MODEL" "${CRED_ARGS[@]}" --num-concurrent "$CONC" --retry-empty \
      --output-dir-suffix "$SUFFIX" --tasks "${BENCHES[@]}" ;;
  judge)  do_judge ;;
  status) do_status ;;
  *)      die "未知命令：$CMD（可用：check a b c all allc retry judge status prefetch）" ;;
esac
