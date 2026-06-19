#!/bin/bash
# ============================================================
# Smoke Test — 验证关键脚本可运行（语法检查 + 纯计算脚本执行）
# 运行：bash scripts/check.sh
# ============================================================

set -e
PASS=0
FAIL=0
SKIP=0

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m'

check_syntax() {
    local file="$1"
    if python -c "import py_compile; py_compile.compile('$file', doraise=True)" 2>/dev/null; then
        echo -e "  ${GREEN}✅ SYNTAX OK${NC}  $file"
        PASS=$((PASS + 1))
    else
        echo -e "  ${RED}❌ SYNTAX FAIL${NC}  $file"
        FAIL=$((FAIL + 1))
    fi
}

run_script() {
    local file="$1"
    local timeout_sec="${2:-30}"
    if timeout "$timeout_sec" python "$file" > /dev/null 2>&1; then
        echo -e "  ${GREEN}✅ RUN OK${NC}     $file"
        PASS=$((PASS + 1))
    else
        echo -e "  ${RED}❌ RUN FAIL${NC}   $file"
        FAIL=$((FAIL + 1))
    fi
}

skip_script() {
    local file="$1"
    local reason="$2"
    echo -e "  ${YELLOW}⏭️  SKIP${NC}      $file  ($reason)"
    SKIP=$((SKIP + 1))
}

echo "============================================================"
echo "  AI-LLM-Learning Smoke Test"
echo "============================================================"
echo ""

# ── 00-数学基础（纯 numpy，可直接运行） ──────────────────
echo "--- 00-数学基础 ---"
for f in 00-数学基础/*.py; do
    check_syntax "$f"
done
# 选几个纯计算脚本实际运行
run_script "00-数学基础/03-点积与余弦相似度.py"
run_script "00-数学基础/17-softmax.py"
run_script "00-数学基础/21-贝叶斯定理.py"

# ── 01-Transformer ──────────────────────────────────────
echo ""
echo "--- 01-Transformer ---"
for f in 01-Transformer/*.py 01-Transformer/self_attention/*.py; do
    check_syntax "$f"
done

# ── 02-LLM基础 ──────────────────────────────────────────
echo ""
echo "--- 02-LLM基础 ---"
for f in 02-LLM基础/*.py; do
    check_syntax "$f"
done
run_script "02-LLM基础/02-采样策略.py"

# ── 03-Prompt工程 ──────────────────────────────────────
echo ""
echo "--- 03-Prompt工程 ---"
for f in 03-Prompt工程/*.py; do
    check_syntax "$f"
done
run_script "03-Prompt工程/04-prompt安全与system_prompt.py"

# ── 04-RAG ──────────────────────────────────────────────
echo ""
echo "--- 04-RAG ---"
for f in 04-RAG/mini_rag/*.py; do
    check_syntax "$f"
done
run_script "04-RAG/mini_rag/hybrid_search.py"
run_script "04-RAG/mini_rag/rerank_demo.py"
run_script "04-RAG/mini_rag/hyde_demo.py"

# ── 05-Agent ──────────────────────────────────────────
echo ""
echo "--- 05-Agent ---"
for f in 05-Agent/*.py; do
    check_syntax "$f"
done
run_script "05-Agent/06-plan_and_execute.py"

# ── 06-微调与部署 ──────────────────────────────────────
echo ""
echo "--- 06-微调与部署 ---"
for f in 06-微调与部署/*.py; do
    check_syntax "$f"
done
run_script "06-微调与部署/06-rlhf_dpo演示.py"
run_script "06-微调与部署/07-qlora与flash_attention.py"

# ── 07-多模态 ──────────────────────────────────────────
echo ""
echo "--- 07-多模态 ---"
for f in 07-多模态/*.py; do
    check_syntax "$f"
done

# ── projects ──────────────────────────────────────────
echo ""
echo "--- projects ---"
check_syntax "projects/智能知识助手/main.py"

# ── 汇总 ──────────────────────────────────────────────
echo ""
echo "============================================================"
echo -e "  结果: ${GREEN}PASS=$PASS${NC}  ${RED}FAIL=$FAIL${NC}  ${YELLOW}SKIP=$SKIP${NC}"
echo "============================================================"

if [ $FAIL -gt 0 ]; then
    exit 1
fi
echo -e "${GREEN}All checks passed!${NC}"
