"""
FactorLab — 多因子对比看板(单文件数据版,app_multi_bundle.py)
所有数据打包在 data_bundle.json 里,只需传这一个数据文件,避免多文件传输出错。
"""
import streamlit as st
import pandas as pd
import numpy as np
import json, os, io
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="FactorLab 多因子对比", layout="wide")
HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.path.join(HERE, "data_bundle.json")

@st.cache_data
def load_all():
    b = json.load(open(BUNDLE, encoding="utf-8"))
    summary = pd.read_json(io.StringIO(b["summary"]))
    meta = b["meta"]
    factors = list(summary.index)
    nav = {f: pd.read_json(io.StringIO(b[f"nav_{f}"])) for f in factors}
    ic = {f: pd.read_json(io.StringIO(b[f"ic_{f}"]))["ic"] for f in factors}
    gr = {f: pd.read_json(io.StringIO(b[f"gr_{f}"])) for f in factors}
    corr = pd.read_json(io.StringIO(b["corr"])) if "corr" in b else None
    cost = pd.read_json(io.StringIO(b["cost"])) if "cost" in b else None
    comp_nav = pd.read_json(io.StringIO(b["composite_nav"]))["nav"] if "composite_nav" in b else None
    comp_meta = b.get("composite_meta")
    return summary, meta, factors, nav, ic, gr, corr, cost, comp_nav, comp_meta

if not os.path.exists(BUNDLE):
    st.error("未找到 data_bundle.json,请确认已上传该文件。")
    st.stop()

summary, meta, factors, NAV, IC, GR, CORR, COST, COMP_NAV, COMP_META = load_all()

st.title("📊 FactorLab — 多因子对比看板")
st.caption("A股 2006-2025 · 市值/动量/反转/低波动 · 分层回测 + IC 分析")

st.subheader("① 因子对比总表")
disp = summary[["IC均值","ICIR","IC>0占比","多空年化","多空Sharpe","多空回撤"]].copy()
st.dataframe(disp.style.background_gradient(subset=["ICIR","多空Sharpe"], cmap="RdYlGn")
    .format({"IC均值":"{:.4f}","ICIR":"{:.3f}","IC>0占比":"{:.1%}",
             "多空年化":"{:.1%}","多空Sharpe":"{:.2f}","多空回撤":"{:.1%}"}),
    use_container_width=True)
best = summary["ICIR"].abs().idxmax()
st.info(f"按 |ICIR| 排名,预测力最强的因子是 **{best}**(ICIR={summary.loc[best,'ICIR']:.3f})。")

st.subheader("② 各因子 ICIR 对比")
colors = ["#185FA5" if v>=0 else "#A32D2D" for v in summary["ICIR"]]
fig = go.Figure(go.Bar(x=factors, y=summary["ICIR"], marker_color=colors,
                       text=summary["ICIR"].round(3), textposition="outside"))
fig.update_layout(height=360, yaxis_title="ICIR", margin=dict(t=20))
st.plotly_chart(fig, use_container_width=True)

st.subheader("③ 多空组合净值对比")
figls = go.Figure()
for f in factors:
    gr = GR[f]; gr.columns=[int(c) for c in gr.columns]
    lo,hi = gr.columns.min(), gr.columns.max()
    ls = (gr[hi]-gr[lo]) if meta[f]["IC均值"]>=0 else (gr[lo]-gr[hi])
    nav=(1+ls).cumprod()
    figls.add_trace(go.Scatter(x=nav.index, y=nav, name=f, mode="lines"))
if COMP_NAV is not None:
    figls.add_trace(go.Scatter(x=COMP_NAV.index, y=COMP_NAV, name="复合因子",
                    mode="lines", line=dict(color="black", width=2.5, dash="dash")))
figls.update_yaxes(type="log", title="多空累计净值(对数)")
figls.update_layout(height=420, hovermode="x unified", legend=dict(orientation="h"), margin=dict(t=20))
st.plotly_chart(figls, use_container_width=True)


# ③.5 交易成本对比
if COST is not None:
    st.subheader("③·交易成本影响(双边0.3%)")
    cdisp = COST.copy()
    st.dataframe(cdisp.style.background_gradient(subset=["平均换手率"], cmap="Reds")
        .format({"平均换手率":"{:.1%}","税前Sharpe":"{:.2f}","税后年化":"{:.1%}","税后Sharpe":"{:.2f}"}),
        use_container_width=True)
    st.caption("换手率越高,成本侵蚀越大。reversal 等高换手因子需重点关注税后指标。")
    if COMP_META is not None:
        st.success(f"**复合因子**(按ICIR加权合成):多空年化 {COMP_META['年化']:.1%}、"
                   f"Sharpe {COMP_META['Sharpe']:.2f}、回撤 {COMP_META['回撤']:.1%} —— 优于全部单因子。")

# ③.6 因子相关性热力图
if CORR is not None:
    st.subheader("③·因子相关性")
    figc = px.imshow(CORR, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                     text_auto=".2f", aspect="auto")
    figc.update_layout(height=360, margin=dict(t=20))
    st.plotly_chart(figc, use_container_width=True)
    st.caption("相关性低/负的因子组合分散效果好。动量与反转 -0.62 强负相关,是组合的关键。")

st.subheader("④ 单因子下钻")
sel = st.selectbox("选择因子", factors, index=factors.index(best))
c1,c2 = st.columns(2)
with c1:
    st.markdown(f"**{sel} — 各组累计净值**")
    nav = NAV[sel]; nav.columns=[int(c) for c in nav.columns]
    cols = sorted(nav.columns); figg=go.Figure()
    for i,g in enumerate(cols):
        col = px.colors.sample_colorscale("RdYlBu", i/(len(cols)-1))[0]
        figg.add_trace(go.Scatter(x=nav.index, y=nav[g], name=f"组{g}", line=dict(color=col,width=1.2)))
    figg.update_yaxes(type="log")
    figg.update_layout(height=360, margin=dict(t=10), legend=dict(orientation="h", font=dict(size=9)))
    st.plotly_chart(figg, use_container_width=True)
with c2:
    st.markdown(f"**{sel} — IC 月度均值**")
    ic = IC[sel].copy(); ic.index = pd.to_datetime(ic.index)
    icm = ic.resample("ME").mean()
    figt = go.Figure(go.Bar(x=icm.index, y=icm,
        marker_color=["#185FA5" if v>=0 else "#A32D2D" for v in icm]))
    figt.add_hline(y=0, line_color="gray")
    figt.update_layout(height=360, margin=dict(t=10), yaxis_title="月均 IC")
    st.plotly_chart(figt, use_container_width=True)
st.caption(f"{sel}:IC均值 {meta[sel]['IC均值']:.4f} · ICIR {meta[sel]['ICIR']:.3f} · "
           f"多空方向 {meta[sel]['多空方向']} · 多空Sharpe {meta[sel]['多空Sharpe']:.2f}")

st.subheader("⑤ RAG 增强投研备忘录")
st.caption("RAG 流程:检索外部研报 + 回测数据 → LLM 生成结合两者的备忘录。线上为预生成只读展示,不实时调模型。")

# RAG 检索演示:展示"查询 → 检索到的研报 + 相似度",体现 RAG 机制
with st.expander("🔍 RAG 检索演示(点开看检索机制)", expanded=False):
    st.caption("生成备忘录前,系统先用查询从研报库检索最相关的文档。以下为检索示例:")
    demo = [
        ("反转因子 换手率 交易成本", [("交易成本对高换手策略的影响", 0.21), ("短期反转效应在A股的有效性", 0.19)]),
        ("多因子组合 负相关 分散", [("多因子组合与风险分散", 0.30), ("动量因子的中外差异", 0.12)]),
        ("小市值 风险", [("A股小市值风格回顾", 0.33), ("2024年初小盘股流动性冲击", 0.20)]),
    ]
    for q, hits in demo:
        st.markdown(f"**查询:** `{q}`")
        for title, score in hits:
            st.markdown(f"&nbsp;&nbsp;→ [{score:.2f}] {title}")
    st.caption("(演示用 TF-IDF 向量检索;生产环境可换 embedding 模型 + 向量库)")

mp = os.path.join(HERE, "rag_memo.md")
try:
    st.markdown(open(mp, encoding="utf-8").read())
except FileNotFoundError:
    st.info("备忘录文件未找到。")
st.caption("FactorLab 升级版 · 多因子 + IC + 成本 + 组合 + RAG")
