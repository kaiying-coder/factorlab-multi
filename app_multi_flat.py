"""
FactorLab — 多因子对比看板(部署版,app_multi_deploy.py)
===========================================================
读 deploy_data/ 下的 parquet+json(稳健格式,非 pickle),展示:
  ① 因子对比总表  ② ICIR柱状图  ③ 多空净值对比  ④ 单因子下钻  ⑤ AI对比备忘录
streamlit run app_multi_deploy.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json, os
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="FactorLab 多因子对比", layout="wide")
HERE = os.path.dirname(os.path.abspath(__file__))
PRE = os.path.join(HERE, "fdata_")


@st.cache_data
def load_all():
    summary = pd.read_csv(PRE + "summary.csv", index_col=0)
    meta = json.load(open(PRE + "meta.json", encoding="utf-8"))
    factors = list(summary.index)
    nav = {f: pd.read_parquet(PRE + f"nav_{f}.parquet") for f in factors}
    ic = {f: pd.read_parquet(PRE + f"ic_{f}.parquet")["ic"] for f in factors}
    gr = {f: pd.read_parquet(PRE + f"gr_{f}.parquet") for f in factors}
    return summary, meta, factors, nav, ic, gr


if not os.path.exists(PRE + "summary.csv"):
    st.error("未找到 fdata_ 数据文件,请确认已上传全部 fdata_ 开头的文件。")
    st.stop()

summary, meta, factors, NAV, IC, GR = load_all()

st.title("📊 FactorLab — 多因子对比看板")
st.caption("A股 2006-2025 · 市值/动量/反转/低波动 · 分层回测 + IC 分析")

# ① 对比总表
st.subheader("① 因子对比总表")
disp = summary[["IC均值", "ICIR", "IC>0占比", "多空年化", "多空Sharpe", "多空回撤"]].copy()
st.dataframe(
    disp.style.background_gradient(subset=["ICIR", "多空Sharpe"], cmap="RdYlGn")
        .format({"IC均值": "{:.4f}", "ICIR": "{:.3f}", "IC>0占比": "{:.1%}",
                 "多空年化": "{:.1%}", "多空Sharpe": "{:.2f}", "多空回撤": "{:.1%}"}),
    use_container_width=True)
best = summary["ICIR"].abs().idxmax()
st.info(f"按 |ICIR| 排名,预测力最强的因子是 **{best}**(ICIR={summary.loc[best,'ICIR']:.3f})。")

# ② ICIR 柱状图
st.subheader("② 各因子 ICIR 对比")
colors = ["#185FA5" if v >= 0 else "#A32D2D" for v in summary["ICIR"]]
fig_ic = go.Figure(go.Bar(x=factors, y=summary["ICIR"], marker_color=colors,
                          text=summary["ICIR"].round(3), textposition="outside"))
fig_ic.update_layout(height=360, yaxis_title="ICIR", margin=dict(t=20))
st.plotly_chart(fig_ic, use_container_width=True)

# ③ 多空净值对比
st.subheader("③ 多空组合净值对比")
fig_ls = go.Figure()
for f in factors:
    gr = GR[f]
    lo, hi = gr.columns.min(), gr.columns.max()
    ls = (gr[hi] - gr[lo]) if meta[f]["IC均值"] >= 0 else (gr[lo] - gr[hi])
    nav = (1 + ls).cumprod()
    fig_ls.add_trace(go.Scatter(x=nav.index, y=nav, name=f, mode="lines"))
fig_ls.update_yaxes(type="log", title="多空累计净值(对数)")
fig_ls.update_layout(height=420, hovermode="x unified", legend=dict(orientation="h"), margin=dict(t=20))
st.plotly_chart(fig_ls, use_container_width=True)

# ④ 单因子下钻
st.subheader("④ 单因子下钻")
sel = st.selectbox("选择因子", factors, index=factors.index(best))
c1, c2 = st.columns(2)
with c1:
    st.markdown(f"**{sel} — 各组累计净值**")
    nav = NAV[sel]; nav.columns = [int(c) for c in nav.columns]
    cols = sorted(nav.columns)
    fig_g = go.Figure()
    for i, g in enumerate(cols):
        col = px.colors.sample_colorscale("RdYlBu", i / (len(cols) - 1))[0]
        fig_g.add_trace(go.Scatter(x=nav.index, y=nav[g], name=f"组{g}", line=dict(color=col, width=1.2)))
    fig_g.update_yaxes(type="log")
    fig_g.update_layout(height=360, margin=dict(t=10), legend=dict(orientation="h", font=dict(size=9)))
    st.plotly_chart(fig_g, use_container_width=True)
with c2:
    st.markdown(f"**{sel} — IC 月度均值**")
    ic_m = IC[sel].copy(); ic_m.index = pd.to_datetime(ic_m.index)
    ic_m = ic_m.resample("ME").mean()
    fig_t = go.Figure(go.Bar(x=ic_m.index, y=ic_m,
                             marker_color=["#185FA5" if v >= 0 else "#A32D2D" for v in ic_m]))
    fig_t.add_hline(y=0, line_color="gray")
    fig_t.update_layout(height=360, margin=dict(t=10), yaxis_title="月均 IC")
    st.plotly_chart(fig_t, use_container_width=True)
st.caption(f"{sel}:IC均值 {meta[sel]['IC均值']:.4f} · ICIR {meta[sel]['ICIR']:.3f} · "
           f"多空方向 {meta[sel]['多空方向']} · 多空Sharpe {meta[sel]['多空Sharpe']:.2f}")

# ⑤ AI 对比备忘录
st.subheader("⑤ AI 多因子对比备忘录")
st.caption("由 LLM 基于上述对比结果生成的投研备忘录(示例)。线上为预生成只读展示,不实时调模型。")
memo_path = os.path.join(HERE, "multifactor_memo.md")
try:
    st.markdown(open(memo_path, encoding="utf-8").read())
except FileNotFoundError:
    st.info("备忘录文件未找到。")

st.caption("FactorLab 升级版 · 多因子框架 + IC + AI 对比备忘录")
