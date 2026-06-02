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
    return summary, meta, factors, nav, ic, gr

if not os.path.exists(BUNDLE):
    st.error("未找到 data_bundle.json,请确认已上传该文件。")
    st.stop()

summary, meta, factors, NAV, IC, GR = load_all()

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
figls.update_yaxes(type="log", title="多空累计净值(对数)")
figls.update_layout(height=420, hovermode="x unified", legend=dict(orientation="h"), margin=dict(t=20))
st.plotly_chart(figls, use_container_width=True)

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

st.subheader("⑤ AI 多因子对比备忘录")
st.caption("由 LLM 基于上述对比结果生成的投研备忘录(示例)。线上为预生成只读展示。")
mp = os.path.join(HERE, "multifactor_memo.md")
try:
    st.markdown(open(mp, encoding="utf-8").read())
except FileNotFoundError:
    st.info("备忘录文件未找到。")
st.caption("FactorLab 升级版 · 多因子框架 + IC + AI 对比备忘录")
