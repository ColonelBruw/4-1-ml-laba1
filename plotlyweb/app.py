"""Веб-интерфейс (Flask + Plotly) поверх БД `ts` пайплайна rosstat_ts_etl.

По каждому датасету и ряду: исходный ряд + скользящие средние (3/6/12),
разложение observed/trend/seasonal/resid, ACF с доверительным коридором,
прогноз (SES/Holt/HW) и результат ADF-теста.

Запуск (compose-сервис ts-web): http://localhost:8899
БД задаётся env TS_DATABASE_URL (по умолчанию — postgres из compose.yaml).
"""
from __future__ import annotations

import os
import re

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from flask import Flask, abort, redirect, url_for
from sqlalchemy import create_engine, text

DB_URL = os.environ.get("TS_DATABASE_URL", "postgresql+psycopg2://airflow:db-password@postgres:5432/ts")
engine = create_engine(DB_URL)
app = Flask(__name__)

PLOTLY_JS = '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>'
CSS = """
body{font-family:system-ui,Segoe UI,Roboto,Arial,sans-serif;margin:0;background:#f6f7fb;color:#1c2333}
.wrap{max-width:1180px;margin:0 auto;padding:24px}
h1{font-size:22px;margin:0 0 4px} h2{font-size:19px;margin:34px 0 6px;border-left:4px solid #4c6ef5;padding-left:10px}
.meta{color:#5b6472;font-size:14px;margin-bottom:14px}
.card{background:#fff;border:1px solid #e3e6ee;border-radius:12px;padding:14px 16px;margin:12px 0;box-shadow:0 1px 3px rgba(20,30,60,.05)}
.badge{display:inline-block;padding:2px 10px;border-radius:999px;font-size:13px;margin-right:6px}
.ok{background:#e6f7ee;color:#0c7a43}.bad{background:#fdecec;color:#b42323}.neutral{background:#eef1f8;color:#3b4a6b}
a{color:#3b5bdb;text-decoration:none} table{border-collapse:collapse} td,th{padding:4px 12px;font-size:14px;text-align:left}
"""

def q(sql, **params):
    with engine.connect() as con:
        return pd.read_sql_query(text(sql), con, params=params or None)

def period_to_ts(p):
    p = str(p).strip()
    if re.fullmatch(r"\d{4}-\d{2}", p):
        return pd.Period(p, freq="M").to_timestamp()
    m = re.fullmatch(r"(\d{4})[- ]?[Qq]([1-4])", p)
    if m:
        return pd.Period(m.group(1) + "Q" + m.group(2), freq="Q").to_timestamp()
    if re.fullmatch(r"\d{4}", p):
        return pd.Period(p, freq="Y").to_timestamp()
    return None

def axis_values(periods):
    ts = [period_to_ts(p) for p in periods]
    if all(t is not None for t in ts):
        return ts, True
    return list(range(len(ts))), False

def fig_html(fig, height=380):
    fig.update_layout(template="plotly_white", height=height,
                      margin=dict(l=50, r=20, t=42, b=40), legend=dict(orientation="h"))
    return fig.to_html(full_html=False, include_plotlyjs=False)

@app.route("/")
def index():
    cfg = q("SELECT series_id, title, freq, n_points FROM series_config ORDER BY series_id")
    if len(cfg) == 1:
        return redirect(url_for("dataset_page", ds_id=cfg.iloc[0]["series_id"]))
    if len(cfg):
        rows = "".join('<tr><td><a href="/dataset/' + r.series_id + '">' + (r.title or r.series_id)
                       + "</a></td><td>" + (r.freq or "") + "</td><td>" + str(r.n_points or "") + "</td></tr>"
                       for r in cfg.itertuples())
        body = ("<h1>Датасеты Росстата в БД ts</h1><div class=card><table>"
                "<tr><th>Датасет</th><th>Частота</th><th>Точек</th></tr>" + rows + "</table></div>")
    else:
        body = ("<h1>Пока пусто</h1><div class=card>Запустите DAG <b>rosstat_ts_etl</b> в Airflow, "
                "затем обновите страницу.</div>")
    return ("<html><head><meta charset='utf-8'><title>ts dashboard</title><style>" + CSS
            + "</style></head><body><div class=wrap>" + body + "</div></body></html>")

@app.route("/dataset/<ds_id>")
def dataset_page(ds_id):
    cfg = q("SELECT * FROM series_config WHERE series_id = :ds", ds=ds_id)
    if cfg.empty:
        abort(404)
    c = cfg.iloc[0]
    long = q("SELECT * FROM series_long WHERE ds_id = :ds ORDER BY series_label, period_order", ds=ds_id)
    rm = q("SELECT * FROM rolling_means WHERE ds_id = :ds", ds=ds_id)
    dec = q("SELECT * FROM decomposition WHERE ds_id = :ds", ds=ds_id)
    acf = q("SELECT * FROM acf_values WHERE ds_id = :ds ORDER BY series_label, lag", ds=ds_id)
    st = q("SELECT * FROM stationarity WHERE ds_id = :ds", ds=ds_id)
    fc = q("SELECT * FROM forecasts WHERE ds_id = :ds ORDER BY series_label, step", ds=ds_id)

    parts = ["<h1>" + (c["title"] or ds_id) + "</h1>",
             "<div class=meta>ds_id: <code>" + ds_id + "</code> · частота: " + str(c["freq"])
             + " · точек: " + str(c["n_points"])
             + (" · ед.: " + str(c["unit"]) if c.get("unit") else "")
             + (' · <a href="' + str(c["source_url"]) + '" target="_blank">источник (roschart.ru)</a>'
                if c.get("source_url") else "") + "</div>"]

    for label, g in long.groupby("series_label", sort=True):
        g = g.sort_values("period_order").reset_index(drop=True)
        x, is_date = axis_values(g["period"])
        pos = {int(o): i for i, o in enumerate(g["period_order"])}
        parts.append("<h2>Ряд: " + str(label) + "</h2>")

        srow = st[st.series_label == label]
        if not srow.empty:
            s = srow.iloc[0]
            badge = ("<span class='badge ok'>стационарен</span>" if bool(s.is_stationary)
                     else "<span class='badge bad'>нестационарен</span>")
            parts.append("<div class=card><b>ADF-тест:</b> " + badge
                         + "<span class='badge neutral'>stat = %.3f</span>" % s.statistic
                         + "<span class='badge neutral'>p-value = %.4g</span>" % s.p_value
                         + "<span class='badge neutral'>nobs = %d</span></div>" % int(s.nobs))

        f1 = go.Figure()
        f1.add_trace(go.Scatter(x=x, y=g.value, name="исходный ряд", mode="lines",
                                line=dict(color="#4c6ef5", width=1.6)))
        gl = rm[rm.series_label == label]
        for w, color in zip((3, 6, 12), ("#f59f00", "#2f9e44", "#e03131")):
            gw = gl[gl.window == w].sort_values("period_order")
            if not gw.empty:
                xx = [x[pos[int(i)]] for i in gw.period_order if int(i) in pos]
                f1.add_trace(go.Scatter(x=xx, y=gw.value, name="MA(%d)" % w, mode="lines",
                                        line=dict(color=color, width=1.4)))
        f1.update_layout(title="Исходный ряд и скользящие средние")
        parts.append("<div class=card>" + fig_html(f1) + "</div>")

        gd = dec[dec.series_label == label]
        if not gd.empty:
            comps = [("observed", "Наблюдаемый"), ("trend", "Тренд"),
                     ("seasonal", "Сезонность"), ("resid", "Остаток")]
            f2 = make_subplots(rows=4, cols=1, shared_xaxes=True,
                               subplot_titles=[t for _, t in comps], vertical_spacing=0.06)
            for i, (comp, _) in enumerate(comps, start=1):
                gc = gd[gd.component == comp].sort_values("period_order")
                xx = [x[pos[int(j)]] for j in gc.period_order if int(j) in pos]
                f2.add_trace(go.Scatter(x=xx, y=gc.value, mode="lines", name=comp,
                                        line=dict(width=1.2)), row=i, col=1)
            f2.update_layout(title="Разложение ряда (seasonal_decompose)", showlegend=False)
            parts.append("<div class=card>" + fig_html(f2, height=640) + "</div>")

        ga = acf[acf.series_label == label].sort_values("lag")
        if not ga.empty:
            f3 = go.Figure()
            f3.add_trace(go.Bar(x=ga.lag, y=ga.acf, name="ACF", marker_color="#4c6ef5"))
            f3.add_trace(go.Scatter(x=ga.lag, y=ga.ci_upper, mode="lines", name="95% ДИ",
                                    line=dict(color="#adb5bd", width=1, dash="dot")))
            f3.add_trace(go.Scatter(x=ga.lag, y=ga.ci_lower, mode="lines", showlegend=False,
                                    line=dict(color="#adb5bd", width=1, dash="dot")))
            f3.update_layout(title="Автокорреляционная функция (ACF)", xaxis_title="лаг")
            parts.append("<div class=card>" + fig_html(f3) + "</div>")

        gf = fc[fc.series_label == label].sort_values("step")
        if not gf.empty:
            model_used, horizon = gf.iloc[0]["model_used"], int(gf.iloc[0]["horizon"])
            tail = g.tail(48)
            monthly = is_date and len(x) > 2 and 26 <= pd.Series(x).diff().dt.days.median() <= 32
            if monthly:
                xt = x[-len(tail):]
                per = pd.Period(str(g.iloc[-1]["period"]), freq="M")
                xf = [(per + int(s)).to_timestamp() for s in gf.step]
            else:
                xt = list(tail["period"])
                xf = ["шаг +%d" % int(s) for s in gf.step]
            f4 = go.Figure()
            f4.add_trace(go.Scatter(x=xt, y=tail.value, name="история (хвост)", mode="lines",
                                    line=dict(color="#4c6ef5", width=1.6)))
            f4.add_trace(go.Scatter(x=xf, y=gf.value,
                                    name="прогноз %s (+%d)" % (model_used.upper(), horizon),
                                    mode="lines+markers",
                                    line=dict(color="#e03131", width=2, dash="dash")))
            f4.update_layout(title="Прогноз: модель %s, горизонт %d" % (model_used.upper(), horizon))
            parts.append("<div class=card>" + fig_html(f4) + "</div>")

    if long.empty:
        parts.append("<div class=card>Данные ещё не загружены — запустите DAG rosstat_ts_etl.</div>")

    return ("<html><head><meta charset='utf-8'><title>" + (c["title"] or ds_id) + "</title>"
            + PLOTLY_JS + "<style>" + CSS + "</style></head><body><div class=wrap>"
            + "".join(parts) + "</div></body></html>")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8899")))