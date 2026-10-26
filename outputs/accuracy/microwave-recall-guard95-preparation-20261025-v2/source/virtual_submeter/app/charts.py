"""Offline history plots with explicit UTC axes and honest missing segments."""

import altair as alt


def history_chart(frame):
    series = [name for name in frame.columns if name != "时间 UTC"]
    frame = frame.assign(**{"时间戳 UTC": frame["时间 UTC"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")})
    return (
        alt.Chart(frame)
        .transform_fold(series, as_=["曲线", "功率 W"])
        .mark_line(invalid="break-paths-show-domains")
        .encode(
            x=alt.X("时间 UTC:T", scale=alt.Scale(type="utc"), axis=alt.Axis(format="%m-%d %H:%M")),
            y=alt.Y("功率 W:Q"),
            color=alt.Color("曲线:N"),
            tooltip=[alt.Tooltip("时间戳 UTC:N"), alt.Tooltip("曲线:N"), alt.Tooltip("功率 W:Q", format=".2f")],
        )
        .properties(height=340)
    )
