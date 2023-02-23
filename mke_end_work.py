from pyecharts.charts import Page  # 画图

Page.save_resize_html(      # 生成最终html网页
    source="临时.html",
    cfg_file="chart_config.json",
    dest='保罗乔治职业生涯数据可视化.html'
)
print("网页生成完成!")
