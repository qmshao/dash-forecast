
import colorlover as cl
import plotly.graph_objs as go
import numpy as np

# def generateGraph(name, totDays, predDays, dates, data):
    
def generateGraph(name, dates, data, totDays, valDays=0, predDays=3):
    # Colorscale
#    bright_cscale = [[0, "#ff3700"], [1, "#0b8bff"]]
    cscale = [
        (0, "#ffffff"),
        (0.125, "#FDEBCF"),
        (0.25, "#FFE0D7"),
        (0.375, "#ffe7dc"),
        (0.5, "#ffb199"),
        (0.625, "#E55A4D"),
        (0.75, "#CB2A2E"),
        (1, "#6F161D"),
    ]

    # Prepare Data
    xhist = dates[:totDays-valDays]
    yhist = np.array(data['hist'][:totDays-valDays])
    yfit = data['pred'][totDays-valDays][1][:totDays-valDays]
    
    xvali = dates[totDays-valDays:totDays]
    yvali = np.array(data['hist'][totDays-valDays:totDays])
    
    
    
    xpred = dates[totDays-valDays-1:totDays-valDays+predDays]
    ypred = [yhist[-1], *data['pred'][totDays-valDays][1][totDays-valDays:totDays-valDays+predDays]]
    ypred0 = [yhist[-1], *data['pred'][totDays-valDays][0][totDays-valDays:totDays-valDays+predDays]]
    
    xpredb = dates[totDays-valDays-1:totDays-valDays+predDays]
    ypredu = data['pred'][totDays-valDays][2][:predDays+1]
    ypredl = data['pred'][totDays-valDays][3][:predDays+1]
    
#    ypred0 = data['pred'][totDays-valDays][3][:predDays+1]
    
    # Plot 
    # sizefactor = 1 if name in ['湖北省','全国','湖北'] else 1
    sizefactor = 3
    tracehist = go.Scatter(
        x=xhist,
        y=yhist,
        mode="markers",
        name="历史确诊数据",
        line=None,
        marker=dict(size=np.log10(yhist+1)*sizefactor, color='#E55A4D',
                line=dict(
                    color='#000',
                    width=0
                ),),
    )

    tracevali = go.Scatter(
        x=xvali,
        y=yvali,
        mode="markers",
        name="验证确诊数据",
        marker=dict(
                size=np.log10(yvali+1)*sizefactor, 
                color='#CB2A2E',
                line=dict(
                    color='#000',
                    width=2
                ),
            ),
    )    
   
                
    # Polinimal Lower Bound
    tracepred0 = go.Scatter(
        x=xpredb,
        y=ypred0,
        mode="lines",
        name="多项式预测下限区间",
        line = dict(dash='dash',color='GoldenRod'),
        showlegend=False
    )     
    
    tracepred0shade = go.Scatter(
        x=xpredb,
        y=ypred[-predDays-1:],
        mode="none",
        fill='tonexty',
        name="多项式预测下限区间",
        fillcolor="rgba(218,165,32,0.4)",
    )    
    
    
    tracefit = go.Scatter(
        x=xhist,
        y=yfit,
        mode="lines",
        name="指数渐近拟合",
        line = dict(dash='dash',color='Violet'),
    ) 
    
    tracepred = go.Scatter(
        x=xpredb,
        y=ypred,
        mode="lines",
        name="指数渐近预测",
        line = dict(dash='dash',color='DeepSkyBlue'),
    ) 
    
    tracepredl = go.Scatter(
        x=xpredb,
        y=ypredl,
        mode="lines",
        name="指数渐近预测",
#        line_color='indigo',
        line = dict(color='DeepSkyBlue',width=0),
        showlegend=False
    )    

    tracepredu = go.Scatter(
        x=xpredb,
        y=ypredu,
        mode="none",
        fill='tonexty',
        name="指数预测模型68%置信空间",
        fillcolor="rgba(0,191,255,0.4)",
        line = dict(color='DeepSkyBlue'),
    )    

    layout = go.Layout(
        xaxis=dict(ticks="", showticklabels=True, showgrid=False, zeroline=False, fixedrange=True),
        yaxis=dict(ticks="", showticklabels=True, showgrid=False, zeroline=False, fixedrange=True),
        hovermode="closest",
        legend=dict(x=0.05, y=0.95, orientation="v"),
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor="#282b38",
        paper_bgcolor="#282b38",
        font={"color": "#a5b1cd"},
    )

    data = [tracehist,tracevali,tracepred0, tracepred0shade, tracefit, tracepred,tracepredl,tracepredu]
    figure = go.Figure(data=data, layout=layout)

    return figure
