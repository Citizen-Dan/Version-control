import dash_auth
# from users import USERNAME_PASSWORD_PAIRS



from dash import Dash, html, dcc, Input, Output, no_update
import dash_daq as daq

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import io
import base64
from PIL import Image
import numpy as np
from textwrap import wrap

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

USERNAME_PASSWORD_PAIRS = {
     'WSP': 'Daisy_project'

}


app = Dash(__name__, external_stylesheets=external_stylesheets,suppress_callback_exceptions=True)
auth = dash_auth.BasicAuth(app, USERNAME_PASSWORD_PAIRS)

server = app.server
#%% Loading data
# Path to csv of data file from Grasshopper analysis
path = '100LH - Hypercube and optimization.csv'
try:
    df = pd.read_csv(path)
except:
    df = pd.read_csv(path, encoding = "ISO-8859-1")
    
df.drop('Time Stamp [yyyy:mm:dd:hh:mm:ss]',axis=1,inplace=True)

# Isolate failed grasshopper runs
# Removing of voided data - Defined as where BEU, EUI or Daylight is blank or NaN
null_indexs = np.where(pd.isnull(df['Building Embodied Carbon [kgCO2/m2]']) |
                       pd.isnull(df['Energy Use Intensity [kWh/m2]']) |
                       pd.isnull(df['Daylight [%sDA]']) 
                      )[0]

df.drop(null_indexs,axis=0,inplace=True)

del(null_indexs,path)

drop_EC = False # Whether to drop EC genes with zero variance
drop_EUI = False # Whether to drop EUI processes with zero variance

# Good data
features_to_drop = []
for feature in df.columns:
    if feature!='Label':
        if 'EC' in feature:
            if drop_EC==True:
                if df[feature].var()<1e-8:
                    features_to_drop.append(feature)

        elif feature in ['Process','Hot Water']:
            if drop_EUI == True:
                if df[feature].var()<1e-8:
                    features_to_drop.append(feature)
        elif pd.api.types.is_numeric_dtype(df[feature]):
            if df[feature].var()<1e-8:
                features_to_drop.append(feature)

# Remove features dropped from dataframe
df.drop(features_to_drop,axis=1,inplace=True)

drop_text_variables = True
if drop_text_variables == True:
    features_to_drop = []
    for feature in df.columns:
        if feature!='Label':
            if not pd.api.types.is_numeric_dtype(df[feature]):
                features_to_drop.append(feature)
    df = df.drop(features_to_drop,axis=1)
del(drop_EC,drop_EUI,drop_text_variables,feature,features_to_drop)

it_GA_skip = 1+np.max(df.loc[df['Label'] == '100LH - Hypercube']['Itteration [n]'])

df.loc[df['Label'] == '100LH - Optimization',
       'Itteration [n]'] = df.loc[df['Label'] ==  '100LH - Optimization','Itteration [n]']+ 1+np.max(
           df.loc[df['Label'] == '100LH - Hypercube']['Itteration [n]'])
df.sort_values(by=['Itteration [n]'], inplace=True)

## Dropping GA iteration 3683 due to very low glazing ratio causing outlier result
df.drop(df[df['Itteration [n]']==3683].index,inplace=True)
df['Daylight [%sDA]']=df['Daylight [%sDA]']*100

df['EC Structures'] = df[['EC Floors', 
                            'EC Columns',
                            'EC Stability',
                            'EC Foundations',
                            'EC Basement',]].sum(axis=1) 
df['EC Facades'] = df[['EC Walls', 
                            'EC Windows',
                            'EC Window Frame',
                            'EC Shades',]].sum(axis=1) 

#%%
# Path to folder containing images
# C:\Users\UKZXW002\OneDrive - WSP O365\Documents\GitHub\Daisy-DSN\Daisy_DSN_Dashboard\v1.3_Password_Protection
image_path =r'C:\Users\UKZXW002\OneDrive - WSP O365\Documents\GitHub\DAISY-Hosted\daisy\images\\'
image_format_GA = image_path+'100LH_Image_{}_DAISY_ISO_SE.png'
image_format_Hypercube = image_path+'Hypercube_100LH_Image_{}_DAISY_ISO_SE.png'

# Removing of voided data - Defined as where BEU, EUI or Daylight is blank or NaN
df.drop(np.where(pd.isnull(df['Building Embodied Carbon [kgCO2/m2]']))[0],axis=0,inplace=True)
df.drop(np.where(pd.isnull(df['Energy Use Intensity [kWh/m2]']))[0],axis=0,inplace=True)
df.drop(np.where(pd.isnull(df['Daylight [%sDA]']))[0],axis=0,inplace=True)

df['EC Structures'] = df[['EC Floors', 
                            'EC Columns',
                            'EC Stability',
                            'EC Foundations',
                            'EC Basement',]].sum(axis=1) 
df['EC Facades'] = df[['EC Walls', 
                            'EC Windows',
                            'EC Window Frame',
                            'EC Shades',]].sum(axis=1) 

                  
def gen_radar_plot(df,bench_val,comp_val):  
    
    titles = ['Building Embodied Carbon [kgCO2/m2]','Energy Use Intensity [kWh/m2]','Daylight [%sDA]']

    r = df[['Itteration [n]']+titles]
    r[titles] = r[titles]-r[titles].min()
    r[titles] = r[titles]/r[titles].max()
    r['Building Embodied Carbon [kgCO2/m2]'] = 1-r['Building Embodied Carbon [kgCO2/m2]']
    r['Energy Use Intensity [kWh/m2]'] = 1-r['Energy Use Intensity [kWh/m2]']
    r_bench = 100*r.loc[r['Itteration [n]'] == bench_val]
    r_comp = 100*r.loc[r['Itteration [n]'] == comp_val]
    
    
    titles = titles+titles[:1]
    r_bench=np.array(r_bench[titles]).squeeze()
    r_comp=np.array(r_comp[titles]).squeeze()


    fig = go.Figure()
    
    labels = ['<br>'.join(wrap(l, 10)) for l in titles]
    fig.add_trace(go.Scatterpolar(
          name = "Selected iteration",
          r = r_comp,
          theta = labels,
          hoverinfo='none',
        ))
    fig.add_trace(go.Scatterpolar(
          name = "Benchmark",
          r = r_bench,
          theta = labels,
          hoverinfo='none',
        ))

    
    
    fig.update_polars()#,radialaxis_showticksuffix="all",radialaxis_ticksuffix='%')
    fig.update_layout(polar = dict(radialaxis = dict(showticklabels = False)))

    fig.update_layout(autosize=True,
                      uirevision=1,
                      legend=dict(
                            x=0.75,
                            y=1.25,
                            orientation="v",
                            traceorder="normal",
                            title='Relative Ranking'
                            
                            ),
                      font=dict(
                          family="sans-serif",
                          size=10,
                          color="black"
                      ),
                      
                      plot_bgcolor='#A4AEB8',
                      paper_bgcolor='#A4AEB8',
                      margin=dict(
                          l=30,
                          r=30,
                          b=30,
                          t=30))
    
    return fig

def gen_BEC_pie_chart(df,iteration_name):  
    
    titles_BEC = ['EC Floors',
                 'EC Columns',
                 'EC Stability',
                 'EC Foundations',
                 'EC Basement',
                 'EC Walls',
                 'EC Windows',
                 'EC Window Frame',
                 'EC Shades']
    parents_BEC = ['EC Structures',
                   'EC Facades']
       
    df_plot = df.loc[df['Itteration [n]'] == iteration_name]
    df_plot = df_plot.round(2)
    fig = go.Figure()
    fig.add_trace(go.Pie(values=df_plot[parents_BEC].values[0],
                            labels=parents_BEC,
                            hole=0.4,
               direction='clockwise',
                            domain={'x':[0.2,0.8], 'y':[0.1,0.9]},
                            sort=False,
                            marker={'colors':['#C61A09','#2E86C1']},
                            hoverinfo='label+value'
        ))
    fig.add_trace(go.Pie(values=df_plot[titles_BEC].values[0],
                 labels=titles_BEC,
                 sort=False,
               direction='clockwise',
                 domain={'x':[0.1,0.9], 'y':[0,1]},
                 hole=0.75,
                 marker={'colors':['#FFC9BB','#FFA590',
                                   '#FF8164','#FF6242','#FF4122',
                                   '#29C5F66','#3A9BDC',
                                   '#5579C6','#1260CC']},
                 hoverinfo='label+value'))

    

    fig.update_layout(autosize=False,
                      uirevision=1,
                      legend=dict(
                            x=0.75,
                            y=1.,
                            orientation="v",
                            traceorder="normal",
                            
                            ),
                      font=dict(
                          family="sans-serif",
                          size=10,
                          color="black"
                      ),
                      
                      plot_bgcolor='#A4AEB8',
                      paper_bgcolor='#A4AEB8',
                      margin=dict(
                          l=30,
                          r=30,
                          b=30,
                          t=30))
    
    return fig
def gen_EUI_pie_chart(df,iteration_name):  
    
    
    titles_EUI = ['Cooling','Heating','Lighting','Equipment','Process']#,'Hot Water']
   
    df_plot = df.loc[df['Itteration [n]'] == iteration_name]
    df_plot = df_plot.round(2)
    fig = go.Figure()

    fig.add_trace(go.Pie(values=df_plot[titles_EUI].values[0],
                 labels=titles_EUI,
                 sort=False,
               direction='clockwise',
                 domain={'x':[0.1,0.9], 'y':[0,1]},
                 hole=0.75,
                 # marker={'colors':['#FFC9BB','#FFA590',
                 #                   '#FF8164','#FF6242','#FF4122',
                 #                   '#29C5F66','#3A9BDC',
                 #                   '#5579C6','#1260CC']},
                 hoverinfo='label+value'))

    

    fig.update_layout(autosize=False,
                      uirevision=1,
                      legend=dict(
                            x=0.75,
                            y=1.,
                            orientation="v",
                            traceorder="normal",
                            
                            ),
                      font=dict(
                          family="sans-serif",
                          size=10,
                          color="black"
                      ),
                      
                      plot_bgcolor='#A4AEB8',
                      paper_bgcolor='#A4AEB8',
                      margin=dict(
                          l=30,
                          r=30,
                          b=30,
                          t=30))
    
    return fig



@app.callback(
    Output("comparison-plot-div", "figure"),
    Input('crossfilter-indicator-scatter', "clickData"),
    Input('crossfilter-benchmark','value')
    )
def update_radial(clickData,bench_value):
    if clickData is None:
        iteration_name = 4
    else:
        iteration_name = clickData["points"][0]['customdata'][0]

    radial_plot=gen_radar_plot(df,int(bench_value.split(' ')[-1]),iteration_name)

    return radial_plot

@app.callback(
    Output("carbon-breakdown-plot-div", "figure"),
    Input('crossfilter-indicator-scatter', "clickData"),
    )
def update_BEC_breakdown_plot(clickData):
    if clickData is None:
        iteration_name = 4
    else:
        iteration_name = clickData["points"][0]['customdata'][0]

    pie_chart=gen_BEC_pie_chart(df,iteration_name)

    return pie_chart

@app.callback(
    Output("energy-breakdown-plot-div", "figure"),
    Input('crossfilter-indicator-scatter', "clickData"),
    )
def update_EUI_breakdown_plot(clickData):
    if clickData is None:
        iteration_name = 4
    else:
        iteration_name = clickData["points"][0]['customdata'][0]

    pie_chart=gen_EUI_pie_chart(df,iteration_name)

    return pie_chart



@app.callback(
    Output("benchmark-div", "children"),
    Input('crossfilter-benchmark','value'),
    )
def update_benchmark(bench_value):

    # Load image 
    if int(bench_value.split(' ')[-1])>=it_GA_skip:
        img_path = image_format_GA.format(int(bench_value.split(' ')[-1])-it_GA_skip)
        it_label = 'Genetic Algotihm'
    else:
        img_path = image_format_Hypercube.format(bench_value.split(' ')[-1])
        it_label = 'Latin Hypercube'
    try:
        image = Image.open(img_path)
    except:
        image=Image.open(image_format_GA.format(8))
    # encoded_image = encoded_image.decode()
    buff = io.BytesIO()
    image.save(buff, format='png')
    encoded_image = base64.b64encode(buff.getvalue()).decode("utf-8")
    im_url = "data:image/jpeg;base64, " + encoded_image
    

    
    children = html.Div([
            
            html.Img(src=im_url, style={"width": "17vw"}),
            html.B('Benchmark 'f"{bench_value}", style={"color": "black", "overflow-wrap": "break-word",
                                                            'font-size':'1vw'}),
            html.Br(),
            html.B('('+it_label+')', style={"color": "black", "overflow-wrap": "break-word",
                                                            'font-size':'1vw'})
         ], style={'height': '97vh', 'white-space': 'none','text-align':'center'})
    

    return True,children



                  
@app.callback(
    Output("comparison-div", "children"),
    Input('crossfilter-indicator-scatter', "clickData"),
    )
def update_comparison(clickData):
    if clickData is None:
        iteration_name = 4
    else:
        iteration_name = clickData["points"][0]['customdata'][0]
    
    # Load image 
    # Load image 
    if iteration_name>=it_GA_skip:
        img_path = image_format_GA.format(iteration_name-it_GA_skip)
        it_label = 'Genetic Algotihm'
    else:
        img_path = image_format_Hypercube.format(iteration_name)
        it_label = 'Latin Hypercube'

    try:
        image = Image.open(img_path)
    except:
        image=Image.open(image_format_GA.format(8))
    # encoded_image = encoded_image.decode()
    buff = io.BytesIO()
    image.save(buff, format='png')
    encoded_image = base64.b64encode(buff.getvalue()).decode("utf-8")
    im_url = "data:image/jpeg;base64, " + encoded_image
    
       
    children = html.Div([
            
            html.Img(src=im_url, style={"width": "17vw",'margin':'0'}),
            html.B('Selected Iteration - ' f"{iteration_name}", style={"color": "black", "overflow-wrap": "break-word",
                                                            'font-size':'1vw'}),
            html.Br(),
            html.B('('+it_label+')', style={"color": "black", "overflow-wrap": "break-word",
                                                            'font-size':'1vw'})
         ], style={'height': '97vh', 'white-space': 'none','text-align':'center',})
    

    return True,children
                  

@app.callback(
    Output('crossfilter-indicator-scatter', 'figure'),
    Input('crossfilter-xaxis-column', 'value'),
    Input('crossfilter-yaxis-column', 'value'),
    Input('crossfilter-zaxis-column', 'value'),
    Input('crossfilter-colaxis-column', 'value'),
    Input('BEU-slider', 'value'),
    Input('EUI-slider', 'value'),
    Input('Daylight-slider', 'value'),
    Input('iteration-slider', 'value'),
    Input('Aspect Ratio-slider', 'value'),
    Input('crossfilter-benchmark','value'),
    Input('2d-toggle', 'value'),
    Input('invert-daylight-toggle', 'value'),
    Input('crossfilter-indicator-scatter', "clickData"),
    Input('show-ga-data', 'value'),
    Input('show-hypercube-data', 'value'),

    )
def update_graph(xaxis_column_name, yaxis_column_name,zaxis_column_name,color_column_name,
                 BEU_range,EUI_range,Daylight_range,
                 iteration_range,aspect_range,
                 bench_value,
                 toggle_2D_value,
                 toggle_daylight_invert,
                 clickData,
                 show_ga_data,
                 show_hypercube_data):
    if clickData is None:
        iteration_name = 4
    else:
        iteration_name = clickData["points"][0]['customdata'][0]
    dff = df.where((df['Building Embodied Carbon [kgCO2/m2]'] >= BEU_range[0]) & 
                   (df['Building Embodied Carbon [kgCO2/m2]'] <= BEU_range[1]) & 
                   
                   (df['Energy Use Intensity [kWh/m2]'] >= EUI_range[0]) & 
                   (df['Energy Use Intensity [kWh/m2]'] <= EUI_range[1]) & 
                   
                   (df['Daylight [%sDA]'] >= Daylight_range[0]) & 
                   (df['Daylight [%sDA]'] <= Daylight_range[1]) & 
                   
                   (df['Aspect Ratio'] >= aspect_range[0]) & 
                   (df['Aspect Ratio'] <= aspect_range[1]) & 
                   
                   (df['Itteration [n]'] >= iteration_range[0]) & 
                   (df['Itteration [n]'] <= iteration_range[1])).dropna()
    if show_ga_data==False:
        dff = dff.drop(dff[dff['Label']=='100LH - Optimization'].index)
    if show_hypercube_data==False:
        dff = dff.drop(dff[dff['Label']=='100LH - Hypercube'].index)
    
    dff_bench = df.where(df['Itteration [n]']==int(bench_value.split(' ')[-1]))
    dff_select_iter = df.where(df['Itteration [n]']==iteration_name)
    

    
    
    if toggle_2D_value==False:
        fig = px.scatter_3d(dff,x=xaxis_column_name,
                y=yaxis_column_name,
                z=zaxis_column_name,
                color=color_column_name,
                symbol = 'Label',
                custom_data =['Itteration [n]'],
                labels=None
                )


        

        fig.update_traces(marker_size=2.5, hoverinfo='none', hovertemplate=None,showlegend=False)
        fig.add_trace(go.Scatter3d(x=dff_bench[xaxis_column_name],
                        y=dff_bench[yaxis_column_name],
                        z=dff_bench[zaxis_column_name],
                        mode='markers',
                        marker_color='red',
                        hoverinfo='none',
                        hovertemplate=None,
                        showlegend =False,
                        ))
        
        fig.add_trace(go.Scatter3d(x=dff_select_iter[xaxis_column_name],
                        y=dff_select_iter[yaxis_column_name],
                        z=dff_select_iter[zaxis_column_name],
                        mode='markers',
                        marker_color='blue',
                        hoverinfo='none',
                        hovertemplate=None,
                        showlegend =False,
                        ))
        
        fig.update_layout(autosize=True,
                          scene=dict(
                              xaxis_title=xaxis_column_name,
                              yaxis_title=yaxis_column_name,
                              zaxis_title=zaxis_column_name,
                              xaxis= dict(gridcolor='grey'),
                              yaxis= dict(gridcolor='grey'),
                              zaxis= dict(gridcolor='grey')
                                  ),
                          hovermode='closest',
                          clickmode='event',
                          uirevision=1,
                          margin=dict(
                              l=0,
                              r=0,
                              b=0,
                              t=0),
                          
                      plot_bgcolor='#e0ebf4 ',
                      paper_bgcolor='#e0ebf4 ',
                          coloraxis_colorbar=dict(
                                len=0.95,
                                xanchor="right", x=1,
                                yanchor='middle', y=0.5,
                                thickness=10)
                          )
        x_eye = 1.25
        y_eye = -1.25
        z_eye = 1.25

        fig.update_layout(scene_camera_eye=dict(x=x_eye, y=y_eye, z=z_eye))
        if toggle_daylight_invert==True:
            if xaxis_column_name=='Daylight [%sDA]':
                fig.update_layout(
                        scene={
                            'xaxis': {'range':[df[xaxis_column_name].max(),df[xaxis_column_name].min()]}, # reverse daylight axis
                        }
                    )
            else:
                fig.update_layout(
                        scene={
                            'xaxis': {'range':[df[xaxis_column_name].min(),df[xaxis_column_name].max()]}, # reverse daylight axis
                        }
                    )
                
            if yaxis_column_name=='Daylight [%sDA]':
                fig.update_layout(
                        scene={
                            'yaxis': {'range':[df[yaxis_column_name].max(),df[yaxis_column_name].min()]}, # reverse daylight axis
                        }
                    )
            else:
                fig.update_layout(
                        scene={
                            'yaxis': {'range':[df[yaxis_column_name].min(),df[yaxis_column_name].max()]}, # reverse daylight axis
                        }
                    )
            

            if zaxis_column_name=='Daylight [%sDA]':
                fig.update_layout(
                        scene={
                            'zaxis': {'range':[df[zaxis_column_name].max(),df[zaxis_column_name].min()]}, # reverse daylight axis
                        }
                    )
            else:
                fig.update_layout(
                        scene={
                            'zaxis': {'range':[df[zaxis_column_name].min(),df[zaxis_column_name].max()]}, # reverse daylight axis
                        }
                    ) 
        else:
            fig.update_layout(
                    scene={
                        'xaxis': {'range':[df[xaxis_column_name].min(),df[xaxis_column_name].max()]}, # reverse daylight axis
                    }
                )
            

            fig.update_layout(
                    scene={
                        'yaxis': {'range':[df[yaxis_column_name].min(),df[yaxis_column_name].max()]}, # reverse daylight axis
                    }
                )
        

            fig.update_layout(
                    scene={
                        'zaxis': {'range':[df[zaxis_column_name].min(),df[zaxis_column_name].max()]}, # reverse daylight axis
                    }
                ) 


    else:
        fig = px.scatter(dff,x=xaxis_column_name,
                y=yaxis_column_name,
                color=color_column_name,
                custom_data =['Itteration [n]'],
                symbol = 'Label',
                labels=None
                )


        

        fig.update_traces(marker_size=5, hoverinfo='none', hovertemplate=None)
        fig.add_trace(go.Scatter(x=dff_bench[xaxis_column_name],
                        y=dff_bench[yaxis_column_name],
                        mode='markers',
                        marker_color='red',
                        hoverinfo='none',
                        hovertemplate=None,
                        showlegend =False,
                        marker=dict(
                            size=10,
                            )
                        )
                      )
        fig.add_trace(go.Scatter(x=dff_select_iter[xaxis_column_name],
                        y=dff_select_iter[yaxis_column_name],
                        mode='markers',
                        marker_color='blue',
                        hoverinfo='none',
                        hovertemplate=None,
                        showlegend =False,
                        marker=dict(
                            size=10,
                            )
                        ))
        
        fig.update_layout(autosize=True,
                          scene=dict(
                              xaxis_title=xaxis_column_name,
                              yaxis_title=yaxis_column_name,
                                  ),
                          hovermode='closest',
                          clickmode='event',
                          uirevision=1,
                          margin=dict(
                              l=0,
                              r=0,
                              b=0,
                              t=0),
                      plot_bgcolor='#e0ebf4 ',
                      paper_bgcolor='#e0ebf4 ',
                          coloraxis_colorbar=dict(
                                len=0.95,
                                xanchor="right", x=1,
                                yanchor='middle', y=0.5,
                                thickness=10),
                          showlegend=False)
        fig.update_xaxes(showgrid=True, gridcolor='grey')
        fig.update_yaxes(showgrid=True,  gridcolor='grey')
        if toggle_daylight_invert==True:
            if xaxis_column_name=='Daylight [%sDA]':
                fig.update_xaxes(range=[df[xaxis_column_name].max(),df[xaxis_column_name].min()])
            else:
                fig.update_xaxes(range=[df[xaxis_column_name].min(),df[xaxis_column_name].max()])
                
            if yaxis_column_name=='Daylight [%sDA]':
                fig.update_yaxes(range=[df[yaxis_column_name].max(),df[yaxis_column_name].min()])
            else:
                fig.update_yaxes(range=[df[yaxis_column_name].min(),df[yaxis_column_name].max()])
        else:
            
            fig.update_xaxes(range=[df[xaxis_column_name].min(),df[xaxis_column_name].max()])
            fig.update_yaxes(range=[df[yaxis_column_name].min(),df[yaxis_column_name].max()])

                    
    fig.data = fig.data[::-1]

    fig.layout.coloraxis.colorbar.title = color_column_name.replace(" ",'<br>')
    


    return fig

def np_image_to_base64(im_matrix):
    im = Image.fromarray(im_matrix)
    buffer = io.BytesIO()
    im.save(buffer, format="jpeg")
    encoded_image = base64.b64encode(buffer.getvalue()).decode()
    im_url = "data:image/jpeg;base64, " + encoded_image
    return im_url

@app.callback(
    Output("graph-tooltip", "show"),
    Output("graph-tooltip", "bbox"),
    Output("graph-tooltip", "children"),
    Input('crossfilter-indicator-scatter', "hoverData"),
    Input('crossfilter-benchmark','value'),

)
def display_hover(hoverData,bench_value):
    if hoverData is None:
        return False, no_update, no_update

    # demo only shows the first point, but other points may also be available
    pt = hoverData["points"][0]
    bbox = pt["bbox"]

    iteration_name = int(hoverData["points"][0]['customdata'][0])
    
    comp_val= iteration_name
    
    bench_val = int(bench_value.split(' ')[-1])

    titles = ['Building Embodied Carbon [kgCO2/m2]','Energy Use Intensity [kWh/m2]','Daylight [%sDA]']

    r = df[['Itteration [n]']+titles]
    r[titles] = r[titles]-r[titles].min()
    r[titles] = r[titles]/r[titles].max()
    r['Building Embodied Carbon [kgCO2/m2]'] = 1-r['Building Embodied Carbon [kgCO2/m2]']
    r['Energy Use Intensity [kWh/m2]'] = 1-r['Energy Use Intensity [kWh/m2]']
    r_bench = 100*r.loc[r['Itteration [n]'] == bench_val]
    r_comp = 100*r.loc[r['Itteration [n]'] == comp_val]
    
    
    titles = titles+titles[:1]
    r_bench=np.array(r_bench[titles]).squeeze()
    r_comp=np.array(r_comp[titles]).squeeze()


    fig = go.Figure()
    
    labels = ['<br>'.join(wrap(l, 10)) for l in titles]
    labels = ['BEC','EUI','Daylight','BEC']
    fig.add_trace(go.Scatterpolar(
          name = "Selected iteration",
          r = r_comp,
          theta = labels,
          hoverinfo='none',
        ))
    fig.add_trace(go.Scatterpolar(
          name = "Benchmark",
          r = r_bench,
          theta = labels,
          hoverinfo='none',
        ))

    
    
    fig.update_polars(radialaxis_showticksuffix="all",radialaxis_ticksuffix='%')
    fig.update_layout(polar = dict(radialaxis = dict(showticklabels = False)))

    
    fig.update_layout(autosize=True,
                      uirevision=1,
                      showlegend=False,
                      plot_bgcolor='rgba(205,222,238,0)',
                      paper_bgcolor='rgba(205,222,238,0)',
                      font=dict(
                          family="sans-serif",
                          size=10,
                          color="black"
                      ),
                      margin=dict(
                          l=10,
                          r=10,
                          b=10,
                          t=10))
    
    # Load image 
    if iteration_name>=it_GA_skip:
        img_path = image_format_GA.format(iteration_name-it_GA_skip)
        it_label = 'Genetic Algorithm'
    else:
        img_path = image_format_Hypercube.format(iteration_name)
        it_label = 'Latin Hypercube'
    try:
        image = Image.open(img_path)
    except:
        image=Image.open(image_format_GA.format(8))
    # encoded_image = encoded_image.decode()
    # image = image.crop((900,300,2400,2400))
    buff = io.BytesIO()
    image.save(buff, format='png')
    encoded_image = base64.b64encode(buff.getvalue()).decode("utf-8")
    im_url = "data:image/jpeg;base64, " + encoded_image
    

    children = [
        html.Div([
            html.B('Iteration ' f"{iteration_name}"' ('+it_label+')', style={"color": "darkblue", "overflow-wrap": "break-word",
                                                        'font-size':'0.9vw','display':'block','height':'1.5vh'}),
            html.Div([
              
                html.Img(src=im_url, style={"width": "8vw"}),
                
                
                
                ],style={'width':'8vw','display':'inline-block'}),
            html.Div([
                
                dcc.Graph(
                    figure=fig,
                    
                    clear_on_unhover=True,
                    responsive=True,
                    config={
                        'displayModeBar': False,
                    },
                    style={'height':'20vh','display':'block'},
                    
                ),
                
                
                ],style={'width':'14vw','height':'20vh','display':'inline-block',
                         'white-space':'normal'}),
            
            
        ]
            ,style={'width':'22vw','white-space':'normal','display':'inline-block',
                    'background-color':'rgba(164,174,184,0)','margin':'0','padding':'0'}),
    ]

    return True, bbox, children


#%%
tab_style = {
     'font-size':'1vw',
     'padding':'0px',
     'border':'1px solid', 
     'border-radius': 5,
     'margin-left':'0.2vw',
     'margin-right':'0.2vw',
     'line-height':'1.8vh',
     "white-space": "pre"
}

tab_selected_style = {
    'fontWeight': 'bold',
     'font-size':'1vw',
     'padding':'0.5px',
     'border':'1px solid',
     'border-radius': 5,
     'margin-left':'0.2vw',
     'margin-right':'0.2vw',
     'line-height':'1.8vh',
     "white-space": "pre"
}

app.layout = html.Div([
    html.Div([
        # DIV 1-1 : Filter parameter tabs
        html.Div([
                 
            html.Div([
                html.B('DAISY Design Space Navigator',style={'font-size':'1.4vw',
                                                               'padding':'1px','margin':'0px','margin-left':'0.5vw'}),
            html.Div([
                
                html.Div([
                    html.Div(id='benchmark-div')
                    ],
                    style={
                           'width':'19vw',
                           'display': 'inline-block',}),

                
                html.Div([
                    html.Div(id='comparison-div')
                    ],
                    style={'width':'19vw',
                           'display': 'inline-block',})
                
                ],
                
                style={'height':'31vh',
                       'display': 'inline-block'
                       }),  
                
                
                
                ],
                
                style={'margin-bottom':'10px',#'border':'1px solid', 'border-radius': 10, 'backgroundColor':'#FFFFFF'
                        }),  
            html.Div([
                dcc.Tabs([
                    dcc.Tab(label='Benchmark\nComparison', children=[
                        html.Div([
                            # html.H4('Comparison to maximum values', style={'font-size':'1.25vw',
                            #                                                'padding':'0px','margin':'0px'}),
                            dcc.Graph(
                                id='comparison-plot-div',
                                clear_on_unhover=True,
                                responsive=True,
                                config={
                                    'displayModeBar': False,
                                },
                                style={'height':'33vh'}
                            ),
                            ],
                            ),
                        
                        ],style=tab_style,selected_style=tab_selected_style),
                    dcc.Tab(label='Embodied\nCarbon', children=[
                        html.Div([
                            # html.H4('Comparison to maximum values', style={'font-size':'1.25vw',
                            #                                                'padding':'0px','margin':'0px'}),
                            dcc.Graph(
                                id='carbon-breakdown-plot-div',
                                clear_on_unhover=True,
                                responsive=True,
                                config={
                                    'displayModeBar': False,
                                },
                                style={'height':'33vh'}
                            ),
                            ],
                           ),
                        
                        ],style=tab_style,selected_style=tab_selected_style),
                    dcc.Tab(label='Energy\nUse', children=[
                        html.Div([
                            # html.H4('Comparison to maximum values', style={'font-size':'1.25vw',
                            #                                                'padding':'0px','margin':'0px'}),
                            dcc.Graph(
                                id='energy-breakdown-plot-div',
                                clear_on_unhover=True,
                                responsive=True,
                                config={
                                    'displayModeBar': False,
                                },
                                style={'height':'33vh'}
                            ),
                            ],
                            style={'white-space':'normal'}),
                        
                        ],style=tab_style,selected_style=tab_selected_style),

                dcc.Tab(label='Objective\nFilters', children=[
                    html.P('Building Embodied Carbon (kgCO2/m^2)',style={'font-size':'0.9vw',
                    'margin-top':'1vh'}),
                    
                    dcc.RangeSlider(df['Building Embodied Carbon [kgCO2/m2]'].min(), df['Building Embodied Carbon [kgCO2/m2]'].max(), 0.1, 
                                    value=[0, max(df['Building Embodied Carbon [kgCO2/m2]'])], 
                                    marks={str(i):str(i) for i in np.round(np.linspace(np.floor(df['Building Embodied Carbon [kgCO2/m2]'].min()),
                                                                        (np.ceil(df['Building Embodied Carbon [kgCO2/m2]'].max())),
                                                                        10),1)},
                                    
                                    id='BEU-slider'),
                    
                    html.P('Energy Use Intensity [kWh/m2]',style={'font-size':'0.9vw'}),
                    dcc.RangeSlider(np.floor(df['Energy Use Intensity [kWh/m2]'].min()), np.ceil(df['Energy Use Intensity [kWh/m2]'].max()), 0.1, 
                                    value=[0, max(df['Energy Use Intensity [kWh/m2]'])], 
                                    marks={str(i):str(i) for i in np.round(np.linspace(np.floor(df['Energy Use Intensity [kWh/m2]'].min()),
                                                                        np.ceil(df['Energy Use Intensity [kWh/m2]'].max()),
                                                                        10),1)},
                                    id='EUI-slider'),
                    
                    html.P('Daylight [%sDA]',style={'font-size':'0.9vw'}),
                    dcc.RangeSlider(np.floor(df['Daylight [%sDA]'].min()), np.ceil(df['Daylight [%sDA]'].max()), 0.1, 
                                    value=[0, max(df['Daylight [%sDA]'])], 
                                    marks={str(i):str(i) for i in np.round(np.linspace(np.floor(df['Daylight [%sDA]'].min()),
                                                                        np.ceil(df['Daylight [%sDA]'].max()),
                                                                        10),1)},
                                    id='Daylight-slider'),
                    ],style=tab_style,selected_style=tab_selected_style),
                
                dcc.Tab(label='Gene\nFilters', children=[
                    html.P('Iteration number',style={'font-size':'0.9vw',
                    'margin-top':'1vh'}),
                    dcc.RangeSlider(0, max(df['Itteration [n]']), 10, 
                                    value=[0, max(df['Itteration [n]'])], 
                                    marks={str(i):str(i) for i in range(0,max(df['Itteration [n]']),1000)},
                                    id='iteration-slider'),
                    
                    
                    html.P('Aspect Ratio',style={'font-size':'0.9vw'}),
                    dcc.RangeSlider(np.floor(df['Aspect Ratio'].min()), np.ceil(df['Aspect Ratio'].max()), 0.1,
                                    value=[0, max(df['Aspect Ratio'])], 
                                    marks={str(i):str(i) for i in np.round(np.linspace(np.floor(df['Aspect Ratio'].min()),
                                                                        np.ceil(df['Aspect Ratio'].max()),
                                                                        10),1)},
                                    id='Aspect Ratio-slider'),
                    ],style=tab_style,selected_style=tab_selected_style),
                
                ])
                ],
                
                style={'height':'39vh',"overflow-y": "auto",'overflow-x':'hidden','margin-left':'0.5vw',
                       'margin-right':'1vw'
                       }),
        ], style={'width': '38vw',
        'height':'96vh',
        'display': 'inline-block',
        'vertical-align': 'top',
        'margin': '1px',
        'background-color':'#A4AEB8',
           'border':'1px solid', 'border-radius': 10
        }),
                  
        
        # DIV 1-2 : 3D plot

                
        html.Div([
            html.Div([
                       
                    html.Div([
                        html.P('X-axis plotted feature',style={'font-size':'1vw'}),
                        dcc.Dropdown(
                            df.columns,
                            'Energy Use Intensity [kWh/m2]',
                            id='crossfilter-xaxis-column',
                            clearable = False,
                            style={'font-size':'0.75vw'})
                        ],
                        style={'width': '20%', 'display': 'inline-block'}),
                    
                    html.Div([
                        html.P('Y-axis plotted feature',style={'font-size':'1vw'}),
                        dcc.Dropdown(
                            df.columns,
                            'Building Embodied Carbon [kgCO2/m2]',
                            id='crossfilter-yaxis-column',
                            clearable = False,
                            style={'font-size':'0.75vw'}
                            )
                        ],
                        style={'width': '20%', 'display': 'inline-block'}),  
                    
                    html.Div([
                        html.P('Z-axis plotted feature',style={'font-size':'1vw'}),
                        dcc.Dropdown(
                            df.columns,
                            'Daylight [%sDA]',
                            id='crossfilter-zaxis-column',
                            clearable = False,
                            style={'font-size':'0.75vw'}
                            )
                        ],
                        style={'width': '20%', 'display': 'inline-block'}),
                    html.Div([
                        html.P('Color scale feature',style={'font-size':'1vw'}),
                        dcc.Dropdown(
                            df.columns,
                            'Aspect Ratio',
                            id='crossfilter-colaxis-column',
                            clearable = False,
                            style={'font-size':'0.75vw'}),
                        ],
                        style={'width': '20%', 'display': 'inline-block'}),
                    html.Div([
                        html.P('Benchmark Structure',style={'font-size':'1vw'}),
                        dcc.Dropdown(
                            options = ['Iteration - '+str(i) for i in df['Itteration [n]']],
                            value = 'Iteration - 0',
                            id='crossfilter-benchmark',
                            clearable = False,
                            style={'font-size':'0.75vw'}
                            )
                        ],
                        style={ 'width': '20%', 'display': 'inline-block'}),
                    
                
                ],
                style={'height': '10vh', 'display': 'in-line block'}),
            html.Div([
                dcc.Graph(
                    id='crossfilter-indicator-scatter',
                    clear_on_unhover=True,
                    responsive=True,
                    config={
                        'responsive': True, # Doesn't seem to help
                        'displayModeBar': False,
                    },
                    style={'height': '83vh',}
                ),
                dcc.Tooltip(id="graph-tooltip",
                            direction='bottom',
                            style={'background-color':'rgba(164,174,184,0.85)',
                                   'border':'1px solid', 'border-radius': 10}),
                ]),
            # PARETO FRONT CODE REQUIRES WORK FOR PLOTTING
            # html.Div([
            # daq.ToggleSwitch(
            #     id='show-pareto-front',
            #     value=False,
            #     label={'label':'Display Pareto Front','style':{'font size':'1vw'}},
            #     labelPosition='right',
            #     size=30,
            #     color='red',
            #     theme = 'dark'
            #         )
            # ],
            # style={'float':'right','height': '5vh','margin-right':'1vw'}),
            # html.Div([
            # daq.ToggleSwitch(
            #     id='smooth-pareto-front',
            #     value=False,
            #     label={'label':'Smooth Pareto Front','style':{'font size':'1vw'}},
            #     labelPosition='right',
            #     size=30,
            #     color='red',
            #     theme = 'dark'
            #         )
            # ],
            # style={'float':'right','height': '5vh','margin-right':'1vw'}),
            html.Div([
            daq.ToggleSwitch(
                id='show-ga-data',
                value=True,
                label={'label':'Genetic Algorithm data','style':{'font size':'1vw'}},
                labelPosition='right',
                size=30,
                color='blue',
                theme = 'dark'
                    )
            ],
            style={'float':'left','height': '5vh','width':'25%','margin-right':'1vw'}),
            html.Div([
            daq.ToggleSwitch(
                id='show-hypercube-data',
                value=True,
                label={'label':'Hypercube data','style':{'font size':'1vw'}},
                labelPosition='right',
                size=30,
                color='blue',
                theme = 'dark'
                    )
            ],
            style={'float':'left','height': '5vh','width':'20%','margin-right':'1vw','margin-left':0}),
            
            html.Div([
            daq.ToggleSwitch(
                id='invert-daylight-toggle',
                value=True,
                label={'label':'Invert daylight axis','style':{'font size':'1vw'}},
                labelPosition='right',
                size=30,
                color='red',
                theme = 'dark'
                    )
            ],
            style={'float':'right','height': '5vh','width':'25%','margin-right':'1vw'}),
            html.Div([
            daq.ToggleSwitch(
                id='2d-toggle',
                value=False,
                label={'label':'2d Mode','style':{'font size':'1vw'}},
                labelPosition='right',
                size=30,
                color='red',
                theme = 'dark'
                    )
            ],
            style={'float':'right','height': '5vh','width':'15%','margin-right':'1vw'}),
            
            
            
            
            
        ],
            style={'width': '60%',
                   'height': '97vh',
                   'display': 'block',
                   'vertical-align': 'top',
                   'float': 'right',
                   'margin':'0px',
                   'padding':'0px',
                   'background-color':'transparent'}),
        
    ],
    
    style={'width': '99.5%',
           'height':'100%',
           'display': 'block',
           'vertical-align': 'middle',
           'float': 'left',
           'background-color':'#e0ebf4',
           'border':'1px solid', 'border-radius': 10,
                         "overflow-y": "hidden",'overflow-x':'hidden',
}),
   
    ],
                  style={
                         'margin':'1px',
                         'padding':'1px',
                         'display': 'block',
                         'position': 'relative',
                         'border-radius': 0,
                         "overflow-y": "hidden",'overflow-x':'hidden',
                         'vertical-align': 'middle',
                         }
    )





if __name__ == '__main__':
    app.run_server(debug=False)
