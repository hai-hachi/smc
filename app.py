"""Launch with: streamlit run app.py"""
from dataclasses import replace, asdict
import json
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from smc_lab.core import Config, CONTROLLERS, PATHS, CASES, make_path, simulate, batch

st.set_page_config(page_title='SMC | Control Laboratory', page_icon='🧭', layout='wide')
st.markdown('''<style>
.block-container {padding-top:1.3rem;max-width:1600px}
[data-testid="stMetric"] {background:rgba(120,130,150,0.07);padding:12px;border-radius:10px}
</style>''',unsafe_allow_html=True)
st.title('SMC · Differential-Drive Control Lab')
st.caption('4 SMC laws × 4 start conditions × 4 paths · Python kinematic simulation · physical velocity/acceleration limits')

with st.sidebar:
    st.header('Experiment setup')
    reference_mode=st.radio('Reference model',('closest_lookahead','trajectory'),
        format_func=lambda x:'Closest point + lookahead' if x=='closest_lookahead' else 'Time-based trajectory',
        help='Closest-point target is robot-dependent. Nominal vR,wR are feedforward, not exact target derivatives.')
    path=st.selectbox('Path',PATHS,index=3)
    case=st.selectbox('Starting pose',tuple(CASES)+('Custom',),index=1)
    default=CASES.get(case,(0.,0.,0.))
    dx=st.number_input('Longitudinal offset dx (m)',value=float(default[0]),step=0.25)
    dy=st.number_input('Lateral offset dy (m)',value=float(default[1]),step=0.25)
    dth=st.slider('Heading offset (deg)',-180.,180.,float(default[2]),5.)
    st.subheader('Reference and dynamics')
    speed=st.slider('Nominal reference speed (m/s)',0.0,1.0,0.40,0.01)
    lookahead=st.slider('Lookahead L (m)',0.0,3.0,0.60,0.05)
    duration=st.slider('Duration (s)',5.,90.,30.,1.)
    dt=st.select_slider('Controller dt (s)',options=[0.01,0.02,0.05,0.1],value=0.05)
    with st.expander('Robot motion limits',expanded=False):
        vmax=st.slider('Max linear speed (m/s)',0.1,2.,0.60,0.05)
        wmax=st.slider('Max angular speed (rad/s)',0.1,5.,1.50,0.1)
        amax=st.slider('Max linear accel (m/s²)',0.1,5.,1.0,0.1)
        alpha=st.slider('Max angular accel (rad/s²)',0.1,10.,2.0,0.2)
        reverse=st.checkbox('Allow reverse (full SMC only)',value=False)
        slowdown=st.slider('Endpoint slowdown length (m)',0.,3.,0.75,0.05)
    with st.expander('SMC tuning',expanded=True):
        ly=st.slider('lambda_y (1/m)',0.1,8.,2.5,0.1)
        lx=st.slider('lambda_x (1/m)',0.,3.,0.6,0.1)
        ks=st.slider('k_s (1/s)',0.1,6.,1.2,0.1)
        kx=st.slider('k_x (1/s)',0.1,4.,0.8,0.1)
        krho=st.slider('k_rho (1/s)',0.1,3.,0.8,0.1)
        stop=st.slider('Heading stop threshold (deg)',10.,180.,70.,5.)
        guard=st.number_input('Singularity threshold |1+lambda_y ex|',0.001,1.,0.06,0.01)
    with st.expander('Path geometry',expanded=True):
        length=st.slider('Path length (m)',4.,30.,14.,0.5)
        rad=st.slider('Circle radius (m)',0.3,8.,2.,0.1)
        zz_ang=st.slider('Zigzag heading change (deg)',5.,165.,75.,5.)
        zz_leg=st.slider('Zigzag leg length (m)',0.5,7.,2.5,0.25)
        zz_turns=st.slider('Zigzag turns',1,12,5)
        zz_radius=st.slider('Zigzag fillet radius (m)',0.05,2.,0.35,0.05)
        smooth_min=st.slider('Smooth min local radius (m)',0.25,5.,0.9,0.05)
        smooth_max=st.slider('Smooth max local radius (m)',0.5,15.,4.,0.25)
        smooth_period=st.slider('Smooth curvature cycle length (m)',2.,20.,6.,0.5)
    with st.expander('Evaluation thresholds',expanded=False):
        settle_d=st.slider('Position tolerance (m)',0.02,1.,0.15,0.01)
        settle_th=st.slider('Heading tolerance (deg)',2.,45.,10.,1.)
        hold=st.slider('Required settle hold (s)',0.,5.,1.5,0.25)
        warmup=st.slider('RMSE warmup (s)',0.,15.,3.,0.5)

cfg=Config(path=path,reference_mode=reference_mode,dx0=dx,dy0=dy,dtheta_deg=dth,
    duration=duration,dt=dt,v_ref=speed,lookahead=lookahead,length=length,
    circle_radius=rad,zigzag_angle_deg=zz_ang,zigzag_leg=zz_leg,zigzag_turns=zz_turns,
    zigzag_radius=zz_radius,smooth_radius_min=smooth_min,smooth_radius_max=smooth_max,
    smooth_period=smooth_period,lambda_y=ly,lambda_x=lx,k_s=ks,k_x=kx,k_rho=krho,
    heading_stop_deg=stop,v_max=vmax,w_max=wmax,a_max=amax,alpha_max=alpha,
    allow_reverse=reverse,slowdown_dist=slowdown,settle_distance=settle_d,
    settle_heading_deg=settle_th,settle_hold=hold,warmup=warmup,singular_guard=guard)
try:
    cfg.validate()
except ValueError as exc:
    st.error(str(exc));st.stop()

st.info('The **truncated** laws deliberately use ωR = 0 and omit the 1 + λy ex denominator. '
    'The **full** laws include ωR = vR κ and the nonlinear denominator; singularities are reported, not hidden. '
    '**None of the four includes an SMC saturation or sign switching term.** Physical speed clamps remain.')
if reference_mode=='closest_lookahead':
    st.warning('Closest point + lookahead is a geometric reference, not a feasible time-parameterized trajectory. '
        'Full-law reaching proofs do **not** apply directly; vR and ωR here are only nominal feedforward. '
        'A nonzero reference-position error may be normal when lookahead > 0.')

chosen=st.multiselect('Controllers to compare',CONTROLLERS,default=list(CONTROLLERS))
if not chosen:
    st.warning('Select at least one controller');st.stop()
path_obj=make_path(cfg)
runs=[simulate(cfg,name,path_obj) for name in chosen]

st.markdown('### Trajectory and reference')
fig=go.Figure()
fig.add_trace(go.Scatter(x=path_obj.x,y=path_obj.y,mode='lines',line=dict(color='#8b93a8',width=4,dash='dot'),name='Fixed path'))
for run in runs:
    h=run.history
    if not len(h['x']):continue
    fig.add_trace(go.Scatter(x=h['x'],y=h['y'],mode='lines',name=run.controller,
        customdata=np.column_stack([h['t'],h['path_error'],h['etheta']]),
        hovertemplate='x=%{x:.2f} m, y=%{y:.2f} m<br>t=%{customdata[0]:.2f} s<br>Path error=%{customdata[1]:.3f} m<br>Heading error=%{customdata[2]:.2f} rad<extra>%{fullData.name}</extra>'))
    fig.add_trace(go.Scatter(x=[h['x'][0]],y=[h['y'][0]],mode='markers',marker=dict(size=9,symbol='circle-open'),
        showlegend=False,name='Start'))
    fig.add_trace(go.Scatter(x=[h['x'][-1]],y=[h['y'][-1]],mode='markers',marker=dict(size=10,symbol='x'),
        showlegend=False,name='End'))
fig.update_layout(height=480,margin=dict(l=10,r=10,b=15,t=15),xaxis_title='X (m)',yaxis_title='Y (m)',
    legend=dict(orientation='h',y=-0.2),yaxis_scaleanchor='x',hovermode='closest')
st.plotly_chart(fig,use_container_width=True)

rows=[{'Controller':r.controller,**r.metrics} for r in runs]
results=pd.DataFrame(rows)
st.markdown('### Performance comparison')
important=['Controller','status','path_rmse_m','max_path_m','final_path_m','reference_rmse_m',
           'final_ex_m','final_ey_m','final_etheta_deg','settle_s','converged',
           'travel_m','angular_effort_rad','limited_pct','min_abs_den']
st.dataframe(results[important].round(4),use_container_width=True,hide_index=True)
st.download_button('Download current metrics CSV',data=results.to_csv(index=False),file_name='smc_metrics.csv',mime='text/csv')

names=[r.controller for r in runs]
st.markdown('### Time histories')
tab_error,tab_cmd,tab_slide,tab_ref=st.tabs(['Tracking errors','Velocity commands','Sliding surfaces','Reference / geometry'])

def lines(runset,series,labels,units):
    fig=make_subplots(rows=len(series),cols=1,shared_xaxes=True,vertical_spacing=0.08,
                      subplot_titles=labels)
    for idx,key in enumerate(series):
        for r in runset:
            h=r.history
            if not len(h['t']):continue
            y=h[key]*180/math.pi if units[idx]=='deg' else h[key]
            fig.add_trace(go.Scatter(x=h['t'],y=y,mode='lines',name=r.controller,
                legendgroup=r.controller,showlegend=idx==0),row=idx+1,col=1)
        fig.update_yaxes(title_text=units[idx],row=idx+1,col=1)
    fig.update_xaxes(title_text='Simulation time (s)',row=len(series),col=1)
    fig.update_layout(height=250*len(series),margin=dict(l=20,r=20,b=20,t=40),
                      legend=dict(orientation='h',y=-0.12))
    return fig

with tab_error:
    st.plotly_chart(lines(runs,['ex','ey','etheta','path_error'],['Longitudinal ex','Lateral ey','Heading eθ','Geometric path error'],['m','m','deg','m']),use_container_width=True)
with tab_cmd:
    st.plotly_chart(lines(runs,['v','omega','desired_v','desired_omega'],['Applied linear speed','Applied angular speed','Unclamped requested linear speed','Unclamped requested angular speed'],['m/s','rad/s','m/s','rad/s']),use_container_width=True)
with tab_slide:
    st.caption('For the one-surface full law: s1 = eθ + λy ey + λx ex. For two-surface full law: s1 = ex, s2 = eθ + λy ey. Truncated versions show s2 only.')
    st.plotly_chart(lines(runs,['s1','s2','denom'],['Sliding variable s1','Sliding variable s2','1 + λy ex denominator'],['mixed','mixed','-']),use_container_width=True)
with tab_ref:
    st.plotly_chart(lines(runs,['vR','omegaR','reference_s','path_s'],['Nominal reference linear speed','Nominal angular feedforward','Reference arc position','Closest-point arc position'],['m/s','rad/s','m','m']),use_container_width=True)

with st.expander('Download full data and save/replay comparison'):
    all_hist=[]
    for r in runs:
        frame=pd.DataFrame(r.history)
        frame.insert(0,'controller',r.controller)
        all_hist.append(frame)
    st.download_button('Download trajectory histories CSV',pd.concat(all_hist).to_csv(index=False),
                       file_name='smc_trajectories.csv',mime='text/csv')
    st.download_button('Download configuration JSON',json.dumps(asdict(cfg),indent=2),
                       file_name='smc_config.json',mime='application/json')
    if 'saved_comparisons' not in st.session_state:st.session_state.saved_comparisons=[]
    if st.button('Save this comparison to session'):
        st.session_state.saved_comparisons.append({'config':asdict(cfg),'metrics':rows})
        st.success('Saved in this browser session (not on disk).')
    saved=st.session_state.saved_comparisons
    st.caption(f'{len(saved)} comparisons saved in this browser session')
    if saved:
        st.download_button('Export saved comparisons JSON',json.dumps(saved,indent=2,default=str),
                           file_name='smc_saved_comparisons.json',mime='application/json')
        if st.button('Clear saved comparisons'):st.session_state.saved_comparisons=[];st.rerun()

st.divider()
st.markdown('### Batch benchmark (4 paths × 4 starting poses × 4 controllers)')
st.caption('64 simulations for the current parameters and reference model. See failing / singular runs rather than filtering them out.')
if st.button('Run all 64 cases',type='primary'):
    with st.spinner('Running benchmark...'):
        table=pd.DataFrame(batch(cfg))
    st.session_state['batch_table']=table
if 'batch_table' in st.session_state:
    table=st.session_state['batch_table']
    st.dataframe(table.round(4),use_container_width=True,hide_index=True)
    chart=go.Figure()
    for name in CONTROLLERS:
        sub=table[table['controller']==name]
        chart.add_trace(go.Bar(name=name,x=[f"{p} / {c}" for p,c in zip(sub['path'],sub['case'])],
                               y=sub['path_rmse_m']))
    chart.update_layout(barmode='group',height=460,yaxis_title='Path RMSE (m)',
                        xaxis_tickangle=-35,margin=dict(l=15,r=15,b=25,t=15))
    st.plotly_chart(chart,use_container_width=True)
    st.download_button('Download all 64 metrics CSV',table.to_csv(index=False),
                       file_name='smc_batch_64.csv',mime='text/csv')

st.caption('Kinematic simulation only. No motor PID, dynamics, wheel slip, localization noise, or Nav2/Gazebo integration.')
