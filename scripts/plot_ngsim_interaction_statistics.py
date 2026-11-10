"""Figures accompanying the fixed-role NGSIM descriptive statistics."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def plot_statistics(events,density,output):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
        'axes.grid':True,'grid.alpha':.2})
    figures=[]
    def new(name,title):
        fig,ax=plt.subplots(figsize=(10,5.7),layout='constrained')
        ax.set_title(title,pad=14);figures.append((name,fig));return fig,ax
    def boxes(name,title,columns,labels,ylabel):
        fig,ax=new(name,title)
        values=[events.get(column,[]).dropna().to_numpy() if column in events else np.array([]) for column in columns]
        ax.boxplot(values,tick_labels=labels,showfliers=True)
        ax.set_ylabel(ylabel)
        fig.supxlabel(f'{len(events)} accepted fixed-role training events; one initial state per event.',fontsize=9)
    if len(events):
        boxes('01_four_vehicle_speeds','Fixed-role initial longitudinal speeds',
            [role+'_speed_mps' for role in ('elcv','rfv','rlv_current','rlv_target')],
            ['ELCV','RFV','Current leader','Target leader'],'Speed (m/s)')
        boxes('02_center_gaps','Initial longitudinal center-to-center gaps',
            [key+'_center_gap_m' for key in ('rear','current_front','target_front')],
            ['ELCV to RFV','Current leader to ELCV','Target leader to ELCV'],'Center gap (m)')
        boxes('03_net_clearances','Initial bumper-to-bumper longitudinal clearances',
            [key+'_clearance_m' for key in ('rear','current_front','target_front')],
            ['ELCV / RFV','Current leader / ELCV','Target leader / ELCV'],'Longitudinal clearance (m)')
        boxes('04_relative_speeds','Relative speed: positive means longitudinal closing',
            [key+'_closing_speed_mps' for key in ('rear','current_front','target_front')],
            ['RFV minus ELCV','ELCV minus current leader','ELCV minus target leader'],'Closing speed (m/s)')
        fig,ax=new('05_joint_speed_rear_gap','Joint initial speed and rear gap; preserve observed combinations')
        points=ax.scatter(events.elcv_speed_mps,events.rear_center_gap_m,
            c=events.rear_closing_speed_mps,cmap='coolwarm',edgecolors='#334155',linewidths=.4,s=50)
        fig.colorbar(points,ax=ax,label='RFV minus ELCV speed (m/s)')
        ax.set(xlabel='ELCV speed (m/s)',ylabel='ELCV to RFV center gap (m)')
        if 'lateral_10_90_duration_s' in events:
            values=events.lateral_10_90_duration_s.dropna().to_numpy()
            fig,ax=new('06_lane_change_duration',f'Observed 10–90% lateral excursion duration (n={len(values)})')
            ax.hist(values,bins=min(15,max(3,len(values)//3)),color='#8b5cf6',edgecolor='white')
            ax.set(xlabel='10–90% duration (s)',ylabel='Events')
            fig.supxlabel('Diagnostic endpoint-based duration, not complete maneuver start-to-finish time.',fontsize=9)
    fig,ax=new('07_density_speed','Observed training-recording traffic states in fixed 200 m ROI')
    ax.scatter(density.density_veh_per_km_per_lane,density.mean_reported_speed_mps,
        alpha=.3,s=14,color='#2563eb')
    ax.set(xlabel='Density (vehicles/km/lane)',ylabel='Mean reported speed in ROI (m/s)')
    fig.supxlabel('Six mainline lanes; one-second snapshots; adjacent samples share traffic. Not MetaDrive density.',fontsize=9)
    with PdfPages(output/'fixed_role_statistics.pdf') as pdf:
        for name,fig in figures:
            fig.savefig(output/(name+'.png'),dpi=180)
            fig.savefig(output/(name+'.svg'));pdf.savefig(fig);plt.close(fig)
