import pandas as pd, os, numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from ILWFP_Main.sqlConnector import engine
from ILWFP_Main.grapher.setupTypeAndColorTables import getCropTypeColors
import seaborn as sns
import joypy

cropTypesColors = getCropTypeColors()

# pd.read_sql("SELECT * FROM crop_type", engine).columns

sql  = """SELECT afwu.*, wt.water_name, kcz.zone_name, cc.crop_category_name, ct.crop_name
	FROM ilwfp2018.agrifield_wateruse_aes_kc_moag_r afwu, ilwfp2018.agrifield af,
	    water_type wt, kc_et_zone kcz, crop_category cc, crop_type ct
	WHERE afwu.agrifield_id = af.agrifield_id AND af.agrifield_id=afwu.agrifield_id 
	AND af.water_type_id = wt.water_id AND af.et_zone_id = kcz.et_zone_id 
	AND af.crop_category_id=cc.crop_category_id AND af.crop_id = ct.crop_id
	
"""
df_sql = pd.read_sql(sql, engine)
# df_sql.columns
df_fw = df_sql.query("water_name == 'Freshwater'")
df_orchard = df_fw.loc[df_fw['crop_category_name'].isin(['Citrus','Plantations'])]

df_sql.loc[df_sql.crop_name=='Orange'].groupby('water_name')['dunam'].sum()
## citrus crop_type as crop_display_name
df_orchard['crop_display_name'] = [y if cc =='Citrus' else x for x,y,cc in
                                   zip(df_orchard.crop_display_name, df_orchard.crop_name, df_orchard.crop_category_name)]
# #test
# df_orchard.loc[df_orchard['total_m3']==0, 'dunam'].sum()

#crops area
crops_area_df = df_orchard.groupby('crop_display_name')['dunam'].sum().sort_values(ascending=False)
# print(crops_area_df)

##selected crops
select_crops = list(crops_area_df[crops_area_df>10000].index)
## crops to remove (Wine Grapes do not have Kc values
for c in ['Other Plantations','Wine Grapes']:
    select_crops.remove(c)
# select_crops = ['Citrus','Olive Oil', 'Avocado','Apple','Mango','Banana', 'Almond',
#          'Table Grapes','Peach','Olive','Plum','Nectarine']

df_national_select = df_orchard.loc[df_orchard.crop_display_name.isin(select_crops)]
def getSelectedCropsAboveXha(select_crops=select_crops, ha=30):
    df_select = pd.DataFrame()
    ## remove if zone has less than x and create df of selected crops
    for c in select_crops:
        print(c)
        df_crop = df_orchard.loc[df_orchard.crop_display_name==c]
        ha_df = df_crop.groupby('zone_name')['dunam'].sum()/10
        valid_zones = ha_df[ha_df>=ha].index
        df_c_filter = df_crop.loc[df_crop.zone_name.isin(valid_zones)]
        df_select = df_select.append(df_c_filter, ignore_index=True)

    return df_select
df_select = getSelectedCropsAboveXha()

## indices agg funcs
agg_d = {'m3_per_ha':['mean','std'],
         'm3_per_yield':['mean','std'],
         'profit_a_per_ha':['mean','std'],
         'profit_b_per_ha':['mean','std'],
         'profit_a_per_m3':['mean','std'],
         'profit_b_per_m3':['mean','std'],
         'ha':'sum',
         'mcm':'sum',
         'total_profit_mil_a':'sum',
         'total_profit_mil_b':'sum'}

##create indices cols - convert from NIS to USD
def assignNewIndices(df, usd_to_nis=3.2):
    df['profit_a_per_dunam'] = df['profit_a_per_dunam'] / usd_to_nis
    df['profit_b_per_dunam'] = df['profit_b_per_dunam'] / usd_to_nis
    df = df.assign(ha=df['dunam'] / 10,
                   mcm=df['total_m3'] / 10 ** 6,
                   m3_per_ha=df['m3_per_dunam'] * 10,
                   total_profit_mil_a=(df['total_profit_a'] / 10 ** 6) / usd_to_nis,
                   total_profit_mil_b=df['total_profit_b'] / 10 ** 6 / usd_to_nis,
                   profit_a_per_ha=df['profit_a_per_dunam'] * 10,
                   profit_b_per_ha=df['profit_b_per_dunam'] * 10,
                   profit_a_per_m3=df['total_profit_a'] / df['total_m3'],
                   profit_b_per_m3=df['total_profit_b'] / df['total_m3']).sort_index(ascending=False)
    return df
df_national_select = assignNewIndices(df_national_select)
df_select = assignNewIndices(df_select)

## add color
df_select = df_select.merge(cropTypesColors, on= 'crop_display_name', how='left')
df_national_select = df_national_select.merge(cropTypesColors, on= 'crop_display_name', how='left')

## convert ET ZONE Names
df_select.zone_name = ['Acre-Carmel Coast' if x=='Akko-Carmel Coast' else
                       "Newe Ya'ar" if x=='Neve Yaar' else
                       'Besor Farm' if x == 'Havat Habesor' else
                       'Eden Farm-Bikaa' if x == 'Havat Eden-Bikaa' else
                       'Lachish' if x == 'Lakhish'
                       else x for x  in df_select.zone_name]

# df_select.loc[df_select.color.isna(), 'crop_display_name'].unique()
##test
# df_select.groupby(['crop_display_name','zone_name'])['ha'].sum()

## CALCULATE comparision to average
def createRegionalComparisionTable(df_select=df_select):
    ##national average
    crop_nat_avg = df_select.groupby('crop_display_name')['m3_per_dunam'].mean()
    #regional average
    crop_reg_avg = df_select.groupby(['crop_display_name','zone_name'])['m3_per_dunam'].mean()
    ##pivot table
    crop_reg_p = pd.pivot(crop_reg_avg.reset_index(), index='crop_display_name', columns='zone_name')
    crop_reg_p.columns = crop_reg_p.columns.droplevel()
    crop_reg_p.columns.name = None
    crop_reg_p.index.name = None

    crop_reg_c = pd.DataFrame()

    # x = crop_reg_p.iloc[0]
    # crop = x.name
    # avg = crop_nat_avg[crop]
    # compareToNationalAverage(x, crop_nat_avg)



    def compareToNationalAverage(x, crop_nat_avg):
        crop = x.name
        # print(crop)
        avg = crop_nat_avg[crop]
        # print(avg)
        result = x/avg
        # print(result)
        return result

    for i, row in crop_reg_p.iterrows():
        #divide zone average by annual average
        result = compareToNationalAverage(row, crop_nat_avg)

        ## remove values if less thnan 50 ha - moved to full table

        # crop = result.name
        # print(crop)
        # area_srs = df_select.loc[df_select['crop_display_name']==crop].groupby('zone_name')['ha'].sum()
        # lt50_zones = area_srs[area_srs<30].index
        # result.loc[result.index.isin(lt50_zones)]
        # print("Zones to remove: "+ lt50_zones)

        ##add to list
        crop_reg_c = crop_reg_c.append(result)
    return crop_reg_c
crop_reg_c= createRegionalComparisionTable()
### GRAPHS
savefolder = 'C:/Users/eliav.ARO/OneDrive - ARO Volcani Center/IsraelWFP/ILWFP-Article/graphs/'
## NATIONAL GRAPHS

def createFolders(savefolder, crops=select_crops):
    try:
        os.mkdir(savefolder+'national')
    except:
        pass
    try:
        os.mkdir(savefolder+'crops')
    except:
        pass

    for crop in crops:
        try:
            os.mkdir(savefolder + 'crops/' + crop)
        except:
            pass
createFolders(savefolder)
df_g = df_select.groupby(['crop_display_name','color']).agg(agg_d).dropna().sort_index(ascending=False)


def joyDistributionGraph(column='m3_per_ha', df_joy=df_national_select, crops=select_crops):#df_orchard=df_orchard, ):
    # df_joy = df_orchard.loc[df_orchard.crop_display_name.isin(crops)]
    ## crop color
    df_crop = df_joy[['crop_display_name','color']].drop_duplicates().sort_values(by='crop_display_name').reset_index(drop=True)
    # df_crop = df_crop.merge(cropTypesColors, on= 'crop_display_name').reset_index(drop=True)
    cmap = colors.ListedColormap(df_crop['color'])
    if column in ['m3_per_ha','dual']:
        df_joy = df_joy.assign(m3_per_ha=df_joy.m3_per_dunam *10)
    # if column == 'dual':
        
    ### distribution of m3_per_dunam
    fig, axes = joypy.joyplot(df_joy, column=column, overlap=1, by="crop_display_name",
                            ylim='own', fill=True, figsize=(6, 6), legend=False, xlabels=True,
                            ylabels=True, alpha=.7, linewidth=.5, colormap=cmap)
    if column == 'm3_per_dunam':
        xlabel = 'Water demand (m$^3$/dunam)'
    elif column == 'm3_per_ha':
        xlabel = 'Water demand (m$^3$/ha)'
    elif column == 'm3_per_yield':
        xlabel = 'Water footprint (m$^3$/ton)'
    else:
        xlabel = 'Unknown'
    # ax.yaxis.set_label_position("right")
    plt.xlabel(xlabel)

    axes[int(np.round(len(axes)/2))].set_ylabel('Number of parcels per crop', labelpad=35)
    crops.sort()
    ##add average
    for i, row in df_crop.iterrows():
        c = row.crop_display_name
        print(c)
        c_mean = df_joy.loc[df_joy.crop_display_name == c, column].mean()
        axes[i].axvline(c_mean,color='black', lw=1, linestyle='--', alpha=.5, ymin=0.05, ymax=.35, zorder=50)
        # axes[i].get_ylim()
    # ax = axes[12]
    # ax.axvline(c_mean, color='black', lw=1, linestyle='--', alpha=.8, ymin=.05, ymax=.3)

    len(crops)
    fig.savefig('{}national/joydist_{}.jpg'.format(savefolder,column), dpi=300, bbox_inches='tight')
for col in ['m3_per_dunam', 'm3_per_ha','m3_per_yield']:
    joyDistributionGraph(column=col)
print("High water demand = {} m3/ha; Low water demand {} m3/ha".format(df_national_select['m3_per_ha'].max(), df_national_select['m3_per_ha'].min()))
print("High WF = {} m3/ton; Low WF {} m3/ton".format(df_national_select['m3_per_yield'].max(), df_national_select['m3_per_yield'].min()))
df_national_select.groupby('crop_display_name')['m3_per_ha'].mean().sort_values()
def nationalBarHCrops(df, y, xlabel, savename, std=True):
    fig, ax = plt.subplots(figsize=(6,6))
    df = df.reset_index(level=1)
    if std:
        ax.barh(df.index,df[y], xerr=df['std'], align='center', alpha=0.7, ecolor='black',
                capsize=2, color=df['color'], error_kw=dict(alpha=0.4, size=1, linewidth=1))
    else:
        ax.barh(df.index, df[y], align='center', alpha=0.7, color=df['color'])
    ## add 0 yline
    if df[y].min() < 0:
        ax.axvline(0, color='black')

    ax.set_ylabel('Crops')
    ax.set_xlabel(xlabel)
    ax.grid(axis='x')
    fig.savefig(savefolder+'national/'+savename, bbox_inches='tight',dpi=300)
def batchNationalCropBarGraphs(df_g=df_g):
    nationalBarHCrops(df_g['m3_per_ha'], y='mean', xlabel = 'Water demand (m$^3$/ha)', savename='m3_ha.jpg')
    nationalBarHCrops(df_g['m3_per_yield'], y='mean', xlabel = 'Water footprint (m$^3$/ton)', savename='m3_ton.jpg')
    nationalBarHCrops(df_g['profit_a_per_ha'], y='mean', xlabel = 'Gross profit economic productivity per area (USD/ha)', savename='profit_a_per_ha.jpg')
    nationalBarHCrops(df_g['profit_b_per_ha'], y='mean', xlabel = 'Net income economic productivity per area (USD/ha)', savename='profit_b_per_ha.jpg')
    nationalBarHCrops(df_g['profit_a_per_m3'], y='mean', xlabel = 'Gross profit economic productivity per irrigation volume (USD/m$^3$)', savename='profit_a_per_m3.jpg')
    nationalBarHCrops(df_g['profit_b_per_m3'], y='mean', xlabel = 'Net income economic productivity per irrigation volume (USD/m$^3$)', savename='profit_b_per_m3.jpg')
    nationalBarHCrops(df_g['ha'], y='sum', xlabel = 'Total area (ha)', savename='ha_total.jpg', std=False)
    nationalBarHCrops(df_g['mcm'], y='sum', xlabel = 'Total water demand (million cubic meters/yr)', savename='mcm_total.jpg', std=False)
    nationalBarHCrops(df_g['total_profit_mil_a'], y='sum', xlabel = 'Total gross profit (million $/yr)', savename='total_profit_a.jpg', std=False)
    nationalBarHCrops(df_g['total_profit_mil_b'], y='sum', xlabel = 'Total net income (million USD/yr)', savename='total_profit_b.jpg', std=False)
## create national graphs
batchNationalCropBarGraphs()


### CROP REGIONAL COMPARISION
def cropsRegionalComparisionHeatmap(crop_reg_c=crop_reg_c, savefolder=savefolder):
    max_c = crop_reg_c.fillna(0).to_numpy().max()
    min_c = crop_reg_c.fillna(999).to_numpy().min()

    divnorm = colors.TwoSlopeNorm(vmin=min_c, vcenter=1, vmax=max_c)
    fig, ax = plt.subplots()
    sns.heatmap(crop_reg_c, norm=divnorm, cmap='PuOr_r', ax=ax, annot=True, annot_kws={'size':7})
    ax.tick_params(bottom=False, left=False)
    ax.set_ylabel('Crops')
    ax.set_xlabel('ET zones')
    # for i in range(len(crop_reg_c.index)):
    #     for j in range(len(crop_reg_c.columns)):
    #         value = np.round(crop_reg_c.to_numpy()[i, j], 2)
    #         if value < 0.9 or value > 1.6:
    #             color = 'lightgrey'
    #         else:
    #             color = 'black'
    #         if value == value:
    #             text = ax.text(j + .5, i + .5, np.round(crop_reg_c.to_numpy()[i, j], 2),
    #                            ha="center", va="center", color=color, size=5, weight='semibold', alpha=.8)

    fig.savefig(savefolder + 'regionl_comp.jpg', bbox_inches='tight', dpi=300)
cropsRegionalComparisionHeatmap()

# df = df_g['m3_per_ha'];
# y='mean';
# ylabel='Mean area water footprint (m$^3$/ha)';
# savename='m3_ha.jpg'

def cropsBarHZonesWFArea(df_g_zone, crop):


    ##For sum color orer
    pal = sns.color_palette("Oranges", len(df_g_zone))
    rank = df_g_zone.sort_values(by=[('m3_per_ha','mean')])['ha']['sum'].argsort().argsort()

    #order by WF
    order_by_wf=df_g_zone.sort_values(by=[('m3_per_ha','mean')]).index

    fig, [ax1, ax2] = plt.subplots(1,2, sharey=True, figsize=(8,6))

    sns.barplot(y=df_g_zone.index, x= df_g_zone['m3_per_ha']['mean'], xerr=df_g_zone['m3_per_ha']['std'],
                ecolor='black', error_kw=(dict(lw=.75, alpha=.5)),
                orient="h", order=order_by_wf, alpha=.7, ax=ax1, palette='Blues')
    sns.barplot(y=df_g_zone.index, x=df_g_zone['ha']['sum'], orient="h", order=order_by_wf,
                alpha=.7, ax=ax2, palette=np.array(pal)[rank])

    ## add 0 yline

    ##LABELS
    ax1.set_ylabel('ET zones')
    ax2.set_ylabel('')
    ax1.set_xlabel('Water footprint (m$^3$/ton)',labelpad=10)
    ax2.set_xlabel('Cultivation area (ha)',labelpad=10)
    ax1.grid(axis='x')
    ax2.grid(axis='x')
    ax1.text(.05,.99,'A', fontsize=14, transform=ax1.transAxes, weight='bold')
    ax2.text(.05,.99,'B', fontsize=14, transform=ax2.transAxes, weight='bold')
    ## spacing
    plt.subplots_adjust(wspace=0.05)
    ax1.tick_params(bottom=False, left=False)
    ax2.tick_params(bottom=False, left=False)

    fig.savefig('{}crops/{}/WF_area.jpg'.format(savefolder,crop), bbox_inches='tight',dpi=300)

def cropsBarHZones(df, y, ylabel, savename, crop):
    # df = df.reset_index(level=1)

    fig, ax = plt.subplots()
    if y=='mean':
        sns.barplot(y=df.index, x= df[y], xerr=df['std'], ecolor='black', error_kw=(dict(lw=.75, alpha=.5)),
                orient="h", order=df.sort_values(by=y).index, alpha=.7, ax=ax, palette='Blues')
    elif y=='sum':
        sns.barplot(y=df.index, x= df[y], orient="h", order=df.sort_values(by=y).index,
                    alpha=.7, ax=ax, palette='Blues')
    # ax.barh(df.index, df[y], align='center', alpha=0.7)
    ## add 0 yline
    if df[y].min() < 0:
        ax.axhline(0, color='black')

    ax.set_xlabel(ylabel, labelpad=10)
    ax.set_ylabel('ET zones')
    ax.tick_params(bottom=False, left=False)

    ax.grid(axis='x')
    fig.savefig('{}crops/{}/{}'.format(savefolder,crop,savename), bbox_inches='tight',dpi=300)

def batchCropZoneBarGraphs(df_g, crop):

    cropsBarHZones(df_g['m3_per_ha'], y='mean', ylabel='Water demand (m$^3$/ha)',
                   savename='m3_ha.jpg', crop=crop)
    cropsBarHZones(df_g['m3_per_yield'], y='mean', ylabel='Water footprint (m$^3$/ha)',
                      savename='m3_ton.jpg', crop=crop)
    cropsBarHZones(df_g['profit_a_per_ha'], y='mean', ylabel='Gross profit economic productivity per area ($/ha)',
                      savename='profit_a_per_ha.jpg', crop=crop)
    cropsBarHZones(df_g['profit_b_per_ha'], y='mean', ylabel='Net income economic productivity per area ($/ha)',
                      savename='profit_b_per_ha.jpg', crop=crop)
    cropsBarHZones(df_g['profit_a_per_m3'], y='mean', crop=crop,
                      ylabel='Gross profit economic productivity per irrigation volume ($/m$^3$)', savename='profit_a_per_m3.jpg')
    cropsBarHZones(df_g['profit_b_per_m3'], y='mean', crop=crop,
                      ylabel='Net income economic productivity per irrigation volume ($/m$^3$)', savename='profit_b_per_m3.jpg')
    cropsBarHZones(df_g['ha'], y='sum', ylabel='Total area (ha)', savename='ha_total.jpg', crop=crop)
    cropsBarHZones(df_g['mcm'], y='sum', ylabel='Total water use (million cubic meters/yr)',
                      savename='mcm_total.jpg', crop=crop)
    cropsBarHZones(df_g['total_profit_mil_a'], y='sum', ylabel='Total gross profit (million USD/yr)',
                      savename='total_profit_a.jpg', crop=crop)
    cropsBarHZones(df_g['total_profit_mil_b'], y='sum', ylabel='Total net income (million USD/yr)',
                      savename='total_profit_b.jpg', crop=crop)

# crop='Almond'
def createAllCropZoneGraphs(df_g=df_g, crops=select_crops):
    for crop in crops:
        print(crop)
        #filter crop fields
        df_crop = df_select.loc[df_select['crop_display_name']==crop]
        ## filter zones with less than 30 ga
        zones_area = df_crop.groupby('zone_name')['ha'].sum()
        zones = list(zones_area.loc[zones_area>30].index)
        df_zone = df_crop.loc[df_crop['zone_name'].isin(zones)]
        ## groupby zone
        df_g_zone = df_zone.groupby('zone_name').agg(agg_d)
        ## batch graphs
        batchCropZoneBarGraphs(df_g_zone, crop)
        cropsBarHZonesWFArea(df_g_zone, crop)
        plt.close("all")
## create all crop zone graphs
createAllCropZoneGraphs()

# df_g_zone['ha']
#
# df_g_zone['m3_per_yield']['mean'].max() / df_g_zone['m3_per_yield']['mean'].min()
#
# dif/df_g_zone['m3_per_yield']['mean'].min()