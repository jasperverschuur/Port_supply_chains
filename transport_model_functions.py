import pandas as pd
import geopandas as gpd
import numpy as np

import shapely.geometry as geom
from shapely.geometry import Point, LineString

import igraph as ig

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def sector_flow_generator(centroids, flows_maritime, sector):
    """ Generate the sector-specific maritime trade flows between subnational admin areas

        Input:
        centroids: centroids dataset per country with population fraction per centroid added
        flows_maritime: bilateral maritime trade flows, aggregated to sector ('Industries'), including both value (v_sea_predict) and quantity (q_sea_predict)
        sector: Sector of interest, ranging from 1 to 11

        Output:
        Dataframe with scaled maritime trade flows between admin regions, including value (v_sea_flow) and quantity (q_sea_flow)"""


    ### select a sector
    flows_mar_sector = flows_maritime[flows_maritime['Industries']==sector].reset_index()
    flows_mar_sector['value_tonnes'] = flows_mar_sector['v_sea_predict']/flows_mar_sector['q_sea_predict']

    ##### Merge centroid information to flows
    flows_mar_sector_OD = flows_mar_sector.merge(centroids[['iso3','ID','pop_frac']].rename(columns = {'iso3':'iso3_O','ID':'ID_O','pop_frac':'pop_frac_O'}), on = ['iso3_O'])
    flows_mar_sector_OD = flows_mar_sector_OD.merge(centroids[['iso3','ID','pop_frac']].rename(columns = {'iso3':'iso3_D','ID':'ID_D','pop_frac':'pop_frac_D'}), on = ['iso3_D'])


    ##### Create the flow per OD pair
    flows_mar_sector_OD['v_sea_flow'] = flows_mar_sector_OD['v_sea_predict'].values * flows_mar_sector_OD['pop_frac_O'].values * flows_mar_sector_OD['pop_frac_D'].values
    flows_mar_sector_OD['q_sea_flow'] = flows_mar_sector_OD['q_sea_predict'].values * flows_mar_sector_OD['pop_frac_O'].values * flows_mar_sector_OD['pop_frac_D'].values

    #### value tonnes for OD pair
    OD_flows_country = flows_mar_sector_OD.groupby(['iso3_O','iso3_D'])[['v_sea_flow','q_sea_flow']].sum().reset_index()
    OD_flows_country['value_tonnes'] = OD_flows_country['v_sea_flow']/OD_flows_country['q_sea_flow']

    #### create ids and add value
    OD_flows_sector_grouped = flows_mar_sector_OD.groupby(['iso3_O','iso3_D','ID_O','ID_D'])[['v_sea_flow','q_sea_flow']].sum().reset_index()
    OD_flows_sector_grouped['from_id'] = 'demand_' + OD_flows_sector_grouped['ID_O'].astype(str)
    OD_flows_sector_grouped['to_id'] = 'demand_' + OD_flows_sector_grouped['ID_D'].astype(str)
    OD_flows_sector_grouped = OD_flows_sector_grouped.merge(OD_flows_country[['iso3_O','iso3_D','value_tonnes']], on = ['iso3_O','iso3_D'])


    del flows_mar_sector, flows_mar_sector_OD

    return OD_flows_sector_grouped


def graph_load_direction(edges):
    """Creates

    Args:
        edges (pandas.DataFrame) : containing road network edges, with from and to ids, and distance / time columns

    Returns:
        igraph.Graph (object) : a graph with distance and time attributes
    """
    #return ig.Graph.TupleList(gdfNet.edges[['from_id','to_id','distance']].itertuples(index=False),edge_attrs=['distance'])
    edges = edges.reindex(['from_id','to_id'] + [x for x in list(edges.columns) if x not in ['from_id','to_id']],axis=1)
    graph = ig.Graph.TupleList(edges.itertuples(index=False), edge_attrs=list(edges.columns)[2:],directed=True)
    graph.vs['id'] = graph.vs['name']
    graph.es['from_id'] = edges['from_id'].values
    graph.es['to_id'] = edges['to_id'].values

    return graph

def graph_load(edges):
    """Creates

    Args:
        edges (pandas.DataFrame) : containing road network edges, with from and to ids, and distance / time columns

    Returns:
        igraph.Graph (object) : a graph with distance and time attributes
    """
    #return ig.Graph.TupleList(gdfNet.edges[['from_id','to_id','distance']].itertuples(index=False),edge_attrs=['distance'])
    edges = edges.reindex(['from_id','to_id'] + [x for x in list(edges.columns) if x not in ['from_id','to_id']],axis=1)
    graph = ig.Graph.TupleList(edges.itertuples(index=False), edge_attrs=list(edges.columns)[2:],directed=False)
    graph.vs['id'] = graph.vs['name']
    graph.es['from_id'] = edges['from_id'].values
    graph.es['to_id'] = edges['to_id'].values

    return graph

def weighted_groupby(df, group, weight, variables):
    """Weighted groupby pandas

    Args: dataframe, group to aggregate, weight and variables

    Returns: grouped dataframe
    """

    df_total = df.groupby(group)[weight].sum().reset_index().rename(columns = {weight: weight+'_total'})
    df = df.merge(df_total, on = group)
    df['frac'] = df[weight]/df[weight+'_total']

    variables_frac_list = []
    for variable in variables:
        df[variable+'_frac'] = df[variable] * df['frac']
        variables_frac_list.append(variable+'_frac')

    df_groupby = df.groupby(group)[variables_frac_list].sum().reset_index()
    df_groupby = df_groupby.merge(df_total, on = group)

    return df_groupby


def generate_port_capacity(port_capacity, sector):
    port_capacity_sector = port_capacity[port_capacity['Industries']==sector].reset_index(drop=True)
    port_capacity_sector['name'] = port_capacity_sector['port-name'].astype(str) + '_' + port_capacity_sector['country']
    port_capacity_sector['capacity_sector_tranship'] = port_capacity_sector['capacity_sector_import']
    port_capacity_sector['capacity_sector_tranship'] = np.where(port_capacity_sector['capacity_sector_export']<port_capacity_sector['capacity_sector_tranship'], port_capacity_sector['capacity_sector_export'], port_capacity_sector['capacity_sector_tranship'])

    return port_capacity_sector

def correct_capacity(OD_sector, port_capacity_sector):
    OD_sector = OD_sector.merge(port_capacity_sector[['name','import_correction']].rename(columns = {'name':'to_name'}), on = 'to_name')
    OD_sector = OD_sector.merge(port_capacity_sector[['name','export_correction']].rename(columns = {'name':'from_name'}), on = 'from_name')

    OD_sector['capacity'] = OD_sector['capacity'] * OD_sector['import_correction']* OD_sector['export_correction']

    return OD_sector.drop(columns = ['import_correction','export_correction'])


def generate_turnaround_handling(OD_capacity, port_utilization):
    port_utilization  = port_utilization[['port-name','country','vessel_type_main','tat_median','count','handling','id']].copy().rename(columns = {'id':'to_id'})
    #vessel_utilization['time_tonnes'] = (vessel_utilization['count'] * vessel_utilization['turn_around_time'])/ (vessel_utilization['Import_mean'] + vessel_utilization['Export_mean'])

    port_utilization['to_name'] = port_utilization['port-name'].astype(str) + '_' + port_utilization['country']
    port_utilization['from_name'] = port_utilization['port-name'].astype(str) + '_' + port_utilization['country']

    ## merge turnaround time
    OD_capacity = OD_capacity.merge(port_utilization[['to_name','vessel_type_main','tat_median']].rename(columns = {'tat_median':'tat_to'}), on = ['to_name','vessel_type_main'])
    OD_capacity = OD_capacity.merge(port_utilization[['from_name','vessel_type_main','tat_median']].rename(columns = {'tat_median':'tat_from'}), on = ['from_name','vessel_type_main'])

    ### merge handling costs
    OD_capacity = OD_capacity.merge(port_utilization[['to_id','handling','vessel_type_main']], on = ['to_id','vessel_type_main'])

    return OD_capacity




def generate_port_capacities(import_capacity, export_capacity, conversion_table_sector):
    """Generate port capacities on network based on vessel types

    Args:
    import_capacity: import capacity of ports per vessel type
    export_capacity: export capacity of ports per vessel type
    conversion_table: conversion table vessel to sector

    Returns: capacities added for import, export and transhipment
    """

    # import
    import_capacity = import_capacity.merge(conversion_table_sector, on = ['sub_vessel_type_AIS'])
    import_capacity['capacity_sector_import'] = import_capacity['Import_capacity_vessel_corrected'] * import_capacity['conversion']
    import_capacity_sector = import_capacity.groupby(['port-name','country','GID_0'])['capacity_sector_import'].sum().reset_index()

    #export
    export_capacity = export_capacity.merge(conversion_table_sector, on = ['sub_vessel_type_AIS'])
    export_capacity['capacity_sector_export'] = export_capacity['Export_capacity_vessel_corrected'] * export_capacity['conversion']
    export_capacity_sector = export_capacity.groupby(['port-name','country','GID_0'])['capacity_sector_export'].sum().reset_index()


    # merge transhipment and constrain by both imports and exports
    port_capacity_sector = import_capacity_sector.merge(export_capacity_sector, on= ['port-name','country','GID_0'], how = 'outer').replace(np.nan,0)
    port_capacity_sector['capacity_sector_tranship'] = port_capacity_sector['capacity_sector_import']
    port_capacity_sector['capacity_sector_tranship'] = np.where(port_capacity_sector['capacity_sector_export']<port_capacity_sector['capacity_sector_tranship'], port_capacity_sector['capacity_sector_export'], port_capacity_sector['capacity_sector_tranship'])

    ### add name
    port_capacity_sector['name'] = port_capacity_sector['port-name'].astype(str) + '_' + port_capacity_sector['country']

    return port_capacity_sector

def generate_maritime_costs(edges_maritime, path_aggregate, VOT_value, length = 'yes'):
    """Add costs to maritime network based on distance, speed, and Value of Time

    Returns: same edge dataframe but with the costs added
    """
    edges_maritime_information = edges_maritime.merge(path_aggregate, on = ['from_id','to_id'])

    ### add the information
    edges_maritime_information['time_total'] = edges_maritime_information['distance']/edges_maritime_information['speed']
    edges_maritime_information['cost'] = edges_maritime_information['distance'] * edges_maritime_information['cost_km']
    edges_maritime_information['VOT'] = VOT_value
    edges_maritime_information['time_cost'] = edges_maritime_information['time_total'] * edges_maritime_information['VOT']/(100*24)  ### VOT in % per day
    edges_maritime_information['time_cost'] = np.where(edges_maritime_information['time_cost'].isna(),0,edges_maritime_information['time_cost'])

    if length == 'yes':
        edges_maritime_information['length'] = edges_maritime_information.geometry.length

    return edges_maritime_information


def correct_centroids(centroids, cost_hinterland_id, iso3_list):
    """Check which centroids are in hinterland network and recalculate population

    Returns: same input dataframe of centroids but now corrected
    """

    #### correct a few centroids that are not connected to the network
    centroids['id'] = 'demand_'+ centroids['ID'].astype(str)

    #### missing centroids
    missing = centroids[~centroids['id'].isin(cost_hinterland_id)]
    missing = missing[missing['iso3'].isin(iso3_list)]

    ### remove the missing ones from centroids
    centroids = centroids[~centroids['ID'].isin(missing.ID.unique())].reset_index(drop = True)

    #### recalculate pop_frac
    pop_total = centroids[['iso3','pop']].groupby(['iso3'])['pop'].sum().reset_index().rename(columns = {'pop':'pop_national'})
    centroids = centroids.drop(columns = ['pop_national', 'pop_frac']).merge(pop_total, on = ['iso3'])
    centroids['pop_frac'] = centroids['pop'] / centroids['pop_national']

    return centroids


def divide_chunks(l, n):
    """Divide a list in chunks of length n
    """
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]

def add_panama_canal(edges_maritime, nodes_maritime):
    """Correct the panama canal in maritime network

    """

    list_delete = ['maritime4046','maritime2929','maritime2931']
    edges_maritime = edges_maritime[~edges_maritime['from_id'].isin(list_delete)]
    edges_maritime = edges_maritime[~edges_maritime['to_id'].isin(list_delete)]

    edges_maritime = edges_maritime[~((edges_maritime['from_id']=='maritime7270') & (edges_maritime['to_id']=='maritime7249'))]

    ### add panama canal
    df_panama = pd.DataFrame({'from_id':'maritime7270','to_id':'maritime455','from_infra':'maritime','to_infra':'maritime','distance': [43.8]})
    df_panama['geometry'] = geom.LineString(nodes_maritime[nodes_maritime['id']=='maritime7270'].geometry.iloc[0].coords[:] + nodes_maritime[nodes_maritime['id']=='maritime455'].geometry.iloc[0].coords[:])
    gdf_panama = gpd.GeoDataFrame(df_panama, geometry=df_panama['geometry'])

    edges_maritime = pd.concat([edges_maritime, gdf_panama], ignore_index = True, sort = False).reset_index(drop = True)

    return edges_maritime



def generate_maritime_network_capacities(mar_network, OD_dataframe, weight):
    """Generate maritime capacities on network based on OD_dataframe

    Args:
        mar_network: maritime network edges
        OD_dataframe: Dataframe containing the capacities on OD port pairs
        weight (str): string used for weight shortest paths

    Returns: distance OD dataframe, maritime network with capacities added
    """
    ### load the network
    G = graph_load(mar_network)

    ## create unique origin ports
    OD_dataframe.sort_values(by = 'from_id').reset_index(drop = True, inplace = True)
    O_nodes = OD_dataframe['from_id'].unique()

    ### empty dataframe
    distance_df_maritime = pd.DataFrame()
    paths_df_maritime = pd.DataFrame()
    ### loop over the origin ports
    for j in range(0,len(O_nodes)):
        #print(j,len(O_nodes))
        ### origin and destination ID
        O_id = O_nodes[j]
        D_id = OD_dataframe[OD_dataframe['from_id']==O_id]['to_id'].to_list()

        ### find all the available shortest paths for origin port
        length = G.shortest_paths(source=O_id, target=D_id, weights=weight)
        df_shortest = pd.DataFrame({'from_id':[O_id] *len(D_id),'to_id':D_id,'distance':length[0]})
        df_shortest = df_shortest[df_shortest['distance']!=np.inf]

        ### append the shortest distance
        distance_df_maritime = pd.concat([distance_df_maritime, df_shortest],ignore_index = True)

        ### get the path it took
        paths_df_all = pd.DataFrame()
        for dest in D_id:
            path = G.get_shortest_paths(O_id, dest, weights=weight, mode='out')

            ID_O = G.vs[path[0][:-1]]['name']
            ID_D = G.vs[path[0][1:]]['name']

            paths_df = pd.DataFrame({'from_id':ID_O,'to_id':ID_D})
            paths_df['port_from_id'] = [O_id] * len(paths_df)
            paths_df['port_to_id'] = [dest] * len(paths_df)


            ### concat
            paths_df_all = pd.concat([paths_df_all, paths_df],ignore_index = True, sort = False)

        #### process and concatenate
        paths_df_all = paths_df_all.merge(OD_dataframe[['from_id','to_id','cost_km','cost_hour','speed','capacity']].rename(columns = {'from_id':'port_from_id','to_id':'port_to_id'}), on = ['port_from_id','port_to_id'])
        path_aggregate = weighted_groupby(paths_df_all, ['from_id','to_id'], 'capacity', ['cost_km','cost_hour','speed'])
        path_aggregate.rename(columns = {'cost_km_frac':'cost_km','cost_hour_frac':'cost_hour','speed_frac':'speed','capacity_total':'capacity'}, inplace = True)


        ### append path
        paths_df_maritime = pd.concat([paths_df_maritime, path_aggregate],ignore_index = True)


    ### aggregate one across all origin pairs
    paths_df_maritime_weighted = weighted_groupby(paths_df_maritime, ['from_id','to_id'], 'capacity', ['cost_km','cost_hour','speed'])
    paths_df_maritime_weighted.rename(columns = {'cost_km_frac':'cost_km','cost_hour_frac':'cost_hour','speed_frac':'speed','capacity_total':'capacity'}, inplace = True)

    del paths_df_maritime

    ### return
    return distance_df_maritime, paths_df_maritime_weighted


def generate_port_specific_costs(OD_dataset, group1_countries, VOT_value, trans = 'no'):
    """Generate costs on the port network

    Args:
        OD_dataset: origin and destination dataset flows
        group1_countries (list): group of countries with lower dwell time
        VOT (float): Value of Time

    Returns: Port specific costs dataframe
    """
    ## get handling costs and dwell time per port, ingoing and outgoing, add country correction
    incoming = weighted_groupby(OD_dataset, ['to_name','to_id','to_iso3'], 'capacity_sector', ['handling','dwell_infra'])
    outgoing = weighted_groupby(OD_dataset, ['from_name','from_id','from_iso3'], 'capacity_sector', ['handling','dwell_infra'])

    ### set dwell
    incoming['dwell_infra_frac'] = np.where(incoming['to_iso3'].isin(group1_countries), (incoming['dwell_infra_frac']*0.5)*24, incoming['dwell_infra_frac']*24)
    outgoing['dwell_infra_frac'] = np.where(outgoing['from_iso3'].isin(group1_countries), (outgoing['dwell_infra_frac']*0.5)*24, outgoing['dwell_infra_frac']*24)

    ### add if trans

    incoming['dwell_infra_trans'] = incoming['dwell_infra_frac'] * 1.2
    incoming['handling_trans'] = incoming['handling_frac'] * 0.75

    outgoing['dwell_infra_trans'] = outgoing['dwell_infra_frac'] * 1.2
    outgoing['handling_trans'] = outgoing['handling_frac'] * 0.75

    ### rename to merge
    incoming.rename(columns = {'to_name':'name','to_id':'id','to_iso3':'iso3'}, inplace = True)
    outgoing.rename(columns = {'from_name':'name','from_id':'id','from_iso3':'iso3'}, inplace = True)

    ### merge
    port_specific_costs = incoming.merge(outgoing, on = ['name','id','iso3'], how = 'outer')
    port_specific_costs['capacity_sector'] = port_specific_costs['capacity_sector_total_x'] + port_specific_costs['capacity_sector_total_y']

    #### specify which ports have not both ingoing and outgoing information
    port_specific_costs['one-side'] = np.where(port_specific_costs['capacity_sector'].isna(), 1, 0)
    port_specific_costs = port_specific_costs.replace(np.nan, 0)

    #### get the correct cost estimates
    ## specificy costs
    port_specific_costs['VOT'] = VOT_value
    port_specific_costs['handling_cost'] = np.where(port_specific_costs['one-side']==1, (port_specific_costs['handling_frac_x'] + port_specific_costs['handling_frac_y']), (port_specific_costs['handling_frac_x'] + port_specific_costs['handling_frac_y'])/2)
    port_specific_costs['dwell_infra'] = np.where(port_specific_costs['one-side']==1,(port_specific_costs['dwell_infra_frac_x'] + port_specific_costs['dwell_infra_frac_y']), (port_specific_costs['dwell_infra_frac_x'] + port_specific_costs['dwell_infra_frac_y'])/2)
    port_specific_costs['dwell_cost'] = port_specific_costs['dwell_infra'] * port_specific_costs['VOT']/(100*24)  ### VOT in % per day

    ## specificy costs
    port_specific_costs['handling_cost_trans'] = np.where(port_specific_costs['one-side']==1, (port_specific_costs['handling_trans_x'] + port_specific_costs['handling_trans_y']), (port_specific_costs['handling_trans_x'] + port_specific_costs['handling_trans_y'])/2)
    port_specific_costs['dwell_infra_trans'] = np.where(port_specific_costs['one-side']==1,(port_specific_costs['dwell_infra_trans_x'] + port_specific_costs['dwell_infra_trans_y']), (port_specific_costs['dwell_infra_trans_x'] + port_specific_costs['dwell_infra_trans_y'])/2)
    port_specific_costs['dwell_cost_trans'] = port_specific_costs['dwell_infra_trans'] * port_specific_costs['VOT']/(100*24)  ### VOT in % per day

    port_specific_costs = port_specific_costs[['name','id','iso3','handling_cost','dwell_infra','dwell_cost','handling_cost_trans','dwell_infra_trans','dwell_cost_trans']]


    return port_specific_costs


def create_directed_maritime_network(cost_time_hinterland, nodes_maritime_port, OD_sector, OD_distance, port_capacity, port_specific_costs):
    ###### HINTERLAND SIDE ###########
    ### demand to port
    demand_port_in = cost_time_hinterland[['from_id','to_id','from_infra','to_infra','to_name','from_name','from_iso3','to_iso3','transport','cost','time_cost','distance','time_total']].copy()
    demand_port_in['to_id'] = demand_port_in['to_id']+'_land'
    demand_port_in = demand_port_in[['from_id','to_id','from_iso3','to_iso3','from_infra','to_infra','cost','time_cost','distance','transport','time_total']].copy()
    demand_port_in['flow'] = 'demand_port'

    ### port to demand
    demand_port_out = cost_time_hinterland[['from_id','to_id','from_infra','to_infra','to_name','from_name','from_iso3','to_iso3','transport','cost','time_cost','distance','time_total']].copy()
    demand_port_out.rename(columns = {'from_id':'to_id','to_id':'from_id','from_infra':'to_infra','to_infra':'from_infra','to_name':'from_name','from_name':'to_name','from_iso3':'to_iso3','to_iso3':'from_iso3'}, inplace = True)
    demand_port_out['from_id'] = demand_port_out['from_id']+'_land'
    demand_port_out = demand_port_out[['from_id','to_id','from_iso3','to_iso3','from_infra','to_infra','cost','time_cost','distance','transport','time_total']].copy()
    demand_port_out['flow'] = 'port_demand'

    ###### PORT SIDE ###########
    ### outgoing maritime flow
    land_port_out = nodes_maritime_port[['id','infra','name','iso3']].copy().rename(columns = {'id':'to_id','infra':'to_infra','name':'to_name','iso3':'to_iso3'})
    land_port_out[['from_id','from_infra','from_name','from_iso3']] = land_port_out[['to_id','to_infra','to_name','to_iso3']]
    land_port_out['from_id'] = land_port_out['from_id']+'_land'
    land_port_out['to_id'] = land_port_out['to_id']+'_out'
    # add capacity
    land_port_out = land_port_out.merge(port_capacity[['name','capacity_sector_export']].rename(columns = {'capacity_sector_export':'capacity','name':'to_name'}), on = 'to_name')
    ### add cost and time
    land_port_out = land_port_out.merge(port_specific_costs[['name','handling_cost','dwell_cost','dwell_infra']].rename(columns = {'handling_cost':'cost','dwell_cost':'time_cost','name':'to_name'}), on = 'to_name')
    land_port_out['distance'] = 0
    land_port_out['time_total'] = land_port_out['dwell_infra']
    land_port_out = land_port_out[['from_id','to_id','from_iso3','to_iso3','from_infra','to_infra','cost','time_cost','distance','capacity','time_total']].copy()
    land_port_out['flow'] = 'port_export'
    land_port_out = land_port_out[land_port_out['capacity']>0]
    land_port_out['transport'] = 'port'

    ### incoming maritime flow
    land_port_in = nodes_maritime_port[['id','infra','name','iso3']].copy().rename(columns = {'id':'to_id','infra':'to_infra','name':'to_name','iso3':'to_iso3'})
    land_port_in[['from_id','from_infra','from_name','from_iso3']] = land_port_in[['to_id','to_infra','to_name','to_iso3']]
    land_port_in['to_id'] = land_port_in['to_id']+'_land'
    land_port_in['from_id'] = land_port_in['from_id']+'_in'
    # add capacity
    land_port_in = land_port_in.merge(port_capacity[['name','capacity_sector_import']].rename(columns = {'capacity_sector_import':'capacity','name':'to_name'}), on = 'to_name')
    ### add cost and time
    land_port_in = land_port_in.merge(port_specific_costs[['name','handling_cost','dwell_cost','dwell_infra']].rename(columns = {'handling_cost':'cost','dwell_cost':'time_cost','name':'to_name'}), on = 'to_name')
    land_port_in['time_total'] = land_port_in['dwell_infra']
    land_port_in['distance'] = 0
    land_port_in = land_port_in[['from_id','to_id','from_iso3','to_iso3','from_infra','to_infra','cost','time_cost','distance','capacity','time_total']].copy()
    land_port_in['flow'] = 'port_import'
    land_port_in = land_port_in[land_port_in['capacity']>0]
    land_port_in['transport'] = 'port'


    ### transhipment flow
    port_in_out = nodes_maritime_port[['id','infra','name','iso3']].copy().rename(columns = {'id':'to_id','infra':'to_infra','name':'to_name','iso3':'to_iso3'})
    port_in_out[['from_id','from_infra','from_name','from_iso3']] = port_in_out[['to_id','to_infra','to_name','to_iso3']]
    port_in_out['to_id'] = port_in_out['to_id']+'_out'
    port_in_out['from_id'] = port_in_out['from_id']+'_in'
    # add capacity
    port_in_out = port_in_out.merge(port_capacity[['name','capacity_sector_tranship']].rename(columns = {'capacity_sector_tranship':'capacity','name':'to_name'}), on = 'to_name')
    port_in_out = port_in_out[port_in_out['capacity']>0]
    ### add cost and time
    port_in_out = port_in_out.merge(port_specific_costs[['name','handling_cost_trans','dwell_cost_trans','dwell_infra_trans']].rename(columns = {'handling_cost_trans':'cost','dwell_cost_trans':'time_cost','name':'to_name'}), on = 'to_name')
    port_in_out['time_total'] = port_in_out['dwell_infra_trans']
    port_in_out['distance'] = 0
    port_in_out = port_in_out[['from_id','to_id','from_iso3','to_iso3','from_infra','to_infra','cost','time_cost','distance','capacity','time_total']].copy()
    port_in_out = port_in_out[port_in_out['cost'].notna()]
    port_in_out['flow'] = 'port_trans'
    port_in_out['transport'] = 'port'


    ###### PORT TO MARITIME TRANSITION ###########
    ### maritime to maritime
    maritime_edge = OD_sector[['from_name','to_name','from_id','to_id','from_iso3','to_iso3','cost_km','cost_hour','speed','capacity','tat_from','tat_to','VOT']].copy()
    maritime_edge = maritime_edge.merge(OD_distance, on = ['from_id','to_id'])

    maritime_edge['from_id'] = maritime_edge['from_id'].astype(str) + '_out'
    maritime_edge['to_id'] = maritime_edge['to_id'].astype(str) + '_in'
    maritime_edge['from_infra'] = 'port_mar'
    maritime_edge['to_infra'] = 'port_mar'
    maritime_edge['transport'] = 'maritime'
    maritime_edge['flow'] = 'maritime'
    maritime_edge['cost'] = maritime_edge['distance'] * maritime_edge['cost_km']

    maritime_edge['time_total'] =  (maritime_edge['distance'] / maritime_edge['speed'])  + maritime_edge['tat_from'] + maritime_edge['tat_to']
    maritime_edge['time_cost'] = maritime_edge['time_total'] * maritime_edge['VOT']/(100*24)  ### VOT in % per day
    maritime_edge_final = maritime_edge[['from_id', 'to_id', 'from_iso3', 'to_iso3', 'from_infra', 'to_infra','cost', 'time_cost', 'distance', 'transport', 'flow', 'capacity','time_total']].copy()


    ### concatenate everything
    maritime_hinterland_edge_network = pd.concat([demand_port_in, demand_port_out, land_port_out, land_port_in, port_in_out, maritime_edge_final], ignore_index = True, sort = False)
    ### add edge_id
    maritime_hinterland_edge_network['edge_id'] = maritime_hinterland_edge_network['from_id'].astype(str) + '_' +maritime_hinterland_edge_network['to_id'].astype(str)

    return maritime_hinterland_edge_network


def set_baseline(df_transport):
    """Set the baseline of the maritime network
    """
    ### set the initial capacity and to be updated after every iteration
    df_transport['capacity_used']=0
    df_transport['fraction_used']=0
    df_transport['capacity_open']= df_transport['capacity']
    df_transport['full'] = 0
    df_transport['time_multiplier'] = 1
    df_transport['time_cost_original'] = df_transport['time_cost']

    return df_transport



def extract_sub_maritime_network(subnetwork, importing_ports, exporting_ports):
    columns_list = ['edge_id','from_id','to_id']

    no_mar = subnetwork[~subnetwork['flow'].isin(['maritime','port_trans'])].reset_index(drop = True)
    mar = subnetwork[subnetwork['flow'].isin(['maritime','port_trans'])].reset_index(drop = True)

    G_dir_mar = graph_load_direction(mar)

    edge_id_list = []
    for o_id in exporting_ports['to_id'].unique():
        d_id = importing_ports['from_id'].unique()
        paths=G_dir_mar.get_shortest_paths(o_id, d_id, mode='out', weights = 'cost_freight', output = 'epath')
        for path in paths:
            paths_df = pd.DataFrame({attr:  G_dir_mar.es[path][attr] for attr in  columns_list})
            edge_id_list = list(dict.fromkeys(edge_id_list + list(paths_df['edge_id'].values)))


    mar_extract = mar[mar['edge_id'].isin(edge_id_list)]

    importing_ports_after = mar_extract[mar_extract['to_id'].isin(importing_ports['from_id'].unique())].reset_index(drop = True)
    importing_ports_after['port'] = importing_ports_after['to_id'].str.split('_',expand = True)[0]

    exporting_ports_after = mar_extract[mar_extract['from_id'].isin(exporting_ports['to_id'].unique())].reset_index(drop = True)
    exporting_ports_after['port'] = exporting_ports_after['from_id'].str.split('_',expand = True)[0]


    no_mar_import = no_mar[no_mar['flow'].isin(['port_demand','port_import'])].reset_index(drop = True)
    no_mar_import['port'] = no_mar_import['from_id'].str.split('_',expand = True)[0]
    no_mar_import = no_mar_import[no_mar_import['port'].isin(importing_ports_after['port'].unique())]


    no_mar_export = no_mar[no_mar['flow'].isin(['demand_port','port_export'])].reset_index(drop = True)
    no_mar_export['port'] = no_mar_export['to_id'].str.split('_',expand = True)[0]
    no_mar_export = no_mar_export[no_mar_export['port'].isin(exporting_ports_after['port'].unique())]


    ### add everything together
    subnetwork_new = pd.concat([mar_extract, no_mar_import, no_mar_export], ignore_index = True, sort = False).reset_index(drop = True)
    return subnetwork_new


def extract_subnetwork(maritime_network, OD_country_pairs, iso3_O, iso3_D, conversion_value):
    """Extract subnetwork from full network

    Args:
        maritime_network: full maritime transport network
        OD_country_pairs: pairs of demand centroids for processing
        iso3_O (str): origin country (iso3)
        iso3_D (str): destination country (iso3)
        conversation value (float): conversion value to estimate total freight costs

    Returns: demand_in, demand_out, and subnetwork
    """

    maritime_part = maritime_network[maritime_network['flow'].isin(['maritime','port_in','port_out','port_trans'])]

    ## demand
    demand_in = maritime_network[maritime_network['from_id'].isin(OD_country_pairs['from_id'].unique())]
    demand_in = demand_in[demand_in['to_iso3']!=iso3_D] ### no direct connection to country of destination

    demand_out = maritime_network[maritime_network['to_id'].isin(OD_country_pairs['to_id'].unique())]
    demand_out = demand_out[demand_out['from_iso3']!=iso3_O] ### no direct connection to country of origin
    ### port to maritime
    port_out = maritime_network[(maritime_network['to_infra']=='port')&(maritime_network['from_id'].isin(demand_in['to_id'].unique()))]
    port_in = maritime_network[(maritime_network['from_infra']=='port')&(maritime_network['to_id'].isin(demand_out['from_id'].unique()))]

    list_index = list(maritime_part.index) + list(demand_in.index) + list(demand_out.index) + list(port_out.index) + list(port_in.index)

    ### create the subnetwork
    subnetwork = maritime_network.loc[list_index]
    subnetwork['cost_freight'] = subnetwork['cost'] + subnetwork['time_cost']*conversion_value
    subnetwork = subnetwork.sort_values(by = 'cost_freight', ascending = True).drop_duplicates(subset= ['from_id','to_id']).reset_index(drop = True)

    ### get only the data of interest
    importing_ports = subnetwork[subnetwork['flow'].isin(['port_import'])] ## import ports
    exporting_ports = subnetwork[subnetwork['flow'].isin(['port_export'])] ## export ports

    try:
        ### try creating the subnetwork, if not, let's use the full one
        subnetwork = extract_sub_maritime_network(subnetwork, importing_ports, exporting_ports).reset_index(drop = True)
    except:
        subnetwork = subnetwork.copy()

    return demand_in, subnetwork

def add_flow_info(df, O_id, D_id, iso3_O, iso3_D, q, v):
    """Add flow information to processed shortest paths
    """

    df['demand_from_id'] = [O_id] * len(df)
    df['demand_to_id'] = [D_id] * len(df)
    df['q_sea_flow'] = q
    df['v_sea_flow'] = v
    df['iso3_O'] = iso3_O
    df['iso3_D'] = iso3_D

    return df


def update_network_capacity(df_transport, paths):
    """Update the capacity information in the network

    Args:
        df_transport: full maritime transport network
        paths: processed shortest paths


    Returns: updated df_transport network
    """
    #### update the maritime_transport_network
    trans_flow = paths[paths['flow']=='port_trans']
    if len(trans_flow)>0:
        port_trans = create_import_export_trans(trans_flow)

        paths = pd.concat([paths, port_trans],ignore_index = True, sort = False).groupby(['from_id','to_id','transport','flow'])[['q_sea_flow']].sum().reset_index()

    df_transport = df_transport.merge(paths.rename(columns = {'q_sea_flow':'q_used'}), on = ['from_id','to_id','transport','flow'], how = 'outer')
    df_transport['q_used'] = df_transport['q_used'].replace(np.nan, 0)
    df_transport['capacity_used'] = df_transport['capacity_used'] + df_transport['q_used']
    df_transport['capacity_open'] = df_transport['capacity'] - df_transport['capacity_used']
    df_transport['fraction_used'] = (df_transport['capacity_used'] /  df_transport['capacity']).replace(np.nan, 0)
    df_transport.drop(columns = 'q_used', inplace = True)

    #### update whether edge is full or not
    df_transport['full'] = np.where(((df_transport['capacity'].notna()) & (df_transport['capacity_used'] > df_transport['capacity'])), 1, df_transport['full'])
    df_transport['fraction_used'] = np.where(df_transport['full'] == 1, 1, df_transport['fraction_used'])

    return df_transport


def update_sub_network_capacity(df_transport, paths, q):
    """Update the capacity information in the subnetwork

    Args:
        df_transport:sub maritime transport network for OD pair
        paths: processed shortest paths
        q: flow associated with admin-to-admin flow


    Returns: updated df_transport network
    """
    #### update the maritime_transport_network
    ### create edge_id to merge
    paths['edge_id'] = paths['from_id'].astype(str) + '_' + paths['to_id'].astype(str)
    df_transport['q_used'] = np.where(df_transport['edge_id'].isin(paths['edge_id']), q, 0)
    df_transport['capacity_used'] = df_transport['capacity_used'] + df_transport['q_used']
    df_transport['capacity_open'] = df_transport['capacity'] - df_transport['capacity_used']
    df_transport.drop(columns = 'q_used', inplace = True)

    return df_transport

def add_congestion(df_transport, alpha, beta):
    """Add congestion to the network based on updated capacity
    """
    ### congestion function = Cong = (1 + alpha * utilization **beta)
    df_transport['time_multiplier'] = np.where(df_transport['flow'].isin(['port_out','port_in','port_export','port_import']), (1 + alpha*df_transport['fraction_used']**beta), 1)
    df_transport['time_cost'] = df_transport['time_cost_original'] * df_transport['time_multiplier']

    return df_transport

def add_multiplier(paths_df, cost_freight, OD_pair, no_feasible_all):
    """Add multiplier if some paths are not available
    """
    ### fraction increase, multiplier
    multiplier = OD_pair['q_sea_flow'].sum() / (OD_pair['q_sea_flow'].sum() - no_feasible_all['q_sea_flow'].sum())

    ### add the multiplier to the flows that are possible
    paths_df[['q_sea_flow','v_sea_flow']] = paths_df[['q_sea_flow','v_sea_flow']]* multiplier
    cost_freight[['q_sea_flow','v_sea_flow']] = cost_freight[['q_sea_flow','v_sea_flow']]* multiplier

    return paths_df, cost_freight

def get_canal_flow(paths):
    """Extract the paths going through the Panama and Suez canal
    """
    ####
    ### set the canals
    canal1 = paths[(paths['from_id']=='maritime1606')&(paths['to_id']=='maritime2927')].reset_index(drop = True)
    canal1['canal'] = 'Suez-south'
    canal2 = paths[(paths['from_id']=='maritime2927')&(paths['to_id']=='maritime1606')].reset_index(drop = True)
    canal2['canal'] = 'Suez-north'
    canal3 = paths[(paths['from_id']=='maritime7249')&(paths['to_id']=='maritime7515')].reset_index(drop = True)
    canal3['canal'] = 'Panama-south'
    canal4 = paths[(paths['from_id']=='maritime7515')&(paths['to_id']=='maritime7249')].reset_index(drop = True)
    canal4['canal'] = 'Panama-north'
    paths_canal = pd.concat([canal1, canal2, canal3, canal4], ignore_index = True, sort = False)
    return paths_canal


def correct_transshipment_flows(network):
    ### transhipment flows
    trans_total = network[network['flow']=='port_trans'][['from_id','to_id','capacity_used']]

    if len(trans_total[trans_total['capacity_used']>0]):
        ### correction imports
        trans_correct_import = trans_total[trans_total['capacity_used']>0][['from_id','capacity_used']].rename(columns = {'capacity_used':'trans_from'})
        trans_correct_import['flow'] = 'port_import'

        ### correction exports
        trans_correct_export = trans_total[trans_total['capacity_used']>0][['to_id','capacity_used']].rename(columns = {'capacity_used':'trans_to'})
        trans_correct_export['flow'] = 'port_export'


        ### add and correct

        network = network.merge(trans_correct_import, on = ['from_id','flow'], how = 'outer').replace(np.nan,0)
        network['capacity_used'] = network['capacity_used'] - network['trans_from']
        network['capacity_open'] = network['capacity_open'] + network['trans_from']

        network = network.merge(trans_correct_export, on = ['flow','to_id'], how = 'outer').replace(np.nan,0)
        network['capacity_used'] = network['capacity_used'] - network['trans_to']
        network['capacity_open'] = network['capacity_open'] + network['trans_to']
    else:
        None

    return network

def create_import_export_trans(trans_flow):
    port_import = trans_flow.copy()
    port_import['to_id'] = port_import['to_id'].str.split('_', expand = True)[0] + '_' + 'land'
    port_import['flow'] = 'port_import'

    port_export = trans_flow.copy()
    port_export['from_id'] = port_export['from_id'].str.split('_', expand = True)[0] + '_' + 'land'
    port_export['flow'] = 'port_export'

    port_trans = pd.concat([port_import, port_export], ignore_index = True, sort = False)

    return port_trans


def hinterland_capacity_increase(network, port_capacity, OD_country, flow_dir):
    if flow_dir == 'import':
        flow_1 = 'to_id'
        flow_2 = 'from_id'
    else:
        flow_1 = 'from_id'
        flow_2 = 'to_id'
    dest_flow = OD_country.groupby([flow_1])['q_sea_flow'].sum().reset_index()
    ### connection ports to ID
    port_demand_connections = network[network[flow_1].isin(dest_flow[flow_1].unique())][['from_id','to_id','cost_freight']]
    port_demand_connections = port_demand_connections.merge(dest_flow, on = [flow_1])
    ### add port capacity
    port_demand_connections = port_demand_connections.merge(port_capacity[[flow_1,'capacity']].rename(columns = {flow_1: flow_2}), on = [flow_2])

    ### estimate the increase in capacity
    port_demand_connections = port_demand_connections.merge(port_demand_connections.groupby([flow_1])['capacity'].sum().reset_index().rename(columns = {'capacity':'capacity_total'}), on = flow_1)
    port_demand_connections['capacity_increase'] = port_demand_connections['q_sea_flow'] * port_demand_connections['capacity']/port_demand_connections['capacity_total']

    ### aggregate by the port
    capacity_increase = port_demand_connections.groupby([flow_2])['capacity_increase'].sum().reset_index()
    capacity_increase['frac'] = capacity_increase['capacity_increase'] / capacity_increase['capacity_increase'].sum()
    capacity_increase = capacity_increase.rename(columns = {flow_2:flow_1})

    #capacity_increase_mar = capacity_increase.copy()

    #print(OD_country['from_id'].iloc[0],OD_country['to_id'].iloc[0],len(capacity_increase_mar))
    ### merge and correct
    #if flow_dir =='export':
        #capacity_increase_mar['from_id'] = capacity_increase_mar['from_id'].str.split('_', expand = True)[0] + '_' + 'out'
    #else:
        #capacity_increase_mar['to_id'] = capacity_increase_mar['to_id'].str.split('_', expand = True)[0] + '_' + 'in'

    return capacity_increase#, capacity_increase_mar

def correct_network_capacity(OD_country, network, network_correct, flow, flow_dir, transport):
    network_non_flow = network[~network['edge_id'].isin(network_correct['edge_id'].unique())].reset_index(drop = True) ### drop index of those that you want to correct
    network_flow = network_correct.reset_index(drop = True)

    if flow_dir == 'import':
        flow_1 = 'to_id'
        flow_2 = 'from_id'
        port_capacity = network[network['flow'].isin(['port_import'])]
    else:
        flow_1 = 'from_id'
        flow_2 = 'to_id'
        port_capacity = network[network['flow'].isin(['port_export'])]

    ## export ports
    capacity_increase = hinterland_capacity_increase(network, port_capacity, OD_country, flow_dir)

    ### fraction of outgoing capacity in ports
    if transport == 'port':
        network_flow = network_flow.merge(capacity_increase[[flow_1,'frac']], on = flow_1)

        #network_flow = network_flow.merge(capacity_increase_mar[[flow_1,'frac']], on = flow_1)

    ### if full, make the capacity_open zero again
    network_flow['capacity_open'] = np.where(network_flow['full']==1, 0, network_flow['capacity_open'])

    if transport == 'port':
        ### spread the total flow over the open capacity
        network_flow['capacity_open'] = network_flow['capacity_open'] + network_flow['frac'] * flow
        ### arbitrary increase the capacity
        network_flow['capacity'] = network_flow['capacity'] + network_flow['frac'] * flow
    else:
        ### spread the total flow over the open capacity
        network_flow['capacity_open'] = network_flow['capacity_open'] + flow
        ### arbitrary increase the capacity
        network_flow['capacity'] = network_flow['capacity'] + flow

    network_flow['capacity_used'] = network_flow['capacity'] - network_flow['capacity_open']
    network_flow['fraction_used'] = (network_flow['capacity_used'] /  network_flow['capacity'])
    network_flow['full'] = np.where((network_flow['capacity_used'] >= network_flow['capacity']), 1, 0)


    sub_network_corrected = pd.concat([network_flow, network_non_flow], ignore_index = True, sort = False)
    if transport == 'port':
        sub_network_corrected.drop(columns = 'frac', inplace = True)


    return sub_network_corrected
