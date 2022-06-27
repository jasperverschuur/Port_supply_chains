#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt

import igraph as ig

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

from transport_model_functions import *

import dask.dataframe as dd
from datetime import datetime



def backtrack_maritime_paths(mar_network, OD_dataframe, weight):
    columns_list = ['from_id','to_id','distance','time_total']

    """Generate maritime capacities on network based on OD_dataframe

    Args:
        mar_network: maritime network edges
        OD_dataframe: Dataframe containing the capacities on OD port pairs
        weight (str): string used for weight shortest paths

    Returns: distance OD dataframe, maritime network with capacities added
    """
    ### load the network
    G = graph_load_direction(mar_network)

    ## create unique origin ports
    OD_dataframe.sort_values(by = 'from_port').reset_index(drop = True, inplace = True)
    O_nodes = OD_dataframe['from_port'].unique()

    ### empty dataframe
    paths_df_maritime = pd.DataFrame()
    ### loop over the origin ports
    for j in range(0,len(O_nodes)):

        #print(j,len(O_nodes))
        ### origin and destination ID
        O_id = O_nodes[j]
        D_id = OD_dataframe[OD_dataframe['from_port']==O_id]['to_port'].to_list()

        ### get the path it took
        paths_df_all = pd.DataFrame()
        for dest in D_id:
            path = G.get_shortest_paths(O_id, dest, weights=weight, mode='out', output = 'epath')

            paths_df = pd.DataFrame({attr:  G.es[path[0]][attr] for attr in  columns_list})

            paths_df['from_port'] = [O_id] * len(paths_df)
            paths_df['to_port'] = [dest] * len(paths_df)

            ### concat
            paths_df_all = pd.concat([paths_df_all, paths_df],ignore_index = True, sort = False)

        #### process and concatenate
        paths_df_all = paths_df_all.merge(OD_dataframe[['from_port','to_port','q_sea_flow','v_sea_flow']], on = ['from_port','to_port'])

        ### append path
        paths_df_maritime = pd.concat([paths_df_maritime, paths_df_all],ignore_index = True)

    ### return
    return paths_df_maritime


def run_correction_factor(edge_network):
    iso3_unique = edge_network['from_iso3'].unique()

    rebalance_export = pd.DataFrame()
    rebalance_import = pd.DataFrame()
    for iso3 in iso3_unique:
        ### exporting ports in country
        export_ports = edge_network[(edge_network['flow']=='port_export')&(edge_network['from_iso3']==iso3)]
        ### importing ports in country
        import_ports = edge_network[(edge_network['flow']=='port_import')&(edge_network['from_iso3']==iso3)]

        if len(export_ports) == 0:
            continue
        else:
            ### too much flow allocated
            export_ports_overcap = export_ports[export_ports['capacity_open']<0].reset_index(drop = True)
            ### too little flow allocated
            export_ports_undercap = export_ports[export_ports['capacity_open']>=0].reset_index(drop = True)

            if len(export_ports_overcap)==0 or len(export_ports_undercap)==0:
                continue
            else:
                ### amount reallocated
                export_overcapacity = np.abs(export_ports_overcap['capacity_open'].sum())
                export_undercapacity = np.abs(export_ports_undercap['capacity_open'].sum())
                export_reallocate = np.min([export_undercapacity, export_overcapacity])

                ### reallocate export flows
                export_ports_overcap['reallocate'] = -(export_ports_overcap['capacity_open']/export_ports_overcap['capacity_open'].sum())*export_reallocate
                export_ports_undercap['reallocate'] = (export_ports_undercap['capacity_open']/export_ports_undercap['capacity_open'].sum())*export_reallocate

                export_ports_overcap['capacity_used_old'] = export_ports_overcap['capacity_used']
                export_ports_overcap['capacity_used'] = export_ports_overcap['capacity_used']+export_ports_overcap['reallocate']
                export_ports_overcap['reallocate_frac'] = (export_ports_overcap['capacity_used']/np.abs(export_ports_overcap['capacity_used_old']))

                export_ports_undercap['capacity_used_old'] = export_ports_undercap['capacity_used']
                export_ports_undercap['capacity_used'] = export_ports_undercap['capacity_used']+export_ports_undercap['reallocate']
                export_ports_undercap['reallocate_frac'] = (export_ports_undercap['capacity_used']/np.abs(export_ports_undercap['capacity_used_old']))

                export_ports_corrected = pd.concat([export_ports_overcap, export_ports_undercap], ignore_index = True, sort = False)[['to_id','reallocate_frac']].rename(columns = {'to_id':'port_export','reallocate_frac':'reallocate_export'})
                export_ports_corrected['iso3_O'] = iso3
                ## concat
                rebalance_export = pd.concat([rebalance_export, export_ports_corrected], ignore_index = True, sort = False)

        if len(import_ports) == 0:
            continue
        else:
            import_ports_overcap = import_ports[import_ports['capacity_open']<0].reset_index(drop = True)
            import_ports_undercap = import_ports[import_ports['capacity_open']>=0].reset_index(drop = True)
            if len(import_ports_overcap)==0 or len(import_ports_undercap)==0:
                continue
            else:
                ### amount reallocated
                import_overcapacity = np.abs(import_ports_overcap['capacity_open'].sum())
                import_undercapacity = np.abs(import_ports_undercap['capacity_open'].sum())
                import_reallocate = np.min([import_undercapacity, import_overcapacity])


                ### reallocate export flows
                import_ports_overcap['reallocate'] = -(import_ports_overcap['capacity_open']/import_ports_overcap['capacity_open'].sum())*import_reallocate
                import_ports_undercap['reallocate'] = (import_ports_undercap['capacity_open']/import_ports_undercap['capacity_open'].sum())*import_reallocate

                import_ports_overcap['capacity_used_old'] = import_ports_overcap['capacity_used']
                import_ports_overcap['capacity_used'] = import_ports_overcap['capacity_used']+import_ports_overcap['reallocate']
                import_ports_overcap['reallocate_frac'] = (import_ports_overcap['capacity_used']/np.abs(import_ports_overcap['capacity_used_old']))

                import_ports_undercap['capacity_used_old'] = import_ports_undercap['capacity_used']
                import_ports_undercap['capacity_used'] = import_ports_undercap['capacity_used']+import_ports_undercap['reallocate']
                import_ports_undercap['reallocate_frac'] = (import_ports_undercap['capacity_used']/np.abs(import_ports_undercap['capacity_used_old']))

                import_ports_corrected = pd.concat([import_ports_overcap, import_ports_undercap], ignore_index = True, sort = False)[['from_id','reallocate_frac']].rename(columns = {'from_id':'port_import','reallocate_frac':'reallocate_import'})
                import_ports_corrected['iso3_D'] = iso3
                ## concat
                rebalance_import = pd.concat([rebalance_import, import_ports_corrected], ignore_index = True, sort = False)


    return rebalance_export, rebalance_import


##### define the sector #####
#sector = 1

sector_list = [1,2,3,4,5,6,7,8,9,10,11]

### import trade data
trade_data = pd.read_csv('Input/Maritime_flows/baci_mode_prediction_2015_EORA.csv')
trade_data['share_mar'] = trade_data['q_sea_predict']/trade_data['q']

port_to_port_df = pd.DataFrame()
canal_df = pd.DataFrame()
cost_freight_df = pd.DataFrame()
for sector in sector_list:
    print('SECTOR', sector, datetime.now())
    ######## LOAD DATA #########
    ### trade data for sector
    trade_data_sector = trade_data[trade_data['Industries']==sector]

    ### edge maritime costs
    edges_maritime_costs = gpd.read_file('Processed/Maritime_network/maritime_network_costs_sector'+str(sector)+'.gpkg')

    ### port paths
    paths_port_flows = dd.read_csv('Output/Port_paths/port_paths'+str(sector)+'.csv').compute()

    ### port paths
    paths_demand_flows = dd.read_csv('Output/Port_paths/hinterland_paths'+str(sector)+'.csv').compute()

    ### maritime network with capacities
    maritime_hinterland_edge_network = pd.read_csv('Output/Network_capacity/Network_allocated'+str(sector)+'.csv')

    ### cost freight
    cost_freight_OD = dd.read_csv('Output/Freight_costs/freight_cost'+str(sector)+'.csv').compute()

    ###
    print('data loaded in')

    ######## CORRECT NETWORK BY REBALACING IMPORTS AND EXPORTS #########
    rebalance_export, rebalance_import = run_correction_factor(maritime_hinterland_edge_network)

    # connections between ports and demand
    demand_export = paths_port_flows[paths_port_flows['flow']=='port_export'][['from_id','to_id','iso3_O','iso3_D','demand_from_id','demand_to_id']].drop_duplicates().rename(columns = {'to_id':'port_export'})
    demand_import = paths_port_flows[paths_port_flows['flow']=='port_import'][['from_id','to_id','iso3_O','iso3_D','demand_from_id','demand_to_id']].drop_duplicates().rename(columns = {'from_id':'port_import'})
    demand_export  = demand_export.merge(rebalance_export, on = ['port_export','iso3_O'])
    demand_import = demand_import.merge(rebalance_import, on = ['port_import','iso3_D'])

    ### merge on the ports flows
    paths_port_flows_corrected = paths_port_flows.merge(demand_export[['demand_from_id','demand_to_id','port_export','reallocate_export']], on = ['demand_from_id','demand_to_id'], how = 'outer').replace(np.nan, 1.0)
    paths_port_flows_corrected = paths_port_flows_corrected.merge(demand_import[['demand_from_id','demand_to_id','port_import','reallocate_import']], on = ['demand_from_id','demand_to_id'], how = 'outer').replace(np.nan, 1.0)

    ######## CORRECT THE PORT PATHS AND COST_FREIGHT DATASETS #########
    paths_port_flows_corrected['correction'] = paths_port_flows_corrected['reallocate_import'] * paths_port_flows_corrected['reallocate_export']
    paths_port_flows_corrected['q_sea_flow'] = paths_port_flows_corrected['q_sea_flow'] * paths_port_flows_corrected['correction']
    paths_port_flows_corrected['v_sea_flow'] = paths_port_flows_corrected['v_sea_flow'] * paths_port_flows_corrected['correction']
    ### drop columns
    paths_port_flows_corrected = paths_port_flows_corrected.drop(columns = ['port_export', 'reallocate_export', 'port_import', 'reallocate_import'])

    ### merge correction factor on OD cost_freight df
    correction_factor = paths_port_flows_corrected[['demand_from_id','demand_to_id','correction']].drop_duplicates().rename(columns = {'demand_from_id':'from_id','demand_to_id':'to_id'})

    ### correct the hinterland flows
    paths_demand_flows_corrected = paths_demand_flows.merge(correction_factor, on = ['from_id','to_id'])
    paths_demand_flows_corrected['q_sea_flow'] = paths_demand_flows_corrected['q_sea_flow'] * paths_demand_flows_corrected['correction']
    paths_demand_flows_corrected['v_sea_flow'] = paths_demand_flows_corrected['v_sea_flow'] * paths_demand_flows_corrected['correction']

    ### correct the flows
    cost_freight_OD_corrected = cost_freight_OD.merge(correction_factor, on = ['from_id','to_id'])
    cost_freight_OD_corrected['q_sea_flow'] = cost_freight_OD_corrected['q_sea_flow'] * cost_freight_OD_corrected['correction']
    cost_freight_OD_corrected['v_sea_flow'] = cost_freight_OD_corrected['v_sea_flow'] * cost_freight_OD_corrected['correction']
    cost_freight_OD_corrected['cost_freight_total'] = cost_freight_OD_corrected['cost_freight_total'] * cost_freight_OD_corrected['correction']
    cost_freight_OD_corrected['tonnes_km'] = cost_freight_OD_corrected['tonnes_km'] * cost_freight_OD_corrected['correction']
    cost_freight_OD_corrected['tonnes_km_maritime'] = cost_freight_OD_corrected['tonnes_km_maritime'] * cost_freight_OD_corrected['correction']

    ##### AGGREGAGET THE COST FREIGHT DATA #####
    cost_freight_OD_corrected['iso3_O'] = cost_freight_OD_corrected['from_id'].str.split('_', n = 1,expand= True)[1].str.split('.',n = 1, expand = True)[0]
    cost_freight_OD_corrected['iso3_D'] = cost_freight_OD_corrected['to_id'].str.split('_', n = 1,expand= True)[1].str.split('.',n = 1, expand = True)[0]

    ### aggregate
    cost_freight_aggregate = cost_freight_OD_corrected.groupby(['iso3_O','iso3_D'])[['q_sea_flow','v_sea_flow','cost_freight_total','tonnes_km','tonnes_km_maritime']].sum().reset_index()
    cost_freight_aggregate = cost_freight_aggregate.merge(trade_data_sector[['iso3_O','iso3_D','share_mar','Industries']], on = ['iso3_O','iso3_D'])


    ######## FIND THE MARITIME PATHS TAKEN BETWEEN O AND D PORTS #########
    print('create maritime paths')
    paths_port_flows_corrected[['id_next','demand_from_id_next','demand_to_id_next','q_sea_flow_next']] = paths_port_flows_corrected[['from_id','demand_from_id','demand_to_id','q_sea_flow']].shift(-1)
    OD_maritime_paths = paths_port_flows_corrected[['to_id','id_next','iso3_O','iso3_D','demand_from_id','demand_to_id','v_sea_flow','q_sea_flow','q_sea_flow_next']].copy().rename(columns = {'to_id':'from_id','id_next':'to_id'})
    OD_maritime_paths = OD_maritime_paths[OD_maritime_paths['q_sea_flow']==OD_maritime_paths['q_sea_flow_next']]
    OD_maritime_paths['from_port'] = OD_maritime_paths['from_id'].str.split('_', n = 1, expand = True)[0]
    OD_maritime_paths['to_port'] = OD_maritime_paths['to_id'].str.split('_', n = 1, expand = True)[0]
    print('maritime paths created')

    ##### Get the maritime paths that connect ports to each other #####
    ### unique port connections
    unique_mar_paths =OD_maritime_paths[['from_port','to_port','q_sea_flow','v_sea_flow']].groupby(['from_port','to_port'])[['q_sea_flow','v_sea_flow']].sum().reset_index()

    ### run paths
    mar_paths = backtrack_maritime_paths(edges_maritime_costs, unique_mar_paths, 'distance')
    print('maritime paths backtracked')
    ### canal flows
    canal_flows = get_canal_flow(mar_paths)
    print('canals extracted')
    ## merge on maritime connections dataframe
    OD_maritime_paths_canal = OD_maritime_paths.merge(canal_flows[['from_port','to_port','canal']], on = ['from_port','to_port'])

    ######## CREATE THE AGGREGATE PORT TRADE AND CANAL NETWORK DEPENDENCIES #########
    ### sum of q and v per OD pair
    OD_maritime_sum = paths_port_flows_corrected.drop_duplicates(subset = ['demand_from_id','demand_to_id']).groupby(['iso3_O','iso3_D'])[['q_sea_flow','v_sea_flow']].sum().rename(columns = {'q_sea_flow':'q_sea_flow_sum','v_sea_flow':'v_sea_flow_sum'})

    # create the port to port trade network
    port_to_port_trade = paths_port_flows_corrected.groupby(['from_id','to_id','from_iso3','to_iso3','flow','iso3_O','iso3_D'])[['q_sea_flow','v_sea_flow']].sum().reset_index()
    port_to_port_trade = port_to_port_trade.merge(OD_maritime_sum, on = ['iso3_O','iso3_D'])
    port_to_port_trade = port_to_port_trade.merge(trade_data_sector[['iso3_O','iso3_D','share_mar','Industries']], on = ['iso3_O','iso3_D'])
    port_to_port_trade['q_share_port'] = port_to_port_trade['q_sea_flow']/port_to_port_trade['q_sea_flow_sum']
    port_to_port_trade['v_share_port'] = port_to_port_trade['v_sea_flow']/port_to_port_trade['v_sea_flow_sum']

    ### Multiply with fraction of trade being maritime
    port_to_port_trade['q_share_trade'] = port_to_port_trade['q_share_port'] * port_to_port_trade['share_mar']
    port_to_port_trade['v_share_trade'] = port_to_port_trade['v_share_port'] * port_to_port_trade['share_mar']

    ### create dependence on canals
    canal_trade = OD_maritime_paths_canal.groupby(['iso3_O','iso3_D','canal'])[['q_sea_flow','v_sea_flow']].sum().reset_index()
    canal_trade = canal_trade.merge(OD_maritime_sum, on = ['iso3_O','iso3_D'])
    canal_trade = canal_trade.merge(trade_data_sector[['iso3_O','iso3_D','share_mar','Industries']], on = ['iso3_O','iso3_D'])
    canal_trade['q_share_canal'] = canal_trade['q_sea_flow']/canal_trade['q_sea_flow_sum']
    canal_trade['v_share_canal'] = canal_trade['v_sea_flow']/canal_trade['v_sea_flow_sum']
    ### Multiply with fraction of trade being maritime
    canal_trade['q_share_trade'] = canal_trade['q_share_canal'] * canal_trade['share_mar']
    canal_trade['v_share_trade'] = canal_trade['v_share_canal'] * canal_trade['share_mar']


    ######## OUTPUT #########
    OD_maritime_paths_canal.to_csv('Output_processed/Mar_paths/canal_paths'+str(sector)+'.csv', index = False)
    canal_trade.to_csv('Output_processed/Mar_paths/canal_trade_network'+str(sector)+'.csv', index = False)

    mar_paths.to_csv('Output_processed/Mar_paths/mar_paths'+str(sector)+'.csv', index = False)

    paths_port_flows_corrected.to_csv('Output_processed/Port_paths/port_paths'+str(sector)+'.csv', index = False)
    port_to_port_trade.to_csv('Output_processed/Port_paths/port_trade_network'+str(sector)+'.csv', index = False)
    paths_demand_flows_corrected.to_csv('Output_processed/Port_paths/hinterland_paths'+str(sector)+'.csv', index = False)

    cost_freight_OD_corrected.to_csv('Output_processed/Port_paths/freight_cost'+str(sector)+'.csv', index = False)
    cost_freight_aggregate.to_csv('Output_processed/Port_paths/freight_cost_aggregate'+str(sector)+'.csv', index = False)

    ### concatenate
    port_to_port_df = pd.concat([port_to_port_df, port_to_port_trade], ignore_index = True, sort = False)
    canal_df = pd.concat([canal_df, canal_trade], ignore_index = True, sort = False)
    cost_freight_df = pd.concat([cost_freight_df, cost_freight_aggregate], ignore_index = True, sort = False)

######## OUTPUT CONCATENATED DATA #########
### concat and output all
port_to_port_df.to_csv('Output_processed/Port_paths/port_trade_network.csv', index = False)
canal_df.to_csv('Output_processed/Mar_paths/canal_trade_network.csv', index = False)
cost_freight_df.to_csv('Output_processed/Port_paths/freight_cost_aggregate.csv', index = False)
