#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt

import igraph as ig

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

from transport_model_functions import *

from datetime import datetime
from pathos.multiprocessing import ProcessPool, cpu_count

import shapely.geometry as geom
from shapely.geometry import Point, LineString

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import sys


def flow_allocation_OD_pair(iso3_O, iso3_D):
    columns_list = ['from_id','to_id','from_infra','to_infra','from_iso3','to_iso3','transport','flow','cost_freight','distance','time_total']

    ### extract the OD flow for the country pair
    OD_country_pairs = OD_flows_sector_grouped[(OD_flows_sector_grouped['iso3_O']== iso3_O)&(OD_flows_sector_grouped['iso3_D']== iso3_D)]

    ### min_flow
    min_flow = np.min(OD_country_pairs['q_sea_flow'].values)
    total_flow = np.sum(OD_country_pairs['q_sea_flow'].values)
    ### conversion value for cost function
    conversion_value = OD_country_pairs['value_tonnes'].iloc[0]

    ### read the latest maritime_network_input file
    maritime_network_input = pd.read_csv('Processed/maritime_network_input_sector'+str(sector)+'.csv', low_memory=False)

    ### load subnetwork
    demand_in, subnetwork = extract_subnetwork(maritime_network_input, OD_country_pairs, iso3_O, iso3_D, conversion_value)

    ### define the importing/exporting ports and importing/exporting maritime networks
    ## import ports
    importing_ports = subnetwork[subnetwork['flow'].isin(['port_import'])] ## import ports
    if total_flow >= importing_ports['capacity_open'].sum():
        try:
            subnetwork = correct_network_capacity(OD_country_pairs, subnetwork, importing_ports, total_flow, 'import', 'port')
        except:
            subnetwork = subnetwork.copy()
    ## export ports
    exporting_ports = subnetwork[subnetwork['flow'].isin(['port_export'])] ## export ports
    if total_flow >= exporting_ports['capacity_open'].sum():
        try:
            subnetwork = correct_network_capacity(OD_country_pairs, subnetwork, exporting_ports, total_flow, 'export', 'port')
        except:
            subnetwork = subnetwork.copy()
    ## import maritime
    importing_maritime = subnetwork[(subnetwork['flow']=='maritime')&(subnetwork['to_id'].isin(importing_ports['from_id'].unique()))]
    if total_flow >= importing_maritime['capacity_open'].sum():
        try:
            subnetwork = correct_network_capacity(OD_country_pairs, subnetwork, importing_maritime, total_flow, 'import', 'maritime')
        except:
            subnetwork = subnetwork.copy()
    ## export maritime
    exporting_maritime = subnetwork[(subnetwork['flow']=='maritime')&(subnetwork['from_id'].isin(exporting_ports['to_id'].unique()))]
    if total_flow >= exporting_maritime['capacity_open'].sum():
        try:
            subnetwork = correct_network_capacity(OD_country_pairs, subnetwork, exporting_maritime, total_flow, 'export', 'maritime')
        except:
            subnetwork = subnetwork.copy()

    ### get the list of indices of the components that directly connect O and D
    list_index_connection = importing_ports['edge_id'].to_list() + exporting_ports['edge_id'].to_list() + importing_maritime['edge_id'].to_list() + exporting_maritime['edge_id'].to_list()

    ## make sure you delete only possible inflow connections and not transhipment hubs
    delete_capacity_check = subnetwork[(subnetwork['capacity'].notna()) & (subnetwork['capacity_open'] < min_flow)]
    delete_capacity_check = delete_capacity_check[delete_capacity_check['edge_id'].isin(list_index_connection)]

    ### check which edges cannot be used because of capacity contraints
    subnetwork_original = subnetwork.copy()
    if len(delete_capacity_check)>0:
        subnetwork = subnetwork[~subnetwork['edge_id'].isin(delete_capacity_check['edge_id'].unique())]

    print(iso3_O, iso3_D, 'flow:',OD_country_pairs['q_sea_flow'].sum(), 'conversion:',conversion_value, len(maritime_network_input), len(subnetwork))

    ### create graph
    G_dir = graph_load_direction(subnetwork.reset_index(drop = True))
    G_dir_orig = graph_load_direction(subnetwork_original.reset_index(drop = True))
    ### Process the country pairs #####
    ### empty dataframes
    paths_df_all = pd.DataFrame()
    cost_freight_pair = pd.DataFrame()
    no_feasible_all = pd.DataFrame(columns = ['to_id','from_id','q_sea_flow'])
    index_delete = []
    for k in range(0, len(OD_country_pairs)):
        ### define input
        O_id = OD_country_pairs['from_id'].iloc[k]
        D_id = OD_country_pairs['to_id'].iloc[k]
        q_flow = OD_country_pairs['q_sea_flow'].iloc[k]
        v_flow = OD_country_pairs['v_sea_flow'].iloc[k]

        ### check if demand is in subnetwork
        if len(subnetwork[subnetwork['from_id']==O_id])==0 or len(subnetwork[subnetwork['to_id']==D_id])==0:
            #print('either origin or destination not in path')
            no_feasible_all = pd.concat([no_feasible_all, pd.DataFrame({'from_id':[O_id],'to_id':[D_id], 'q_sea_flow':[q_flow]})], ignore_index = True, sort = False)
            continue
        else:
            ### check which edges cannot be used because of capacity contraints
            delete_capacity = subnetwork[(subnetwork['capacity'].notna()) & (subnetwork['capacity_open'] < q_flow)]
            delete_capacity = delete_capacity[delete_capacity['edge_id'].isin(list_index_connection)]
            #rint('length delete capacity', len(delete_capacity))
            if len(delete_capacity)>0:
                list_delete = delete_capacity.edge_id.to_list()
                diff = [x for x in list_delete if x not in index_delete]
                index_delete = index_delete + diff

                if len(diff)>0:
                    ### update the network if new edges to delete are found (diff>0)
                    G_dir.es.select(edge_id_in=diff).delete()
                    #print(iso3_O, iso3_D,O_id, D_id,'length network after', len(G_dir.es))

            ### get the shortest path
            path = G_dir.get_shortest_paths(O_id, D_id, weights='cost_freight', mode='out',output='epath')

            ### check if there is a feasible path
            if len(path[0])==0:
                #rint(O_id, D_id, iso3_O, iso3_D, 'run no capacity model')
                ## run with uncapacity network
                path = G_dir_orig.get_shortest_paths(O_id, D_id, weights='cost_freight', mode='out',output='epath')
                #paths_df = pd.DataFrame({attr:  G_dir_orig.es[path[0]][attr] for attr in  columns_list})
                if len(path[0])==0:
                    no_feasible_all = pd.concat([no_feasible_all, pd.DataFrame({'from_id':[O_id],'to_id':[D_id], 'q_sea_flow':[q_flow]})], ignore_index = True, sort = False)
                else:
                    paths_df = pd.DataFrame({attr:  G_dir_orig.es[path[0]][attr] for attr in  columns_list})
                    paths_df = add_flow_info(paths_df, O_id, D_id, iso3_O, iso3_D, q_flow, v_flow)
                    paths_df_all = pd.concat([paths_df_all, paths_df],ignore_index = True, sort = False)

                    ## concat the total costs
                    df_shortest = pd.DataFrame({'from_id':[O_id],'to_id':[D_id],'cost_freight':[paths_df['cost_freight'].sum()],'distance':[paths_df['distance'].sum()], 'distance_maritime':[paths_df[paths_df['transport']=='maritime']['distance'].sum()],'q_sea_flow':[OD_country_pairs['q_sea_flow'].iloc[k]],'v_sea_flow':OD_country_pairs['v_sea_flow'].iloc[k]})
                    cost_freight_pair = pd.concat([cost_freight_pair, df_shortest], ignore_index= True, sort = False)

                    ### update subnetwork
                    subnetwork = update_sub_network_capacity(subnetwork, paths_df[paths_df['flow'].isin(['port_export','port_import','maritime'])][['from_id','to_id','transport','flow','q_sea_flow']].reset_index(drop = True), q_flow)

            else:
                paths_df = pd.DataFrame({attr:  G_dir.es[path[0]][attr] for attr in  columns_list})

                #### check if path is maritime
                if len(paths_df[paths_df['flow']=='maritime'])==0:
                    ### correct by remove possible paths through mutual neighbour
                    delete_connection = subnetwork[(subnetwork['to_iso3'] == iso3_D)&(subnetwork['from_id'].isin(demand_in[demand_in['to_iso3']!=iso3_O]['to_id'].unique()))]
                    ### diff_list
                    diff_list_correct = [x for x in delete_connection.edge_id.to_list() if x not in index_delete]
                    ### create copy of graph but delete edges
                    G_dir_new = G_dir.copy()
                    G_dir_new.es.select(edge_id_in=diff_list_correct).delete()
                    ### try path with new network
                    try:
                        path = G_dir_new.get_shortest_paths(O_id, D_id, weights = 'cost_freight', mode ='out', output ='epath')
                    except:
                        ### add to no feasible flow
                        no_feasible_all = pd.concat([no_feasible_all, pd.DataFrame({'from_id':[O_id],'to_id':[D_id], 'q_sea_flow':[q_flow]})], ignore_index = True, sort = False)
                        continue

                    ### create dataframe if path is found
                    paths_df = pd.DataFrame({attr:  G_dir_new.es[path[0]][attr] for attr in  columns_list})

                    ### check if new path has maritime
                    if len(paths_df[paths_df['flow']=='maritime'])==0:
                        ### add to no feasible flow
                        no_feasible_all = pd.concat([no_feasible_all, pd.DataFrame({'from_id':[O_id],'to_id':[D_id],'q_sea_flow':[q_flow]})], ignore_index = True, sort = False)
                        continue

                    else:
                        ##path was correct, add information
                        paths_df =  add_flow_info(paths_df, O_id, D_id, iso3_O, iso3_D, q_flow, v_flow)
                        paths_df_all = pd.concat([paths_df_all, paths_df],ignore_index = True, sort = False)

                        ## concat the total costs
                        df_shortest = pd.DataFrame({'from_id':[O_id],'to_id':[D_id],'cost_freight':[paths_df['cost_freight'].sum()],'distance':[paths_df['distance'].sum()], 'distance_maritime':[paths_df[paths_df['transport']=='maritime']['distance'].sum()],'q_sea_flow':[OD_country_pairs['q_sea_flow'].iloc[k]],'v_sea_flow':OD_country_pairs['v_sea_flow'].iloc[k]})
                        cost_freight_pair = pd.concat([cost_freight_pair, df_shortest], ignore_index= True, sort = False)

                        ### update subnetwork
                        subnetwork = update_sub_network_capacity(subnetwork, paths_df[paths_df['flow'].isin(['port_export','port_import','maritime'])][['from_id','to_id','transport','flow','q_sea_flow']].reset_index(drop = True), q_flow)


                else:
                    #### path was correct, add information
                    paths_df = add_flow_info(paths_df, O_id, D_id, iso3_O, iso3_D, q_flow, v_flow)
                    paths_df_all = pd.concat([paths_df_all, paths_df],ignore_index = True, sort = False)

                    ## concat the total costs
                    df_shortest = pd.DataFrame({'from_id':[O_id],'to_id':[D_id],'cost_freight':[paths_df['cost_freight'].sum()],'distance':[paths_df['distance'].sum()], 'distance_maritime':[paths_df[paths_df['transport']=='maritime']['distance'].sum()],'q_sea_flow':[OD_country_pairs['q_sea_flow'].iloc[k]],'v_sea_flow':OD_country_pairs['v_sea_flow'].iloc[k]})
                    cost_freight_pair = pd.concat([cost_freight_pair, df_shortest], ignore_index= True, sort = False)

                    ### update subnetwork
                    subnetwork = update_sub_network_capacity(subnetwork, paths_df[paths_df['flow'].isin(['port_export','port_import','maritime'])][['from_id','to_id','transport','flow','q_sea_flow']].reset_index(drop = True), q_flow)

                ## concat the total costs
                #df_shortest = pd.DataFrame({'from_id':[O_id],'to_id':[D_id],'cost_freight':[paths_df['cost_freight'].sum()],'distance':[paths_df['distance'].sum()], 'distance_maritime':[paths_df[paths_df['transport']=='maritime']['distance'].sum()],'q_sea_flow':[OD_country_pairs['q_sea_flow'].iloc[k]],'v_sea_flow':OD_country_pairs['v_sea_flow'].iloc[k]})
                #cost_freight_pair = pd.concat([cost_freight_pair, df_shortest], ignore_index= True, sort = False)
                #del paths_df

    ### port flows
    #### create multiplier if needed
    if len(paths_df_all)>0 and len(no_feasible_all)>0:
        print(iso3_O, iso3_D, 'multiplier needed')
        paths_df_all, cost_freight_pair = add_multiplier(paths_df_all, cost_freight_pair, OD_country_pairs, no_feasible_all)
    elif len(paths_df_all)==0:
        print(iso3_O, iso3_D, OD_country_pairs['q_sea_flow'].sum(),'not allocated')
        paths_df_all = pd.DataFrame()
        cost_freight_pair = pd.DataFrame()
    else:
        paths_df_all = paths_df_all.copy()
        cost_freight_pair = cost_freight_pair.copy()
    return paths_df_all, cost_freight_pair


##########  IMPORT  SOME GENERAL INFORMATION ############
## group1 countries
country_continent = pd.read_excel('Input/Information/continent_database.xlsx').rename(columns = {'Three_Letter_Country_Code':'iso3','Two_Letter_Country_Code':'iso2'}).drop_duplicates(subset = 'iso3',keep = 'first').replace(np.nan,'NA')
group1_countries = list(country_continent[country_continent['Continent_Code'].isin(['EU'])]['iso3'].unique())
group1_countries = group1_countries + ['USA','CAN','JPN','NZL','AUS','SGP']

#### VOT
VOT = pd.read_excel('Input/Information/value_time_sector.xlsx')

### conversion tables
conversion_table = pd.read_csv('Input/Information/conversion_industries_vessel_main.csv')

########### SET THE SECTOR ############
### set sector
sector = int(sys.argv[1]) #1
#print( int(sys.argv[1]))
create_maritime_costs = str(sys.argv[2])

print('########SECTOR', sector, '################')
######### GET VOT FOR SECTOR #######
# VOT
VOT_value = VOT[VOT['sector']== sector]['VOT'].iloc[0]

# vessel to sector conversion
conversion_table_sector = conversion_table[conversion_table['Industries']== sector]

########### EXTRACT MARITIME FLOWS ############
#### get the OD maritime freight flows
flows = pd.read_csv('Input/Maritime_flows/baci_mode_prediction_2015_HS6.csv')
# extract maritime trade flows and aggregate to the sector
flows_mar = flows.groupby(['iso3_O','iso3_D','WOIT_sector','Industries'])[['v','q','v_sea_predict','q_sea_predict']].sum().reset_index()
flows_mar['share'] = flows_mar['v_sea_predict'] / flows_mar['v']

### at least 2 percent needs to be maritime to be included
flows_mar = flows_mar[flows_mar['share']>0.02]
# unique iso3 in maritime flows
iso3_list = list(set(list(flows_mar['iso3_O'].unique()) + list(flows_mar['iso3_D'].unique())))

del flows

print('Maritime flows extracted')
########### LOAD HINTERLAND NETWORK ############
## read costs, time and distance
cost_hinterland, time_hinterland, distance_hinterland = pd.read_csv('Input/Hinterland_network/costs_normal.csv'), pd.read_csv('Input/Hinterland_network/time_normal.csv'), pd.read_csv('Input/Hinterland_network/distance_normal.csv')

## merge together
cost_time_hinterland = cost_hinterland.merge(time_hinterland[['from_id','to_id','time_total','from_infra','to_infra','transport']], on = ['from_id','to_id','from_infra','to_infra','transport'])
cost_time_hinterland = cost_time_hinterland.merge(distance_hinterland[['from_id','to_id','distance','from_infra','to_infra','transport']], on = ['from_id','to_id','from_infra','to_infra','transport'])

## add the value of time and create VOT costs
cost_time_hinterland['VOT'], cost_time_hinterland['time_cost'] = VOT_value, (cost_time_hinterland['time_total'])* cost_time_hinterland['VOT']/(100*24)  ### VOT in % per day

########### GENERATE FLOWS BETWEEN CENTROIDS ############
### import the centroids with population estimates
centroids = gpd.read_file('Input/Centroids/admin_centroids_pop.gpkg')
# remove centroids where only a very small fraction of the population lives (0.3%) and correct centroids
centroids = centroids[centroids['pop_frac']>0.003]
centroids = correct_centroids(centroids, cost_time_hinterland['from_id'].unique(), iso3_list)

##### generate flows between admin regions based on maritime flow and sector
OD_flows_sector_grouped = sector_flow_generator(centroids, flows_mar, sector)
OD_flows_sector_grouped.sort_values(by = 'q_sea_flow', inplace = True)

OD_flows_sector_grouped.to_csv('Processed/OD_mar_flows/OD_flows_sector'+str(sector)+'.csv', index = False)
### pairs
OD_pairs = OD_flows_sector_grouped.groupby(['iso3_O','iso3_D'])['q_sea_flow'].sum().reset_index().sort_values(by = 'q_sea_flow').reset_index(drop = True)
print('Number of OD pairs:', len(OD_pairs))
print('Number of flows:', len(OD_flows_sector_grouped))


########### PROCESS CAPACITY DATA PORTS AND MARITIME NETWORK ############
### read capacity
OD_capacity = pd.read_csv('Input/Maritime_network/OD_maritime_capacities.csv').drop(columns ='handling')
### get the port distance
OD_distance = OD_capacity[['from_id','to_id','distance']].drop_duplicates(keep = 'first')
### vessel to sector conversion
OD_capacity = OD_capacity.merge(conversion_table_sector, on = ['sub_vessel_type_AIS'])
OD_capacity['capacity_sector'] = OD_capacity['capacity'] * OD_capacity['conversion']

#### vessel utilization
port_utilization = pd.read_csv('Input/Maritime_network/port_utilization.csv')
## turnaroundtime
OD_capacity = generate_turnaround_handling(OD_capacity, port_utilization)

### Sector maritime capacities
OD_sector = weighted_groupby(OD_capacity, ['from_name','to_name','from_id','to_id','from_iso3','to_iso3'], 'capacity_sector', ['cost_km','cost_hour','speed','tat_to','tat_from'])
OD_sector.rename(columns = {'cost_km_frac':'cost_km','cost_hour_frac':'cost_hour','speed_frac':'speed','capacity_sector_total':'capacity','tat_to_frac':'tat_to','tat_from_frac':'tat_from'}, inplace = True)

### read the new port capacity information
port_capacity = pd.read_csv('Input/Maritime_network/port_capacity_info.csv').rename(columns ={'Import_capacity':'capacity_sector_import','Export_capacity':'capacity_sector_export'})
port_capacity_sector = generate_port_capacity(port_capacity, sector)

### correct the OD capacities based on port correction factors
OD_sector = correct_capacity(OD_sector, port_capacity_sector)
OD_sector['VOT'] = VOT_value ### add value of time to it

### generate the port-specific costs for handling and dwell time
port_specific_costs = generate_port_specific_costs(OD_capacity, group1_countries, VOT_value).reset_index(drop = True)

print('port capacity data processed')

########### PROCESS MARITIME NETWORK ############
### edges and nodes
edges_maritime_full =gpd.read_file('Input/Maritime_network/edges_maritime.gpkg',driver = 'GPKG').drop(columns = ['length_new','count'])
nodes_maritime =gpd.read_file('Input/Maritime_network/nodes_maritime.gpkg',driver = 'GPKG')
nodes_maritime_port = nodes_maritime[nodes_maritime['infra']=='port']
### add and correct panama canal
edges_maritime = add_panama_canal(edges_maritime_full, nodes_maritime)

del edges_maritime_full
## add the reverse and only add one directions
edges_maritime = pd.concat([edges_maritime, edges_maritime.rename(columns = {'from_id':'to_id','to_id':'from_id','from_infra':'to_infra','to_infra':'from_infra'})], ignore_index = True).drop_duplicates(subset = ['from_id','to_id'], keep = 'first')

########### ADD CAPACITIES TO MARITIME NETWORK ############
if create_maritime_costs == 'yes':
    ### create the maritime capacity
    distances_port, paths_df_maritime = generate_maritime_network_capacities(edges_maritime, OD_sector, 'distance')

    ### add costs to the maritime network
    edges_maritime_costs = generate_maritime_costs(edges_maritime, paths_df_maritime, VOT_value)
    edges_maritime_costs['from_id_merge'] = np.where(edges_maritime_costs['from_infra']=='port', edges_maritime_costs['from_id']+'_out', edges_maritime_costs['from_id'])
    edges_maritime_costs['to_id_merge'] = np.where(edges_maritime_costs['to_infra']=='port', edges_maritime_costs['to_id']+'_in', edges_maritime_costs['to_id'])

    edges_maritime_costs.to_file('Processed/Maritime_network/maritime_network_costs_sector'+str(sector)+'.gpkg', driver = 'GPKG')
    paths_df_maritime.to_csv('Processed/OD_paths/maritime_paths_sector'+str(sector)+'.csv', index = False)

else:
    edges_maritime_costs = gpd.read_file('Processed/Maritime_network/maritime_network_costs_sector'+str(sector)+'.gpkg')


########### RUN THE FREIGHT MODEL ############
#### create the maritime network
maritime_hinterland_edge_network = create_directed_maritime_network(cost_time_hinterland, nodes_maritime_port, OD_sector, OD_distance, port_capacity_sector, port_specific_costs)
maritime_hinterland_edge_network.to_csv('Processed/Network_directed/maritime_hinterland_edge_network_sector'+str(sector)+'.csv', index = False)
print('Number of edges network:', len(maritime_hinterland_edge_network))

### set the capacity and to be updated
maritime_hinterland_edge_network = set_baseline(maritime_hinterland_edge_network)

## set to around 20 nodes
nodes_used = cpu_count()-3
if nodes_used>20:
    nodes_used = 20
pool = ProcessPool(nodes=nodes_used)

#### rank, so you sample different countries
OD_pairs["rank"] = OD_pairs.groupby('iso3_O')["q_sea_flow"].rank("dense", ascending=False)
OD_pairs = OD_pairs.sort_values(by = 'rank', ascending = False)

print(datetime.now())
OD_pairs_subset = OD_pairs.copy()

print(len(OD_pairs_subset))

iso3O_list, iso3D_list = list(OD_pairs_subset['iso3_O'].values), list(OD_pairs_subset['iso3_D'].values)

### create sublinks after which to update
print('Number of nodes for processing:', nodes_used)
iso3O_sublist, iso3D_sublist = list(divide_chunks(iso3O_list, nodes_used)), list(divide_chunks(iso3D_list, nodes_used))

### load the network
## empty dataframes
paths_port_flows= pd.DataFrame()
paths_demand_flows= pd.DataFrame()
cost_freight_OD = pd.DataFrame()
### loop over OD_pairs_subset
for i in range(0, len(iso3O_sublist)):
    ### set the pool
    print('capacity employed', maritime_hinterland_edge_network['capacity'].sum()/1e9, maritime_hinterland_edge_network['capacity_used'].sum()/1e9, len(maritime_hinterland_edge_network[maritime_hinterland_edge_network['full']==1]))

    ### save the maritime input file which will be read by function
    maritime_hinterland_edge_network.to_csv('Processed/maritime_network_input_sector'+str(sector)+'.csv', index = False)

    ### run the flow allocation in parallel
    output1, output2  =  zip(*pool.map(flow_allocation_OD_pair, iso3O_sublist[i], iso3D_sublist[i]))
    paths = pd.concat(output1) ### concat output1
    cost_pairs = pd.concat(output2) ### concat output2

    print(i+1,'/',len(iso3O_sublist),':', len(paths), len(cost_pairs))
    if len(paths)>0:
        paths_port = paths[(paths['from_infra']=='port')&(paths['to_infra']=='port')]
        ### demand_statistics
        paths_demand = paths[paths['flow'].isin(['port_demand','demand_port'])]
        #### aggregate flows for network use
        path_aggregate = paths.groupby(['from_id','to_id','transport','flow'])[['q_sea_flow']].sum().reset_index()
        #### update the maritime_transport_network
        maritime_hinterland_edge_network = update_network_capacity(maritime_hinterland_edge_network, path_aggregate)

        ### concatenate
        paths_port_flows = pd.concat([paths_port_flows, paths_port],ignore_index = True, sort = False)
        paths_demand_flows = pd.concat([paths_demand_flows, paths_demand],ignore_index = True, sort = False)
        cost_freight_OD = pd.concat([cost_freight_OD, cost_pairs], ignore_index = True, sort = False)
        del cost_pairs, paths_port, paths_demand, path_aggregate, output1, output2


### total freight costs
cost_freight_OD['cost_freight_total'] = cost_freight_OD['cost_freight'] * cost_freight_OD['q_sea_flow']
cost_freight_OD['tonnes_km'] = cost_freight_OD['q_sea_flow'] * cost_freight_OD['distance']
cost_freight_OD['tonnes_km_maritime'] = cost_freight_OD['q_sea_flow'] * cost_freight_OD['distance_maritime']

### correct maritime_hinterland_network
maritime_hinterland_edge_network = correct_transshipment_flows(maritime_hinterland_edge_network)
### output results
paths_port_flows.to_csv('Output/Port_paths/port_paths'+str(sector)+'.csv', index = False)
paths_demand_flows.to_csv('Output/Port_paths/hinterland_paths'+str(sector)+'.csv', index = False)
cost_freight_OD.to_csv('Output/Freight_costs/freight_cost'+str(sector)+'.csv', index = False)
maritime_hinterland_edge_network.to_csv('Output/Network_capacity/Network_allocated'+str(sector)+'.csv', index = False)

print(datetime.now())
print('total flow:',cost_freight_OD['q_sea_flow'].sum()/1e6)
print('cost freight total:', cost_freight_OD['cost_freight_total'].sum()/1e6)
print('tonnes-km total:', cost_freight_OD['tonnes_km'].sum()/1e6)
print('tonnes-km  maritime total:', cost_freight_OD['tonnes_km_maritime'].sum()/1e6)
