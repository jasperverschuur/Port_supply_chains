#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import dask.dataframe as dd
import pymrio
import datetime


def modify_A_matrix(A_matrix, Z_matrix, flows, list_countries, sectors):
    A_matrix_append = A_matrix.copy()
    trade_bil_total = 0
    ### first loop over all the flows
    for flow in range(0,len(flows)):
        ### check if country is within list of countries
        country_import = flows.iso3_D.iloc[flow]
        country_export = flows.iso3_O.iloc[flow]
        industry = sectors[flows.iloc[flow]['Industries']-1]
        trade_share = flows.iloc[flow]['v_share_trade']
        if list_countries.count(country_import) == 0 or list_countries.count(country_export) == 0:
            continue
        else:
            ### new row
            new_row = A_matrix_append.loc[(country_export, industry), country_import]*(1-trade_share)
            ### replace A matrix
            A_matrix_append.loc[(country_export, industry), country_import] =  new_row.to_list()

            ### get the Z matrix
            trade_bil = Z_matrix.loc[(country_export, industry), country_import].sum()*(trade_share)
            trade_bil_total =trade_bil_total+trade_bil

    return A_matrix_append, trade_bil_total

def modify_B_matrix(B_matrix, Z_matrix, flows, list_countries, sectors):
    B_matrix_append = B_matrix.copy()
    trade_bil_total = 0
    ### first loop over all the flows
    for flow in range(0,len(flows)):
        ### check if country is within list of countries
        country_import = flows.iso3_D.iloc[flow]
        country_export = flows.iso3_O.iloc[flow]
        industry = sectors[flows.iloc[flow]['Industries']-1]
        trade_share = flows.iloc[flow]['v_share_trade']
        if list_countries.count(country_import) == 0 or list_countries.count(country_export) == 0:
            continue
        else:
            ### new row
            new_row = B_matrix_append.loc[(country_export, industry), country_import]*(1-trade_share)
            ### replace A matrix
            B_matrix_append.loc[(country_export, industry), country_import] =  new_row.to_list()

            ### get the Z matrix
            trade_bil = Z_matrix.loc[(country_export, industry), country_import].sum()*(trade_share)
            trade_bil_total =trade_bil_total+trade_bil

    return B_matrix_append, trade_bil_total

def modify_A_matrix_individual(A_matrix, Z_matrix, flows, list_countries, sectors):
    trade_bil_total = 0
    if len(flows) == 0:
        return A_matrix, trade_bil_total
    else:
        ### first loop over all the flows
        for flow in range(0,len(flows)):
                ### check if country is within list of countries
                country_import = flows.iso3_D.iloc[flow]
                country_export = flows.iso3_O.iloc[flow]
                industry = sectors[flows.iloc[flow]['Industries']-1]
                trade_share = flows.iloc[flow]['v_share_trade']
                if list_countries.count(country_import) == 0 or list_countries.count(country_export) == 0:
                    continue
                else:
                    ### new row
                    new_row = A_matrix.loc[(country_export, industry), country_import]*(1-trade_share)
                    ### replace A matrix
                    A_matrix.loc[(country_export, industry), country_import] =  new_row.to_list()

                    ### get the Z matrix
                    trade_bil = Z_matrix.loc[(country_export, industry), country_import].sum()*(trade_share)
                    trade_bil_total =trade_bil_total+trade_bil

        return A_matrix, trade_bil_total


def modify_Y_matrix(Y_matrix, flows, list_countries, sectors):
    Y_matrix_append = Y_matrix.copy()
    C_bil_total = 0
    ### first loop over all the flows
    for flow in range(0,len(flows)):
        ### check if country is within list of countries
        country_import = flows.iso3_D.iloc[flow]
        country_export = flows.iso3_O.iloc[flow]
        industry = sectors[flows.iloc[flow]['Industries']-1]
        trade_share = flows.iloc[flow]['v_share_trade']
        if list_countries.count(country_import) == 0 or list_countries.count(country_export) == 0:
            continue
        else:
            ### new row
            new_row = Y_matrix_append.loc[(country_export, industry), country_import]*(1-trade_share)

            ### replace A matrix
            Y_matrix_append.loc[(country_export, industry), country_import] =  new_row.to_list()

            ### get the Z matrix
            C_bil = Y_matrix.loc[(country_export, industry), country_import].sum()*(trade_share)
            C_bil_total = C_bil_total + C_bil

    return Y_matrix_append, C_bil_total

def process_output(df):
    df['trade_total'] = df['trade_ind'] + df['C']


    df['Dind_total_bw'] = df['Dind_int_bw'] + df['Dind_C_bw']
    df['Dind_iso3_bw'] = df['Dind_iso3_int_bw'] + df['Dind_iso3_C_bw']
    df['Dind_row_bw'] = df['Dind_row_int_bw'] + df['Dind_row_C_bw']

    df['Dind_total_fw'] = df['Dind_int_fw'] + df['C']
    df['Dind_iso3_fw'] = df['Dind_iso3_int_fw'] + df['DC_iso3_fw']
    df['Dind_row_fw'] = df['Dind_row_int_fw'] + df['DC_row_fw']

    df['Dind_total'] = df['Dind_total_bw'] + df['Dind_total_fw']
    df['Dind_iso3'] = df['Dind_iso3_bw'] + df['Dind_iso3_fw']
    df['Dind_row'] = df['Dind_row_bw'] + df['Dind_row_fw']

    df['multiplier'] = np.round(df['Dind_total']/df['trade_total'], 3)
    df['multiplier_dom'] = np.round(df['Dind_iso3']/df['trade_total'], 3)
    df['multiplier_row'] = np.round(df['Dind_row']/df['trade_total'], 3)

    df['frac_row_dom'] = df['Dind_row']  / df['Dind_iso3']
    df['frac_bw_fw'] = df['Dind_total_bw']  / df['Dind_total_fw']

    return df


def get_country_import_demand_export(eora, country, sectors, list_countries):
    ### only commodity sectors output and
    country_import = eora.A[str(country)][eora.A[str(country)].index.get_level_values('sector').isin(sectors)].iloc[:, :11]

    #### demand
    demand = eora.Y[str(country)][['Household final consumption P.3h','Non-profit institutions serving households P.3n','Government final consumption P.3g']]
    demand = demand[demand.index.get_level_values('sector').isin(sectors)]
    sector_df = pd.DataFrame({'sector':sectors})
    demand_country = sector_df.merge(demand.sum(axis = 1).groupby('sector').sum().reset_index(), on = 'sector')[0]


    ## export
    countries_df = pd.DataFrame({'countries':list_countries})
    countries_df = countries_df[countries_df['countries']!=str(country)]
    export_matrix = eora.Z.loc[str(country)]
    export_matrix = export_matrix[export_matrix.index.get_level_values('sector').isin(sectors)]
    export_country = export_matrix[countries_df.countries.to_list()].sum(axis = 1)

    return country_import, demand_country, export_country

def fill_imports_port(port_import_share):
    country_import_append = country_import.copy()
    country_import_append = country_import_append*0.0
    for flow in range(0,len(port_import_share)):
        ### check if country is within list of countries
        D_import = str(country)
        O_export = port_import_share.iso3_O.iloc[flow]
        industry = sectors[port_import_share.iloc[flow]['Industries']-1]
        trade_share = port_import_share.iloc[flow]['v_share_trade']

        if list_countries.count(D_import) == 0 or list_countries.count(O_export) == 0:
            continue
        else:
            ### new row
            new_row = country_import.loc[O_export].loc[industry]*trade_share
            ### replace the row
            country_import_append.loc[(O_export), (industry)] =  new_row.to_list()

    return country_import_append


def port_import_coef(country_import_append, country):
    import_coef  = country_import_append[country_import_append.index.get_level_values('region') != str(country)]
    countries_row = import_coef.index.get_level_values('region').unique()
    start_matrix = np.zeros((11, 11))
    for country_row in countries_row:
        matrix = import_coef[import_coef.index.get_level_values('region')==str(country_row)]
        start_matrix = start_matrix+matrix.values
    import_coef =start_matrix

    return import_coef


### Paths
path_output_processed = 'Input/'
path_output = 'Output/'
eora_storage = 'Input/EORA/Eora26_2015_bp'

### load eora
eora = pymrio.parse_eora26(year=2015, path=eora_storage)
eora.calc_all()

### regions
list_countries = eora.get_regions().to_list()

# port nodes
nodes_maritime =gpd.read_file(path_output_processed+'nodes_maritime.gpkg',driver = 'GPKG')
nodes_port = nodes_maritime[nodes_maritime['infra']=='port']

### transport model output
mar_network = dd.read_csv(path_output_processed+'port_trade_network.csv')

### replace Wuhan with Shanghai
mar_network['from_id'] = mar_network['from_id'].replace({'port451_in':'port1188_in'})
mar_network['from_id'] = mar_network['from_id'].replace({'port451_land':'port1188_land'})
mar_network['to_id'] = mar_network['to_id'].replace({'port451_land':'port1188_land'})
mar_network['to_id'] = mar_network['to_id'].replace({'port451_out':'port1188_out'})
mar_network['id'] = mar_network['from_id'].str.split('_',n = 1, expand = True)[0]

### extract port ids that are in dataset
ports_id = mar_network['from_id'].str.split('_',n = 1, expand = True)[0].unique().compute()
nodes_port_trade = nodes_port[nodes_port['id'].isin(ports_id)].reset_index(drop = True)


### get the original matrix
A_matrix, Y_matrix, Z_matrix, x_vector, v = eora.A, eora.Y, eora.Z, eora.x, eora.VA.F.values.sum(axis = 0)

### sectors
sectors = A_matrix.columns[0:11].get_level_values('sector').tolist()

### global output
output = x_vector['indout'].reset_index()
output['C'] =Y_matrix.sum(axis = 1).values
C_countries = Y_matrix.sum(axis = 0).reset_index().groupby(['region'])[0].sum().reset_index().rename(columns = {0:'C_dom','region':'iso3'})
Y_matrix_com = Y_matrix.reset_index()[Y_matrix.reset_index()['sector'].isin(sectors)]
C_com_countries =Y_matrix_com.drop(columns = Y_matrix_com.columns[0:2].values).sum(axis = 0).reset_index().groupby(['region'])[0].sum().reset_index().rename(columns = {0:'C_com_dom','region':'iso3'})

output_country = output.groupby(['region'])[['indout']].sum().reset_index().rename(columns = {'region':'iso3'})
output_country = output_country.merge(C_countries, on = 'iso3').merge(C_com_countries, on = 'iso3')


### get the original matrix
A_matrix, Y_matrix, Z_matrix, x_vector, v = eora.A, eora.Y, eora.Z, eora.x, eora.VA.F.values.sum(axis = 0)

### sectors
sectors = A_matrix.columns[0:11].get_level_values('sector').tolist()
### rows of index
rows_index = x_vector.reset_index()

### diagonal
I_full = np.zeros(A_matrix.shape)
np.fill_diagonal(I_full, 1)

###  create Ghosian matrices
col_vector = np.array(eora.x.indout.to_list()).T
d = np.diag(col_vector)
x_values = np.nan_to_num(I_full/d)
B = np.dot(x_values,Z_matrix.values)

### make a nice dataframe from B
B_matrix = A_matrix.copy()
B_matrix[:] = B
x_vector['v'] = np.dot(np.array(x_vector.indout.to_list()),(I_full - B)) ### value added


###### OUTPUT MULTIPLIER ######
multiplier_ports = pd.DataFrame()
#### Loop over ports
for i in range(0, len(nodes_port_trade)):
    port_select = nodes_port_trade.iloc[i]
    print(i, port_select['name'], datetime.datetime.now())
    country_domestic = port_select['iso3']

    ### get ports flows
    mar_network_port = mar_network[mar_network['id']==port_select['id']].compute()

    ### flows
    all_flows = mar_network_port.groupby(['iso3_O','iso3_D','Industries'])['v_share_trade'].sum().reset_index()

    if country_domestic not in list_countries:
        ### check if country data is in EORA
        continue
    else:
        ### original output
        global_output, dom_output, row_output = x_vector['indout'].sum(), x_vector.loc[str(country_domestic)].sum()[0], x_vector['indout'].sum() - x_vector.loc[str(country_domestic)].sum()[0]
        global_C = Y_matrix.sum().sum()
        dom_C = Y_matrix[str(country_domestic)].sum().sum()
        row_C = global_C - dom_C


        #### RUN INDUSTRY OUTPUT ####
        A_matrix_append, trade_total =  modify_A_matrix(A_matrix, Z_matrix, all_flows, list_countries, sectors)
        output_new = np.dot(np.linalg.inv(I_full - A_matrix_append.values),Y_matrix.values)
        global_output_new = output_new.sum()
        ### domestic output
        output_new_dom = output_new.sum(axis = 1)[rows_index[rows_index['region']==str(country_domestic)].index].sum()

        print('Industry output done: ',port_select['name'])

        ### RUN CONSUMPTION ####
        Y_matrix_append, C_bil_total =  modify_Y_matrix(Y_matrix, all_flows, list_countries, sectors)
        output_new_C = np.dot(np.linalg.inv(I_full - A_matrix.values),Y_matrix_append.values)
        global_output_new_C = output_new_C.sum()
        ### domestic output C
        output_new_dom_C = output_new_C.sum(axis = 1)[rows_index[rows_index['region']==str(country_domestic)].index].sum()

        ### final consumption
        global_C_new = Y_matrix_append.sum().sum()
        dom_C_new = Y_matrix_append[str(country_domestic)].sum().sum()
        row_C_new = global_C_new-dom_C_new


        print('Consumption done: ',port_select['name'])


        #### RUN FORWARD ####
        B_matrix_append, trade_total =  modify_B_matrix(B_matrix, Z_matrix, all_flows, list_countries, sectors)

        x_vector['indin_trade'] = np.dot(np.array(x_vector.v.to_list()),np.linalg.inv(I_full - B_matrix_append))
        x_vector['total_diff'] = x_vector['indout']-x_vector['indin_trade']

        total_diff = x_vector['total_diff'].sum()
        total_diff_dom = x_vector.loc[str(country_domestic)]['total_diff'].sum()
        total_diff_row = x_vector['total_diff'].sum()-x_vector.loc[str(country_domestic)]['total_diff'].sum()

        print('Forward done: ', port_select['name'])
        ### create a df
        backward_forward_port = pd.DataFrame({'name':[port_select['name']],'id':[port_select['id']],'iso3':[port_select['iso3']],'trade_ind':[trade_total],'C':[C_bil_total],'Dind_int_bw':[global_output-global_output_new],'Dind_iso3_int_bw':[dom_output - output_new_dom],'Dind_row_int_bw':[row_output - (global_output_new-output_new_dom)],'Dind_C_bw':[global_output - global_output_new_C],'Dind_iso3_C_bw':[dom_output-output_new_dom_C],'Dind_row_C_bw':[row_output - (global_output_new_C-output_new_dom_C)],'DC_iso3_fw':[dom_C - dom_C_new],'DC_row_fw':[row_C - row_C_new], 'Dind_int_fw':[total_diff],'Dind_iso3_int_fw':[total_diff_dom],'Dind_row_int_fw':[total_diff_row]})
        backward_forward_port = process_output(backward_forward_port)

        print(port_select['name'], C_bil_total, trade_total , backward_forward_port['Dind_total'][0], backward_forward_port['Dind_total_bw'][0] , backward_forward_port['Dind_total_fw'][0], 'multiplier:', backward_forward_port['multiplier'][0] )
        print(datetime.datetime.now())
        multiplier_ports = pd.concat([multiplier_ports, backward_forward_port],ignore_index = True, sort = False)
        ####


multiplier_ports.to_csv(path_output+'output_multiplier.csv', index = False)
output_country.to_csv(path_output+'country_indout_C.csv', index = False)

###### IMPORT MULTIPLIER ######

#### import multiplier
country_list_import = mar_network['iso3_D'].unique().compute().tolist()

### create some standard matrices
import_coef_all = pd.DataFrame()
import_multiplier_all = pd.DataFrame()
import_requirement_all = pd.DataFrame()

### loop over countries
for country in country_list_import:
    ### check if country data is in EORA
    if country not in list_countries:
        continue
    else:
        print('RUN', country)
        ### A matrix
        A_country = eora.A[str(country)].loc[str(country)].iloc[0:11,0:11]
        sectors = A_country.columns[0:11]

        ### get country imports, demand and exports
        country_import, demand_country, export_country = get_country_import_demand_export(eora, country, sectors, list_countries)

        #### get the trade data for the country
        country_import_share = mar_network[mar_network['iso3_D']==str(country)].compute()
        country_import_share = country_import_share[country_import_share['flow']=='port_import']

        #### number of ports to loop over
        country_ports = country_import_share[country_import_share['flow']=='port_import'].id.unique()
        print('country',str(country),'has ',len(country_ports),'ports to loop over')

        import_coef_country = pd.DataFrame()
        import_multiplier_country = pd.DataFrame()
        import_requirement_country = pd.DataFrame()
        for port in country_ports:
            I = np.zeros((11, 11))
            np.fill_diagonal(I, 1)
            ### extract port imports
            port_import_share = country_import_share[country_import_share['id']==str(port)]

            ### country imports matrix for port
            country_import_append = fill_imports_port(port_import_share)

            ### get the import coefficient per port
            import_coef =  port_import_coef(country_import_append, country)

            ### port
            step1 = np.dot(np.ones(11),import_coef)
            ### step 2 are the import coefficents
            step2 = np.dot(step1,np.linalg.inv(I - A_country.values))
            multiplier = step2.sum()
            ## step 3
            import_requirement_export = np.dot(step2,np.array([export_country.values]).T)
            import_requirement_demand = np.dot(step2,np.array([demand_country.values]).T)

            #### import coefficient df
            import_coef_country_port = pd.DataFrame({'Industries':np.linspace(1,11,11).astype(int),'sector':sectors,'Import_coef':step2})
            import_coef_country_port['id'] = str(port)
            import_coef_country_port['iso3_import'] = str(country)

            ## import requirements for export and final demand
            import_requirement_country_port = pd.DataFrame({'iso3_import': str(country),'id': str(port),'Import_export':[import_requirement_export[0]],'Total_export':[export_country.values.sum()],'Import_demand':[import_requirement_demand[0]],'Total_demand':[demand_country.values.sum()]})

            #### import total multiplier
            import_multiplier_country_port = pd.DataFrame({'iso3_import': str(country),'id': str(port),'Import_multiplier':[multiplier]})

            ### concat
            import_coef_country = pd.concat([import_coef_country, import_coef_country_port], ignore_index = True, sort = False)
            import_requirement_country = pd.concat([import_requirement_country, import_requirement_country_port], ignore_index = True, sort = False)
            import_multiplier_country = pd.concat([import_multiplier_country, import_multiplier_country_port], ignore_index = True, sort = False)


            print(country, port, import_requirement_export[0],import_requirement_demand[0],multiplier)


        #### concat all
        import_coef_all = pd.concat([import_coef_all, import_coef_country], ignore_index = True, sort = False)
        import_requirement_all = pd.concat([import_requirement_all, import_requirement_country], ignore_index = True, sort = False)
        import_multiplier_all = pd.concat([import_multiplier_all, import_multiplier_country], ignore_index = True, sort = False)




import_coef_all.to_csv(path_output+'import_coef_sector.csv', index = False)
import_multiplier_all.to_csv(path_output+'import_multiplier.csv', index = False)
import_requirement_all.to_csv(path_output+'import_requirement.csv', index = False)
