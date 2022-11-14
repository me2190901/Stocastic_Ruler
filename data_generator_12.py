import numpy as np, pandas as pd
import os
import p_median

# global data_df
# data_df = pd.DataFrame(columns = ['Distribution_type', 'U', 'Sigma',"U2","Sigma2","mixing_prob", "limitk", "per_red", "Objective_function_value","Solution_locations", "Running_time(sec)" ])

# def make_biomodal_input_command(U1,Sigma1, U2,Sigma2,mixing_prob,limitk, per_red):
#     if(U1>U2):
#         U1,U2 = U2,U1 
#         Sigma1, Sigma2 = Sigma2, Sigma1
#     temp = ["Normal_bimodal",U1,Sigma1, U2, Sigma2, mixing_prob, limitk, per_red]
#     temp = list(map(str, temp))
#     return " ".join(temp)

# def write_input(text, file_name):
#     f = open(file_name,'w')
#     f.write(text)
#     f.close()

# def read_file(file_name):
#     f = open(file_name,'r')
#     data = sorted([item.strip() for item in f.readlines()])
#     f.close()
#     return data

# def process_solution_string(s):
#     Locations = []
#     second_comma = 0
#     for i in range(len(s)):
#         if(s[i]==","):
#             second_comma +=1
#         if(second_comma==2):
#             second_comma = i
#             break
#     fourth_comma = 0
#     for i in range(len(s)):
#         if(s[i]==","):
#             fourth_comma +=1
#         if(fourth_comma==4):
#             fourth_comma = i
#             break
#     Locations = [ s[0:second_comma] , s[second_comma+1:fourth_comma] , s[fourth_comma+1:] ]
#     for i in range(len(Locations)):
#         Locations[i] = Locations[i].strip().strip("(").strip(")").replace(" ","").replace(","," ")
#     return "|".join(Locations)

# def process_output_data(data):
#     Data = [item.split() for item in data]
#     print(Data)
#     obj_value = float(Data[0][-1])
#     run_time = float(Data[1][-2])
#     per_red = float(Data[3][-1])
#     locations = data[-1][18:].strip()[1:-1]
#     locations = process_solution_string(locations)
#     return [per_red, obj_value, locations, run_time]

# bimodel_input_set = [[(180,30),(300,60)], [(180,30),(300,50)], [(180,30),(300,40)],[(180,30),(300,30)], [(180,30),(300,20)], [(180,30),(120,30)],[(180,30),(120,20)] ,[(180,30),(120,10)]]
# bimodel_input_set.extend([[(180,20),(300,60)], [(180,20),(300,50)], [(180,20),(300,40)],[(180,20),(300,30)],[(180,20),(300,20)], [(180,20),(120,30)],[(180,20),(120,20)] ,[(180,20),(120,10)]])
# bimodel_input_set.extend([[(180,40),(300,60)], [(180,40),(300,50)], [(180,40),(300,40)],[(180,40),(300,30)],[(180,40),(300,20)], [(180,40),(120,30)],[(180,40),(120,20)] ,[(180,40),(120,10)]])


# limitk = 300

# for input_set in bimodel_input_set:
#     U1 = input_set[0][0]
#     Sigma1 = input_set[0][1]
#     U2 = input_set[1][0]
#     Sigma2 = input_set[1][1]
#     mixing_prob_set = [0.2, 0.4, 0.5 ,0.6, 0.8]
#     per_red_set = [5,15,25,40,50]
#     for mixing_prob in mixing_prob_set:
#         for per_red in per_red_set:
#             write_input(make_biomodal_input_command(U1,Sigma1, U2, Sigma2, mixing_prob, limitk, per_red),"temp_input.txt" )
#             os.system("python facility_problem.py < temp_input.txt > temp_output.txt")
#             data = read_file("temp_output.txt")
#             out_ = process_output_data(data)
#             data_df.loc[len(data_df.index)] = ["Normal_bimodal", U1, Sigma1, U2, Sigma2, mixing_prob, limitk] + out_
# data_df.to_csv("bimodal_stochastic_results.csv")

data_df_r = pd.read_csv("bimodal_stochastic_results_12.csv")
# For formatting the output
df = pd.DataFrame(columns = ['Distribution','Per_Redn',"0.2","0.4","0.5","0.6","0.8"])

# iterate over the data and create a dataframe
for i in range(0, len(data_df_r), 25):
    for j in range(i,i+5):
        # append data to the dataframe
        string=""
        if j%5 == 0:
            string = data_df_r.iloc[j,1]+"_N~({},{}),N~({},{})".format(data_df_r.iloc[j,2],data_df_r.iloc[j,3],data_df_r.iloc[j,4],data_df_r.iloc[j,5])
        else:
            string = ""
        df.loc[len(df)] = [string,data_df_r.iloc[j,8],data_df_r.iloc[j,9],data_df_r.iloc[j+5,9],data_df_r.iloc[j+10,9],data_df_r.iloc[j+15,9],data_df_r.iloc[j+20,9]]
    u1=data_df_r.iloc[j,2]
    sigma1=data_df_r.iloc[j,3]
    u2=data_df_r.iloc[j,4]
    sigma2=data_df_r.iloc[j,5]
    u_1=p_median.find_u_bimodal(u1,u2,0.2)
    u_2=p_median.find_u_bimodal(u1,u2,0.4)
    u_3=p_median.find_u_bimodal(u1,u2,0.5)
    u_4=p_median.find_u_bimodal(u1,u2,0.6)
    u_5=p_median.find_u_bimodal(u1,u2,0.8)
    obj1=p_median.solver(36,3,u_1,30,u1=u1,sigma1=sigma1,u2=u2,sigma2=sigma2,weight=0.2)
    obj2=p_median.solver(36,3,u_2,30,u1=u1,sigma1=sigma1,u2=u2,sigma2=sigma2,weight=0.4)
    obj3=p_median.solver(36,3,u_3,30,u1=u1,sigma1=sigma1,u2=u2,sigma2=sigma2,weight=0.5)
    obj4=p_median.solver(36,3,u_4,30,u1=u1,sigma1=sigma1,u2=u2,sigma2=sigma2,weight=0.6)
    obj5=p_median.solver(36,3,u_5,30,u1=u1,sigma1=sigma1,u2=u2,sigma2=sigma2,weight=0.8)

    df.loc[len(df)] = ["","P-median",obj1,obj2,obj3,obj4,obj5]
    # add an empty row
    df.loc[len(df)] = ["","","","","","",""]

# save the dataframe to csv file
df.to_csv('bimodal_combined_results_12.csv', index=False)