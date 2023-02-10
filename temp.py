import pandas as pd
import p_median_12
import matplotlib.pyplot as plt
import bimodal

# # read the data from Distribution_Normal.csv
# u=180
# sigma=30
# p_median_symmetric = p_median_12.solver(12*12, 3, 180, 30)
# u = p_median_12.find_u_triangle(u-3*sigma,u-sigma,u+3*sigma) # u for right skewed triangle distribution
# p_median_right_skewed = p_median_12.solver(12*12, 3, u, 30)
# u = p_median_12.find_u_triangle(u-3*sigma,u+sigma,u+3*sigma) # u for left skewed triangle distribution
# p_median_left_skewed = p_median_12.solver(12*12, 3, u, 30)

# # from every row in the data_df substract the objective function value from the p_median_symmetric
# data_df = pd.read_csv("./iterate_k_results/Distribution_Normal.csv")
# data_df["obj_value"] = data_df["obj_value"].apply(lambda x: (x - p_median_symmetric)*100/p_median_symmetric)
# data_df.rename(columns={"obj_value": "per_diff"}, inplace=True)
# # plot the data as a line plot with the "per_diff" column as the y axis and the "per_red" column as the x axis
# plt.plot(data_df["k"], data_df["per_diff"])

# data_df = pd.read_csv("./iterate_k_results/Distribution_Uniform.csv")
# data_df["obj_value"] = data_df["obj_value"].apply(lambda x: (x - p_median_symmetric)*100/p_median_symmetric)
# data_df.rename(columns={"obj_value": "per_diff"}, inplace=True)
# # plot the data as a line plot with the "per_diff" column as the y axis and the "per_red" column as the x axis
# plt.plot(data_df["k"], data_df["per_diff"])

# data_df = pd.read_csv("./iterate_k_results/Distribution_Triangular_Symmetric.csv")
# data_df["obj_value"] = data_df["obj_value"].apply(lambda x: (x - p_median_symmetric)*100/p_median_symmetric)
# data_df.rename(columns={"obj_value": "per_diff"}, inplace=True)
# # plot the data as a line plot with the "per_diff" column as the y axis and the "per_red" column as the x axis
# plt.plot(data_df["k"], data_df["per_diff"])


# data_df = pd.read_csv("./iterate_k_results/Distribution_Triangular_Right_Skewed.csv")
# data_df["obj_value"] = data_df["obj_value"].apply(lambda x: (x - p_median_right_skewed)*100/p_median_right_skewed)
# data_df.rename(columns={"obj_value": "per_diff"}, inplace=True)
# # plot the data as a line plot with the "per_diff" column as the y axis and the "per_red" column as the x axis
# plt.plot(data_df["k"], data_df["per_diff"])

# data_df = pd.read_csv("./iterate_k_results/Distribution_Triangular_Left_Skewed.csv")
# data_df["obj_value"] = data_df["obj_value"].apply(lambda x: (x - p_median_left_skewed)*100/p_median_left_skewed)
# data_df.rename(columns={"obj_value": "per_diff"}, inplace=True)
# # plot the data as a line plot with the "per_diff" column as the y axis and the "per_red" column as the x axis
# plt.plot(data_df["k"], data_df["per_diff"])


# # set the x and y labels
# plt.xlabel("k")
# plt.ylabel("% Difference between Stochastic and Deterministic")
# # set the title
# plt.title("Diff b/w Stochastic and Deterministic for unimodal vs K u=180 sigma=30")
# # set different legends for the plot
# plt.legend(["Normal", "Uniform", "Triangular Symmetric", "Right Skewed", "Left Skewed"])

# # plt.show()
# # save the plot as a png file
# plt.savefig("Stochastic_vs_Deterministic_Unimodal.png")
# plt.close()

# read the data from Distribution_Normal.csv
u1=180
sigma1=30
u2=300
sigma2=30
mixing_prob_set = [0.2, 0.4, 0.5, 0.6, 0.8]
for mixing_prob in mixing_prob_set:
    u = bimodal.find_u_bimodal(u1, u2, mixing_prob)
    p_median_bimodal = p_median_12.solver(12*12, 3, u, 30)
    data_df = pd.read_csv("./iterate_k_results/Distribution_Normal_bimodal_{}_{}_{}_{}_{}.csv".format(u1, sigma1, u2, sigma2, mixing_prob))
    data_df["obj_value"] = data_df["obj_value"].apply(lambda x: (x - p_median_bimodal)*100/p_median_bimodal)
    data_df.rename(columns={"obj_value": "per_diff"}, inplace=True)
    # plot the data as a line plot with the "per_diff" column as the y axis and the "per_red" column as the x axis
    plt.plot(data_df["k"], data_df["per_diff"])

# set the x and y labels
plt.xlabel("Number of Iterations(k)")
plt.ylabel("% Difference between Stochastic and Deterministic")
# set the title
# plt.title("Diff b/w Stochastic and Deterministic vs K for Bimodal Distribution N~(180,30) N~(300,30)")
# set different legends for the plot
def get_skewness(u1, sigma1, u2, sigma2, weight1):
    return weight1*(3*u1*(sigma1**2)+u1**3)+(1-weight1)*(3*u2*(sigma2**2)+u2**3)
legend_list = ["$\gamma={}$,$\\alpha={}$,$\mu_1={}$,$\sigma_1={}$,$\mu_2={}$,$\sigma_2={}$".format(get_skewness(u1, sigma1, u2, sigma2, i),i,u1,sigma1,u2,sigma2) for i in mixing_prob_set]
# change location of the legend to not overlap with the plot
plt.legend(loc='upper right')
plt.legend(legend_list)

# set y axis limit
plt.ylim(0, 20)

# plt.show()
# save the plot as a png file
plt.savefig("Stochastic_vs_Deterministic_Bimodal.png",bbox_inches='tight')
plt.close()




