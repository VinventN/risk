from ecdf import *
def plt_func():
    i = 0
    i_max = len(result)
    x_max_list = []
    y_max_list = []
    while i < i_max:
        plt.plot(result[i][0], result[i][1](result[i][0]), label=result[i][5])
        x_max_list.append(result[i][2])
        y_max_list.append(result[i][3])
        i += 1
    # while j < i_max:
    #     x_max_list.append(result[i][2])
    #     y_max_list.append(result[i][3])
    #     print(x_max_list, y_max_list)
    #     j += 1
    if x_max_list != []:
        plt.xlim(0, max(x_max_list))
        plt.ylim(ymin=0)
    plt.legend()


def ecdf_1(i=0):
    kde = result[i][1]
    x_grid = np.linspace(0, 500, 1000)
    kde = kde.evaluate(x_grid)
    cdf = np.cumsum(kde)
    cdf = cdf / cdf[-1]
    if result[i][5] == 'hpp_model':
        plt.plot(cdf, label='HPP')
    if result[i][5] == 'npp_model':
        plt.plot(cdf, label='NPP')
    if result[i][5] == 'fhpp_model':
        plt.plot(cdf, label='FHPP')
    if result[i][5] == 'fnpp_model':
        plt.plot(cdf, label='FNPP')
    plt.legend()

# print(result)
# kde_single(runs=10)

def open_csv(name='file.csv'):
    with open(name, newline='') as inputfile:
        global csv_result
        xx = []
        for row in csv.reader(inputfile):
            xx.append(row[0])
    for i in range(len(xx)):
        xx[i] = float(xx[i])
    xx = np.array(xx)
    xx = [xx]
    xs = np.linspace(0, T, 5*T)
    xx.append(xs)
    global csv_result
    csv_result = []
    csv_result.append(xx)
    temp = csv_result[0][0]
    csv_result[0][0] = csv_result[0][1]
    sample = temp
    # density = gaussian_kde(sample, bw_method=0.005)
    density = gaussian_kde(sample, bw_method=0.0075)
    # density.covariance_factor = lambda : 0.01
    density._compute_covariance()
    csv_result[0][1] = density
    csv_result[0].append(100)
    csv_result[0].append(0.06)
    csv_result[0].append(temp)
    return csv_result

open_csv(name='fnpp_0_0.9_500.csv')
csv_result[0].append('fnpp_0_500')
mtfhpp_0_500 = csv_result
result += mtfhpp_0_500

def ecdf2(label='na'):
    i = 0
    i_max = len(result)
    x_max_list = []
    plt.figure(figsize=(8,3))
    while i < i_max:
        kde = result[i][1]
        x_grid = np.linspace(0, 500, 1000)
        kde = kde.evaluate(x_grid)
        cdf = np.cumsum(kde)
        cdf = cdf / cdf[-1]
        plt.plot(cdf, label=result[i][5])
        i += 1
    plt.legend()
    plt.ylim(0, 1)

def hist():
    x_grid = np.linspace(0, 1000, 1000)
    y = result[0][4]
    cdf = np.cumsum(y)
    
def cdf(x=result[0][4], plot=True, *args, **kwargs):
    x, y = sorted(x), np.arange(len(x)) / len(x)
    # plt.hist(x, bins=1000, density=True, alpha=0.4)
    return plt.plot(x, y, *args, **kwargs, label='ecdf') if plot else (x, y)

def ecdf():
    kde = result[0][1]
    x_grid = np.linspace(0, 1000, 1000)
    kde = kde.evaluate(x_grid)
    cdf_kde = np.cumsum(kde)
    cdf_kde = cdf_kde / cdf_kde[-1]
    plt.plot(cdf_kde, label='ecdf')

def fit_cdf():
    og_result = result[0][4]
    og_result_len = len(og_result)
    new_result = [x for x in og_result if x < 1000]
    # ig
    invgauss_mu, invgauss_loc, invgauss_scale = stats.distributions.invgauss.fit(new_result)
    invgauss_x = np.linspace(0,1000,10000)
    fitted_invgauss_cdf = stats.distributions.invgauss.cdf(invgauss_x, invgauss_mu, invgauss_loc, invgauss_scale)
    plt.plot(invgauss_x, fitted_invgauss_cdf, label='inverse gaussian')
    # gig
    geninvgauss_p, geninvgauss_b, geninvgauss_loc, geninvgauss_scale = stats.distributions.geninvgauss.fit(new_result)
    geninvgauss_x = np.linspace(0,1000,10000)
    fitted_geninvgauss_cdf = stats.distributions.geninvgauss.cdf(geninvgauss_x, geninvgauss_p, geninvgauss_b, geninvgauss_loc, geninvgauss_scale)
    plt.plot(geninvgauss_x, fitted_geninvgauss_cdf, label='generalised inverse gaussian')


# cdf()
# fit_cdf()
# plt.legend()
# plt.ylim(0, 1)
# plt.xlim(0,200)
# plt.show()


def ruin_num():
    x = 20
    y = x
    time_of_ruin = []
    while True:
        time_of_ruin.append(sum(i < x for i in list(result[0][4]))/20000)
        x += y
        if x == 240:
        # if time_of_ruin[-1] == 1:
            time_of_ruin.pop()
            time_of_ruin.append(1 - time_of_ruin[-1])
            break
        # print('<', x, ':', sum(i < x for i in list(result[0][4])))
    # if x == 1000:
        
    # time_list = (time_of_ruin, time_of_ruin)
    df = pd.DataFrame([time_of_ruin]) 
    df.to_csv('aa.csv', index=False, header=False)

ruin_num()


# def ecdf():
#     i = 0
#     i_max = len(result)
#     x_max_list = []
#     # plt.figure(figsize=(8,6))
#     while i < i_max:
#         kde = result[i][1]
#         x_grid = np.linspace(0, 500, 1000)
#         kde = kde.evaluate(x_grid)
#         cdf = np.cumsum(kde)
#         cdf = cdf / cdf[-1]
#         if result[i][5] == 'hpp_model':
#             plt.plot(cdf, label='HPP')
#             x_max_list.append(100)
#         elif result[i][5] == 'npp_model':
#             plt.plot(cdf, label='NPP')
#             x_max_list.append(400)
#         elif result[i][5] == 'fhpp_model':
#             plt.plot(cdf, label='FHPP')
#             x_max_list.append(600)
#         elif result[i][5] == 'fnpp_model':
#             plt.plot(cdf, label='FNPP')
#             x_max_list.append(600)
#         else:
#             plt.plot(cdf)
#         i += 1
#     plt.legend()
#     plt.xlim(0, max(x_max_list))
#     plt.ylim(0, 1)
#     # kde = result[0][1]
#     # x_grid = np.linspace(0, 500, 1000)
#     # kde = kde.evaluate(x_grid)
#     # cdf = np.cumsum(kde)
#     # cdf = cdf / cdf[-1]
#     # return cdf
