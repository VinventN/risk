from open_save import *
import matplotlib.ticker as mtick
og_result = result[0][4]
og_result_len = len(og_result)
new_result = [x for x in og_result if x < 1000]
data = plt.hist(new_result, bins=ceil(max(new_result)/2), density=True, alpha=0.4)
sns.kdeplot(new_result, label='kde')

invgauss_mu, invgauss_loc, invgauss_scale = stats.distributions.invgauss.fit(new_result)
invgauss_x = np.linspace(0,1000,10000)
fitted_invgauss = stats.distributions.invgauss.pdf(invgauss_x, invgauss_mu, invgauss_loc, invgauss_scale)
plt.plot(invgauss_x, fitted_invgauss, label='inverse gaussian')

geninvgauss_p, geninvgauss_b, geninvgauss_loc, geninvgauss_scale = stats.distributions.geninvgauss.fit(new_result)
geninvgauss_x = np.linspace(0,1000,10000)
fitted_geninvgauss = stats.distributions.geninvgauss.pdf(geninvgauss_x, geninvgauss_p, geninvgauss_b, geninvgauss_loc, geninvgauss_scale)
plt.plot(geninvgauss_x, fitted_geninvgauss, label='generalised inverse gaussian')

plt.ylim(ymin=0)
plt.xlim(xmin=0)
sns.despine()
plt.legend()
plt.tight_layout()
plt.savefig('fhpp_1_0.9_fit.png')

# gamma_a, gamma_loc, gamma_scale = stats.distributions.gamma.fit(new_result)
# gamma_x = np.linspace(0,1000,10000)
# fitted_gamma = stats.distributions.gamma.pdf(gamma_x, gamma_a, gamma_loc, gamma_scale)
# plt.plot(gamma_x, fitted_gamma, label='gamma')

# gengamma_a, gengamma_c, gengamma_loc, gengamma_scale = stats.distributions.gengamma.fit(new_result)
# gengamma_x = np.linspace(0,1000,10000)
# fitted_gengamma = stats.distributions.gengamma.pdf(gengamma_x, gengamma_a, gengamma_c, gengamma_loc, gengamma_scale)
# plt.plot(gengamma_x, fitted_gengamma, label='generalised gamma')

# invgamma_a, invgamma_loc, invgamma_scale = stats.distributions.invgamma.fit(new_result)
# invgamma_x = np.linspace(0,1000,10000)
# fitted_invgamma = stats.distributions.invgamma.pdf(invgamma_x, invgamma_a, invgamma_loc, invgamma_scale)
# plt.plot(invgamma_x, fitted_invgamma, label='inverse gamma')

# betaprime_a, betaprime_b, betaprime_loc, betaprime_scale = stats.distributions.betaprime.fit(new_result)
# betaprime_x = np.linspace(0,1000,10000)
# fitted_betaprime = stats.distributions.betaprime.pdf(betaprime_x, betaprime_a, betaprime_b, betaprime_loc, betaprime_scale)
# plt.plot(betaprime_x, fitted_betaprime, label='betaprime')

# f_dfn, f_dfd, f_loc, f_scale = stats.distributions.f.fit(new_result)
# f_x = np.linspace(0,1000,10000)
# fitted_f = stats.distributions.f.pdf(f_x, f_dfn, f_dfd, f_loc, f_scale)
# plt.plot(f_x, fitted_f, label='f')

# fatiguelife_c, fatiguelife_loc, fatiguelife_scale = stats.distributions.fatiguelife.fit(new_result)
# fatiguelife_x = np.linspace(0,1000,10000)
# fitted_fatiguelife = stats.distributions.fatiguelife.pdf(fatiguelife_x, fatiguelife_c, fatiguelife_loc, fatiguelife_scale)
# plt.plot(fatiguelife_x, fitted_fatiguelife, label='Birnbaum–Saunders')

# genlogistic_c, genlogistic_loc, genlogistic_scale = stats.distributions.genlogistic.fit(new_result)
# genlogistic_x = np.linspace(0,1000,10000)
# fitted_genlogistic = stats.distributions.genlogistic.pdf(genlogistic_x, genlogistic_c, genlogistic_loc, genlogistic_scale)
# plt.plot(genlogistic_x, fitted_genlogistic, label='genlogistic')

# gumbel_r_loc, gumbel_r_scale  = stats.distributions.gumbel_r.fit(new_result)
# gumbel_r_x = np.linspace(0,1000,10000)
# fitted_gumbel_r = stats.distributions.gumbel_r.pdf(gumbel_r_x, gumbel_r_loc, gumbel_r_scale)
# plt.plot(gumbel_r_x, fitted_gumbel_r, label='gumbel_r')

# invweibull_c, invweibull_loc, invweibull_scale = stats.distributions.invweibull.fit(new_result)
# invweibull_x = np.linspace(0,1000,10000)
# fitted_invweibull = stats.distributions.invweibull.pdf(invweibull_x, invweibull_c, invweibull_loc, invweibull_scale)
# plt.plot(invweibull_x, fitted_invweibull, label='inverse weibull')

# nct_df, nct_nc, nct_loc, nct_scale = stats.distributions.nct.fit(new_result)
# nct_x = np.linspace(0,1000,10000)
# fitted_nct = stats.distributions.nct.pdf(nct_x, nct_df, nct_nc, nct_loc, nct_scale)
# plt.plot(nct_x, fitted_nct, label='non-central t')

# ncf_dfn, ncf_dfd, ncf_nc, ncf_loc, ncf_scale = stats.distributions.ncf.fit(new_result)
# ncf_x = np.linspace(0,1000,10000)
# fitted_ncf = stats.distributions.ncf.pdf(ncf_x, ncf_dfn, ncf_dfd, ncf_nc, ncf_loc, ncf_scale)
# plt.plot(ncf_x, fitted_ncf, label='non-central f')

# plt.yscale('log')



# sns.kdeplot(new_result, label='kde')

# def change_yscale(x, *args):
#     scale_y = len(new_result) / og_result_len
#     """
#     The function that will you be applied to your y-axis ticks.
#     """
#     x = float(x) * scale_y
#     # return "{:.3f}".format(x)
#     return x
# ax = plt.gca()
# ax.yaxis.set_major_formatter(mtick.FuncFormatter(change_yscale))



# burr_c, burr_d, burr_loc, burr_scale = stats.distributions.burr.fit(new_result)
# burr_x = np.linspace(0,1000,10000)
# fitted_burr = stats.distributions.burr.pdf(burr_x, burr_c, burr_d, burr_loc, burr_scale)
# plt.plot(burr_x, fitted_burr, label='burr')

# exponpow_b, exponpow_loc, exponpow_scale = stats.distributions.exponpow.fit(new_result)
# exponpow_x = np.linspace(0,1000,10000)
# fitted_exponpow = stats.distributions.exponpow.pdf(exponpow_x, exponpow_b, exponpow_loc, exponpow_scale)
# plt.plot(exponpow_x, fitted_exponpow, label='exponpow')

# exponweib_a, exponweib_c, exponweib_loc, exponweib_scale = stats.distributions.exponweib.fit(new_result)
# exponweib_x = np.linspace(0,1000,10000)
# fitted_exponweib = stats.distributions.exponweib.pdf(exponweib_x, exponweib_a, exponweib_c, exponweib_loc, exponweib_scale)
# plt.plot(exponweib_x, fitted_exponweib, label='exponweib')

# chi2_df, chi2_loc, chi2_scale = stats.distributions.chi2.fit(new_result)
# chi2_x = np.linspace(0,1000,10000)
# fitted_chi2 = stats.distributions.chi2.pdf(chi2_x, chi2_df, chi2_loc, chi2_scale)
# plt.plot(chi2_x, fitted_chi2, label='chi-squared')

# exponnorm_K, exponnorm_loc, exponnorm_scale = stats.distributions.exponnorm.fit(new_result)
# exponnorm_x = np.linspace(0,1000,10000)
# fitted_exponnorm = stats.distributions.exponnorm.pdf(exponnorm_x, exponnorm_K, exponnorm_loc, exponnorm_scale)
# plt.plot(exponnorm_x, fitted_exponnorm, label='exGaussian')

# beta_a, beta_b, beta_loc, beta_scale = stats.distributions.beta.fit(new_result)
# beta_x = np.linspace(0,1000,10000)
# fitted_beta = stats.distributions.beta.pdf(beta_x, beta_a, beta_b, beta_loc, beta_scale)
# plt.plot(beta_x, fitted_beta, label='beta')

# geninvgauss_p, geninvgauss_b, geninvgauss_loc, geninvgauss_scale = stats.distributions.geninvgauss.fit(new_result)
# geninvgauss_x = np.linspace(0,1000,10000)
# fitted_geninvgauss = stats.distributions.geninvgauss.pdf(geninvgauss_x, geninvgauss_p, geninvgauss_b, geninvgauss_loc, geninvgauss_scale)
# plt.plot(geninvgauss_x, fitted_geninvgauss, label='generalised inverse gaussian')

# norm_loc, norm_scale  = stats.distributions.norm.fit(new_result)
# norm_x = np.linspace(0,1000,10000)
# fitted_norm = stats.distributions.norm.pdf(norm_x, norm_loc, norm_scale)
# plt.plot(norm_x, fitted_norm, label='normal')

# t_loc, t_scale, df = stats.distributions.t.fit(new_result)
# t_x = np.linspace(0,1000,10000)
# fitted_t = stats.distributions.t.pdf(t_x, t_loc, t_scale, df)
# plt.plot(t_x, fitted_t, label='t')

# alpha_loc, alpha_scale, df = stats.distributions.alpha.fit(new_result)
# alpha_x = np.linspace(0,1000,10000)
# fitted_alpha = stats.distributions.alpha.pdf(alpha_x, alpha_loc, alpha_scale, df)
# plt.plot(alpha_x, fitted_alpha, label='alpha')

# genpareto_c, genpareto_loc, genpareto_scale = stats.distributions.genpareto.fit(new_result)
# genpareto_x = np.linspace(0,1000,10000)
# fitted_genpareto = stats.distributions.genpareto.pdf(genpareto_x, genpareto_c, genpareto_loc, genpareto_scale)
# plt.plot(genpareto_x, fitted_genpareto, label='genpareto')

# gilbrat_loc, gilbrat_scale  = stats.distributions.gilbrat.fit(new_result)
# gilbrat_x = np.linspace(0,1000,10000)
# fitted_gilbrat = stats.distributions.gilbrat.pdf(gilbrat_x, gilbrat_loc, gilbrat_scale)
# plt.plot(gilbrat_x, fitted_gilbrat, label='gilbrat')

# gompertz_c, gompertz_loc, gompertz_scale = stats.distributions.gompertz.fit(new_result)
# gompertz_x = np.linspace(0,1000,10000)
# fitted_gompertz = stats.distributions.gompertz.pdf(gompertz_x, gompertz_c, gompertz_loc, gompertz_scale)
# plt.plot(gompertz_x, fitted_gompertz, label='gompertz')

# gumbel_l_loc, gumbel_l_scale = stats.distributions.gumbel_l.fit(new_result)
# gumbel_l_x = np.linspace(0,1000,10000)
# fitted_gumbel_l = stats.distributions.gumbel_l.pdf(gumbel_l_x, gumbel_l_loc, gumbel_l_scale)
# plt.plot(gumbel_l_x, fitted_gumbel_l, label='gumbel_l')

# hypsecant_loc, hypsecant_scale = stats.distributions.hypsecant.fit(new_result)
# hypsecant_x = np.linspace(0,1000,10000)
# fitted_hypsecant = stats.distributions.hypsecant.pdf(hypsecant_x, hypsecant_loc, hypsecant_scale)
# plt.plot(hypsecant_x, fitted_hypsecant, label='hypsecant')

# laplace_loc, laplace_scale = stats.distributions.laplace.fit(new_result)
# laplace_x = np.linspace(0,1000,10000)
# fitted_laplace = stats.distributions.laplace.pdf(laplace_x, laplace_loc, laplace_scale)
# plt.plot(laplace_x, fitted_laplace, label='laplace')

# levy_loc, levy_scale = stats.distributions.levy.fit(new_result)
# levy_x = np.linspace(0,1000,10000)
# fitted_levy = stats.distributions.levy.pdf(levy_x, levy_loc, levy_scale)
# plt.plot(levy_x, fitted_levy, label='levy')

# levy_l_loc, levy_l_scale = stats.distributions.levy_l.fit(new_result)
# levy_l_x = np.linspace(0,1000,10000)
# fitted_levy_l = stats.distributions.levy_l.pdf(levy_l_x, levy_l_loc, levy_l_scale)
# plt.plot(levy_l_x, fitted_levy_l, label='levy_l')

# diverge
# levy_stable_alpha, levy_stable_beta, levy_stable_loc, levy_stable_scale = stats.distributions.levy_stable.fit(new_result)
# levy_stable_x = np.linspace(0,1000,10000)
# fitted_levy_stable = stats.distributions.levy_stable.pdf(levy_stable_x, levy_stable_alpha, levy_stable_beta, levy_stable_loc, levy_stable_scale)
# plt.plot(levy_stable_x, fitted_levy_stable, label='levy_stable')

# loggamma_c, loggamma_loc, loggamma_scale = stats.distributions.loggamma.fit(new_result)
# loggamma_x = np.linspace(0,1000,10000)
# fitted_loggamma = stats.distributions.loggamma.pdf(loggamma_x, loggamma_c, loggamma_loc, loggamma_scale)
# plt.plot(loggamma_x, fitted_loggamma, label='loggamma')

# loglaplace_c, loglaplace_loc, loglaplace_scale = stats.distributions.loglaplace.fit(new_result)
# loglaplace_x = np.linspace(0,1000,10000)
# fitted_loglaplace = stats.distributions.loglaplace.pdf(loglaplace_x, loglaplace_c, loglaplace_loc, loglaplace_scale)
# plt.plot(loglaplace_x, fitted_loglaplace, label='loglaplace')

# ncx2_df, ncx2_nc, ncx2_loc, ncx2_scale = stats.distributions.ncx2.fit(new_result)
# ncx2_x = np.linspace(0,1000,10000)
# fitted_ncx2 = stats.distributions.ncx2.pdf(ncx2_x, ncx2_df, ncx2_nc, ncx2_loc, ncx2_scale)
# plt.plot(ncx2_x, fitted_ncx2, label='non-central chi-squared')


# weibull_min_c, weibull_min_loc, weibull_min_scale = stats.distributions.weibull_min.fit(new_result)
# weibull_min_x = np.linspace(0,1000,10000)
# fitted_weibull_min = stats.distributions.weibull_min.pdf(weibull_min_x, weibull_min_c, weibull_min_loc, weibull_min_scale)
# plt.plot(weibull_min_x, fitted_weibull_min, label='Weibull')
