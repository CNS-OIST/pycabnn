# %%
import pds_plots as ppl

# %%

ppl.plot_mf_1(mf_points, [1500, 750], 4, save=True)
ppl.plot_mf_2(mf_points, [1500, 750], save=True)


# %%
ppl.plot_goc(goc_points, [1500, 750], 100, 16, save=True)

ppl.plot_glo(glo_points, [1500, 750], 50, 3.3)

# %%
ppl.plot_goc_glo((goc_points, 15), (glo_points, 6.6 / 2), [1500, 750], 100, save=True)
plt.xlim([25, 425])
plt.ylim([25, 225])

# %%
ppl.plot_goc_glo((glo_points, 6.6 / 1.75), (grc_points, 3), [1500, 750], 100)
plt.xlim([25, 425])
plt.ylim([25, 225])

# %%
ppl.plot_all_pop(
    (goc_points, 17),
    (glo_points, 6.6 / 1.75),
    (grc_points, 3),
    [1500, 750],
    100,
    save=True,
)
plt.xlim([25, 425])
plt.ylim([25, 225])

