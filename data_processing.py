import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from pymatgen.symmetry.groups import SpaceGroup, sg_symbol_from_int_number
import json


def plot_ccs_stats(ccs_df: pd.DataFrame) -> None:
    """
    Plot the number of structures for each space group and dopant content.
    :param ccs_df: complete CCS. For example, CCS of Cd-substituted δ-CsPbI3 contained in data/CCS_yellow_Cd.pkl.gz.
    :return: None.
    """
    ccs_df = ccs_df[["Dopant_content", "Br_content", "Space_group_no"]]
    numbers_df = ccs_df.groupby(["Dopant_content", "Br_content", "Space_group_no"]).size().to_frame()
    numbers_df = numbers_df.rename(mapper={0: "Number_of_structures"}, axis=1).reset_index()
    numbers_df = numbers_df.pivot(columns=["Dopant_content", "Br_content"], index="Space_group_no",
                                  values="Number_of_structures")
    numbers_df = numbers_df.sort_index(ascending=False).T
    log_max_number = np.log(np.nanmax(numbers_df.values)).item()
    dopant_contents = sorted(ccs_df["Dopant_content"].unique().tolist())

    with open("venv/Lib/site-packages/pymatgen/symmetry/symm_data.json", "r") as f:
        symm_data = json.load(f)
    symm_data_abbr = {v: k for k, v in symm_data["abbreviated_spacegroup_symbols"].items()}

    plt.rc("font", size=8)
    cmap = "YlOrRd"
    fig, ax = plt.subplots(len(dopant_contents), 1, figsize=(16, 8), sharex=True)
    for i, dopant_content in enumerate(dopant_contents):
        num = numbers_df.loc[(dopant_content, )]
        sns.heatmap(np.log(num), cmap=cmap, cbar=False, annot=num, vmin=0, vmax=log_max_number, linewidth=.1, fmt='g',
                    xticklabels=[f"{symm_data_abbr.get(sg_symbol_from_int_number(sg), sg_symbol_from_int_number(sg))}\n({sg})" for sg in num.columns],
                    yticklabels=[str(round(x, 1)) for x in num.index],
                    ax=ax[i])
        ax[i].set_xlabel(None)
        ax[i].set_ylabel("I substituted, %", fontsize=10)
        ax[i].set_title(f"{dopant_content} % Pb substituted", fontsize=10, fontweight="bold")
    ax[-1].set_xlabel("Space group", fontsize=10)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=log_max_number))
    sm.set_array([])
    cbar = plt.colorbar(sm, aspect=70, ax=ax)
    cbar.ax.set_ylabel("log (number of inequivalent structures)", fontsize=10)
    cbar.outline.set_visible(False)
    # plt.savefig("fig_plot_ccs_stats.jpg", bbox_inches="tight", dpi=300)
    plt.show()


def plot_group_subgroup_graph(ccs_df: pd.DataFrame) -> None:
    """
    Plot the group-subgroup graph.
    :param ccs_df: complete CCS. For example, CCS of Cd-substituted δ-CsPbI3 contained in data/CCS_yellow_Cd.pkl.gz.
    :return: None.
    """
    with open("venv/Lib/site-packages/pymatgen/symmetry/symm_data.json", "r") as f:
        symm_data = json.load(f)
    symm_data_subg = symm_data["maximal_subgroups"]
    symm_data_abbr = {v: k for k, v in symm_data["abbreviated_spacegroup_symbols"].items()}

    sgs = sorted(ccs_df["Space_group_no"].unique(), reverse=True)
    sg_info = {sg: ((ccs_df["Space_group_no"] == sg).sum(),
                    symm_data_abbr.get(SpaceGroup.from_int_number(sg).symbol, SpaceGroup.from_int_number(sg).symbol))
               for sg in sgs}
    sg_info2 = {i[1]: i[0] for i in sg_info.values()}
    label_map = {v[1]: v[1] + f"\n({k})" for k, v in sg_info.items()}

    graph = nx.DiGraph()
    for i in range(len(sgs)):
        for j in range(len(sgs)):
            if sgs[j] in symm_data_subg[str(sgs[i])] and i != j:
                graph.add_edge(sg_info[sgs[i]][1], sg_info[sgs[j]][1])

    not_connected_nodes = set(graph.nodes) - set([i[1] for i in graph.edges])
    for node2 in not_connected_nodes:
        for node1 in graph.nodes:
            if SpaceGroup(node2).is_subgroup(SpaceGroup(node1)):
                graph.add_edge(node1, node2)
                break

    nodes = [i for i in graph.nodes]
    orders = np.array([SpaceGroup(nodes[i]).order for i in range(len(nodes))])
    pos_x = [0] * len(nodes)
    unique, counts = np.unique(orders, return_counts=True)
    for count_pos in range(len(counts)):
        for i in range(counts[count_pos]):
            pos_x[np.where(orders == unique[count_pos])[0][i]] = (i + 1) / (counts[count_pos] + 1)
    pos = {nodes[i]: (pos_x[i], orders[i]) for i in range(len(nodes))}

    edges_curved = {("Pc", "P1"), ("P-1", "P1")}  # It can be happened that some edges are not shown because of
    # overlapping. One can curve them manually to avoid this.
    edges_straight = set(graph.edges) - edges_curved

    fig, ax = plt.subplots(figsize=(6, 7))
    cmap = "YlOrRd"
    nx.draw_networkx_nodes(graph, pos, node_color=[np.log(sg_info2[i]) for i in graph.nodes], node_size=1200,
                           edgecolors="black", linewidths=1, cmap=cmap, vmin=0,
                           vmax=np.log(max([i for i in sg_info2.values()])), ax=ax)
    nx.draw_networkx_labels(graph, pos, labels=label_map, font_size=8, font_color="black")
    nx.draw_networkx_edges(graph, pos, edgelist=edges_straight, edge_color="grey", node_size=1200, width=1,
                           arrowsize=12, ax=ax)
    nx.draw_networkx_edges(graph, pos, edgelist=edges_curved, edge_color="grey", width=1, node_size=1200,
                           arrowsize=12, connectionstyle='arc3, rad = -0.1', ax=ax)
    ax.tick_params(left=True, labelleft=True)
    ax.set_ylabel("Space group order", fontsize=12)
    ax.set_yticks(orders)
    ax.yaxis.set_tick_params(labelsize=12)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=np.log(max([i for i in sg_info2.values()]))))
    sm.set_array([])
    cbar = plt.colorbar(sm, aspect=70)
    cbar.ax.set_ylabel("log (number of inequivalent structures)", fontsize=12)
    cbar.ax.yaxis.set_tick_params(labelsize=12)
    cbar.outline.set_visible(False)
    plt.tight_layout()
    plt.show()


def plot_energy_distribution(data_df: pd.DataFrame) -> None:
    """
    Plot the distributions of DFT-derived formation energies for PHS and PLS structures
    within the training/validation and test datasets.
    :param data_df: Slice of complete CCSs including DFT-calculated structures only.
    :return: None.
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    phs_tr_val = data_df.loc[data_df["PHS_train"] | data_df["PHS_val"], "Formation_energy_pa"]
    pls_tr_val = data_df.loc[data_df["PLS_train"] | data_df["PLS_val"], "Formation_energy_pa"]
    bins = np.arange(min(phs_tr_val.min(), pls_tr_val.min()), max(phs_tr_val.max(), pls_tr_val.max()) + 0.01, 0.01)
    ax[0].hist([pls_tr_val, phs_tr_val],
               bins=bins, label=["PLS structures", "PHS structures"], color=["blue", "orange"])
    ax[0].set_title("Train/validation datasets", fontsize=12)
    ax[0].set_xlabel("Formation energy, eV/atom", fontsize=12)
    ax[0].set_ylabel("Number of unique structures", fontsize=12)
    ax[0].yaxis.set_tick_params(labelsize=12)
    ax[0].grid(linestyle="--", color="lightgray", which="both")
    ax[0].set_axisbelow(True)
    ax[0].legend()

    pls_test = data_df.loc[data_df["PLS_test"], "Formation_energy_pa"]
    phs_test = data_df.loc[data_df["PHS_test"], "Formation_energy_pa"]
    ax[1].hist([pls_test, phs_test], bins=bins, label=["PLS structures", "PHS structures"], color=["blue", "orange"])
    ax[1].set_title("Test datasets", fontsize=12)
    ax[1].set_xlabel("Formation energy, eV/atom", fontsize=12)
    ax[1].set_ylabel("Number of unique structures", fontsize=12)
    ax[1].yaxis.set_tick_params(labelsize=12)
    ax[1].grid(linestyle="--", color="lightgray", which="both")
    ax[1].set_axisbelow(True)
    ax[1].legend()
    fig.tight_layout()
    plt.show()


def plot_weight_vs_group(ccs_black_cd_df: pd.DataFrame, ccs_yellow_cd_df: pd.DataFrame) -> None:
    """
    Plot the group-subgroup graph.
    :param ccs_black_cd_df: complete CCS of Cd-substituted γ-CsPbI3.
    :param ccs_yellow_cd_df: complete CCS of Cd-substituted δ-CsPbI3.
    :return: None.
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].scatter(ccs_black_cd_df["Space_group_no"], ccs_black_cd_df["Weight"], color="black")
    ax[0].set_title("Black γ-CsPbI\N{SUBSCRIPT THREE}", fontsize=12)
    ax[0].set_xlabel("Space group number", fontsize=12)
    ax[0].set_ylabel("Weight", fontsize=12)
    ax[0].yaxis.set_tick_params(labelsize=12)
    ax[0].grid(linestyle="--", color="lightgray", which="both")
    ax[0].set_axisbelow(True)

    ax[1].scatter(ccs_yellow_cd_df["Space_group_no"], ccs_yellow_cd_df["Weight"], color="orange")
    ax[1].set_title("Yellow δ-CsPbI\N{SUBSCRIPT THREE}", fontsize=12)
    ax[1].set_xlabel("Space group number", fontsize=12)
    ax[1].set_ylabel("Weight", fontsize=12)
    ax[1].yaxis.set_tick_params(labelsize=12)
    ax[1].grid(linestyle="--", color="lightgray", which="both")
    ax[1].set_axisbelow(True)
    fig.tight_layout()
    plt.show()


DATA_DIR = "data"
ccs_yellow_cd_df = pd.read_pickle(os.path.join(DATA_DIR, "CCS_yellow_Cd.pkl.gz"))
ccs_black_cd_df = pd.read_pickle(os.path.join(DATA_DIR, "CCS_black_Cd.pkl.gz"))
plot_weight_vs_group(ccs_black_cd_df, ccs_yellow_cd_df)
plot_ccs_stats(ccs_yellow_cd_df)
plot_group_subgroup_graph(ccs_yellow_cd_df)

dft_df = []
for filename in os.listdir(DATA_DIR):
    temp_df = pd.read_pickle(os.path.join(DATA_DIR, filename))
    temp_df = temp_df.loc[temp_df["Formation_energy_pa"].notna(), "Formation_energy_pa": "PLS_test"]
    dft_df.append(temp_df)
dft_df = pd.concat(dft_df, axis=0, ignore_index=True)
plot_energy_distribution(dft_df)
