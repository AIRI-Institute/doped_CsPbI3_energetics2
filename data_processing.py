import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from pymatgen.symmetry.groups import SpaceGroup
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
                    xticklabels=[f"{symm_data_abbr.get(SpaceGroup.from_int_number(sg).symbol, SpaceGroup.from_int_number(sg).symbol)}\n({sg})" for sg in num.columns],
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
    cmap = plt.cm.YlOrRd
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


def plot_energy_distribution(ccs_df: pd.DataFrame) -> None:
    pass


def plot_weight_vs_group(ccs_df: pd.DataFrame) -> None:
    pass


def plot_inference(ccs_df: pd.DataFrame) -> None:
    pass


data_df = pd.read_pickle("data/CCS_yellow_Cd.pkl.gz")
plot_ccs_stats(data_df)
plot_group_subgroup_graph(data_df)
