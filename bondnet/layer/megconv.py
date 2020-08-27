"""Torch modules for GAT for heterograph."""
import torch
from torch import nn
from dgl import function as fn
from bondnet.layer.utils import LinearN


class BondUpdateLayer(nn.Module):
    def __init__(
        self,
        master_node,
        attn_nodes,
        attn_edges,
        in_feats,
        out_feats,
        num_fc_layers=3,
        residual=False,
        activation=nn.Softplus(),
    ):

        super(BondUpdateLayer, self).__init__()
        self.master_node = master_node
        self.attn_nodes = attn_nodes
        self.edge_types = [(n, e, master_node) for n, e in zip(attn_nodes, attn_edges)]

        in_size = in_feats["bond"] + in_feats["atom"] * 2 + in_feats["global"]
        act = [activation] * (num_fc_layers - 1) + [nn.Identity()]
        use_bias = [True] * num_fc_layers
        self.fc_layers = LinearN(in_size, out_feats, act, use_bias)

    def forward(self, graph, master_feats, attn_feats):
        graph = graph.local_var()

        # assign feats
        graph.nodes[self.master_node].data.update({"ft": master_feats})
        for ntype, feats in zip(self.attn_nodes, attn_feats):
            graph.nodes[ntype].data.update({"ft": feats})

        for etype in self.edge_types:
            graph.update_all(
                fn.copy_u("ft", "m"), self.reduce_fn, self.apply_node_fn, etype=etype
            )

        feats = self.fc_layers(graph.nodes[self.master_node].data["ft"])

        return feats

    @staticmethod
    def reduce_fn(nodes):
        return {"m": torch.flatten(nodes.mailbox["m"], start_dim=1)}

    @staticmethod
    def apply_node_fn(nodes):
        return {"ft": torch.cat([nodes.data["ft"], nodes.data["m"]], dim=1)}


class AtomUpdateLayer(nn.Module):
    def __init__(
        self,
        master_node,
        attn_nodes,
        attn_edges,
        in_feats,
        out_feats,
        num_fc_layers=3,
        residual=False,
        activation=nn.Softplus(),
    ):

        super(AtomUpdateLayer, self).__init__()
        self.master_node = master_node
        self.attn_nodes = attn_nodes
        self.edge_types = [(n, e, master_node) for n, e in zip(attn_nodes, attn_edges)]

        in_size = in_feats["atom"] + in_feats["bond"] + in_feats["global"]
        act = [activation] * (num_fc_layers - 1) + [nn.Identity()]
        use_bias = [True] * num_fc_layers
        self.fc_layers = LinearN(in_size, out_feats, act, use_bias)

    def forward(self, graph, master_feats, attn_feats):
        graph = graph.local_var()

        # assign feats
        graph.nodes[self.master_node].data.update({"ft": master_feats})
        for ntype, feats in zip(self.attn_nodes, attn_feats):
            graph.nodes[ntype].data.update({"ft": feats})

        # for et in self.edge_types:
        #     graph.update_all(
        #         fn.copy_u("ft", "m"), fn.mean("m", "mean"), self.apply_node_fn, etype=et
        #     )
        # feats = self.fc_layers(graph.nodes[self.master_node].data["ft"])

        # dgl gives the below error:
        #
        # RuntimeError: one of the variables needed for gradient computation has been
        # modified by an inplace operation: [torch.FloatTensor [991, 32]],  which is
        # output 0 of AddmmBackward, is at version 1; expected version 0 instead.
        # Hint: enable anomaly detection to find the operation that failed to compute
        # its gradient, with torch.autograd.set_detect_anomaly(True).
        #
        # when using the above commented block.
        # Cannot easily create a small snippet to reproduce it. So for now, use the below
        # instead.
        #

        for et in self.edge_types:
            graph.update_all(
                self.msg_fn, fn.mean("m", "mean"), self.apply_node_fn, etype=et
            )
        feats = self.fc_layers(graph.nodes[self.master_node].data["ft"])

        return feats

    @staticmethod
    def msg_fn(edges):
        return {"m": edges.src["ft"]}

    @staticmethod
    def apply_node_fn(nodes):
        return {"ft": torch.cat([nodes.data["ft"], nodes.data["mean"]], dim=1)}


GlobalUpdateLayer = AtomUpdateLayer


class MEGConv(nn.Module):
    """
    Graph attention convolution layer for hetero graph that attends between different
    (and the same) type of nodes.

    Args:
        attn_mechanism (dict of dict): The attention mechanism, i.e. how the node
            features will be updated. The outer dict has `node types` as its key
            and the inner dict has keys `nodes` and `edges`, where the values (list)
            of `nodes` are the `node types` that the master node will attend to,
            and the corresponding `edges` are the `edge types`.
        attn_order (list): `node type` string that specify the order to attend the node
            features.
        in_feats (list): input feature size for the corresponding (w.r.t. index) node
            in `attn_order`.
        out_feats (int): output feature size, the same for all nodes
        num_heads (int): number of attention heads, the same for all nodes
        num_fc_layers (int): number of fully-connected layer before attention
        feat_drop (float, optional): [description]. Defaults to 0.0.
        attn_drop (float, optional): [description]. Defaults to 0.0.
        negative_slope (float, optional): [description]. Defaults to 0.2.
        residual (bool, optional): [description]. Defaults to False.
        batch_norm(bool): whether to apply batch norm to the output
        activation (nn.Moldule or str): activation fn
    """

    def __init__(
        self,
        attn_mechanism,
        attn_order,
        in_feats,
        out_feats,
        num_fc_layers=3,
        residual=False,
        activation=None,
        first_block=False,
    ):

        super(MEGConv, self).__init__()

        self.attn_mechanism = attn_mechanism
        self.master_nodes = attn_order

        self.residual = residual

        in_feats_map = dict(zip(attn_order, in_feats))

        # linear fc
        self.linear_fc = nn.ModuleDict()
        for ntype in self.master_nodes:
            if first_block:
                in_size = in_feats_map[ntype]
                out_sizes = [out_feats[0], out_feats[-1]]
                act = [activation] * 2
                use_bias = [True] * 2
                self.linear_fc[ntype] = LinearN(in_size, out_sizes, act, use_bias)
            else:
                self.linear_fc[ntype] = nn.Identity()

        in_size = {k: out_feats[-1] for k in in_feats_map}
        self.layers = nn.ModuleDict()
        for ntype in self.master_nodes:
            if ntype == "bond":
                self.layers[ntype] = BondUpdateLayer(
                    master_node=ntype,
                    attn_nodes=self.attn_mechanism[ntype]["nodes"],
                    attn_edges=self.attn_mechanism[ntype]["edges"],
                    in_feats=in_size,
                    out_feats=out_feats,
                    num_fc_layers=num_fc_layers,
                    residual=residual,
                    activation=activation,
                )
            elif ntype == "atom":
                self.layers[ntype] = AtomUpdateLayer(
                    master_node=ntype,
                    attn_nodes=self.attn_mechanism[ntype]["nodes"],
                    attn_edges=self.attn_mechanism[ntype]["edges"],
                    in_feats=in_size,
                    out_feats=out_feats,
                    num_fc_layers=num_fc_layers,
                    residual=residual,
                    activation=activation,
                )
            elif ntype == "global":
                self.layers[ntype] = GlobalUpdateLayer(
                    master_node=ntype,
                    attn_nodes=self.attn_mechanism[ntype]["nodes"],
                    attn_edges=self.attn_mechanism[ntype]["edges"],
                    in_feats=in_size,
                    out_feats=out_feats,
                    num_fc_layers=num_fc_layers,
                    residual=residual,
                    activation=activation,
                )

    def forward(self, graph, feats):
        """
        Args:
            graph (dgl heterograph): the graph
            feats (dict): node features with node type as key and the corresponding
            features as value.

        Returns:
            dict: updated node features with the same keys as in `feats`.
                Each feature value has a shape of `(N, out_feats*num_heads)`, where
                `N` is the number of nodes (different for different key) and
                `out_feats` and `num_heads` are the out feature size and number
                of heads specified at instantiation (the same for different keys).
        """
        feats_in = {k: v for k, v in feats.items()}

        feats_linear_fc = dict()
        for ntype in self.master_nodes:
            feats_linear_fc[ntype] = self.linear_fc[ntype](feats_in[ntype])

        updated_feats = {k: v for k, v in feats_linear_fc.items()}
        for ntype in self.master_nodes:
            master_feats = updated_feats[ntype]
            attn_feats = [updated_feats[t] for t in self.attn_mechanism[ntype]["nodes"]]
            updated_feats[ntype] = self.layers[ntype](graph, master_feats, attn_feats)

        # residual
        if self.residual:
            for k in updated_feats:
                updated_feats[k] += feats_linear_fc[k]

        return updated_feats
